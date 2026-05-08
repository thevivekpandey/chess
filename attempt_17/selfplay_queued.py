#!/usr/bin/env python3
"""
Ultra-High Performance Zero-Copy Shared Memory Inference Server.
Contiguous Memory Mapping + Vectorized MCTS for H100.
"""

import argparse
import csv
import torch.multiprocessing as mp
import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn
import numpy as np
import torch

from chess_engine import ChessNet, boards_to_tensors, MOVE_FLAT_INDEX_TABLE
from play_games_mcts import (
    MCTSNode,
    _position_key,
    _terminal_value_given,
    terminal_score_white,
    _check_repetition_terminal,
)
from selfplay_generator import (
    sample_move_with_temperature,
    assign_game_outcome_to_examples,
    save_training_examples_to_csv,
)

# =============================================================================
# Inference Server (Contiguous DMA)
# =============================================================================

def inference_server_process(
    model_path: str,
    device_str: str,
    shared_in: torch.Tensor,
    shared_val: torch.Tensor,
    shared_pol: torch.Tensor,
    request_queue: mp.Queue,
    response_events: List[mp.Event],
    max_batch_size: int,
    num_workers: int,
    worker_batch_size: int,
    stop_event: mp.Event,
):
    device = torch.device(device_str)
    model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=16)
    
    print(f"[Server] Loading model on {device}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Pre-allocate GPU buffer for fixed-size DMA
    gpu_in = torch.zeros(max_batch_size, 18, 8, 8, device=device)
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # Note: We skip torch.compile here to ensure baseline stability, 
        # but you can re-enable it for another 20% boost once this works.
        # model = torch.compile(model, mode='reduce-overhead')

    print(f"[Server] Contiguous Inference Server Ready.")

    total_items = 0
    last_stat_time = time.time()
    inf_time_acc = 0.0

    while not stop_event.is_set():
        batch_requests = []
        try:
            # Short timeout to keep latency low but batching high
            req = request_queue.get(timeout=0.005) 
            batch_requests.append(req)
        except:
            continue

        # Drain queue as much as possible up to max_batch_size
        curr_n = batch_requests[0][1]
        while curr_n < max_batch_size:
            try:
                req = request_queue.get_nowait()
                batch_requests.append(req)
                curr_n += req[1]
            except:
                break
        
        t_loop_start = time.time()
        
        # Optimization: Sequential contiguous copies are much faster than indexed gather
        curr = 0
        active_workers = []
        for worker_id, n in batch_requests:
            if curr + n > max_batch_size: break # Simple cap
            
            # This is a contiguous slice copy - very efficient
            gpu_in[curr : curr + n].copy_(shared_in[worker_id, :n], non_blocking=True)
            active_workers.append((worker_id, n, curr))
            curr += n
        
        if curr == 0: continue

        # Forward Pass
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == 'cuda' else torch.inference_mode():
            values, policies = model(gpu_in[:curr])
        
        # Vectorized distribution back to shared memory
        v_cpu = values.float().cpu().view(-1)
        p_cpu = policies.float().cpu()
        
        for worker_id, n, offset in active_workers:
            shared_val[worker_id, :n].copy_(v_cpu[offset : offset + n])
            shared_pol[worker_id, :n].copy_(p_cpu[offset : offset + n])
            response_events[worker_id].set()
            
        inf_time_acc += (time.time() - t_loop_start)
        total_items += curr

        if time.time() - last_stat_time > 10:
            elapsed = time.time() - last_stat_time
            print(f"[Server] IPS: {total_items/elapsed:.1f}, Util: {inf_time_acc/elapsed:.1%}, Batch: {curr}")
            total_items = 0; last_stat_time = time.time(); inf_time_acc = 0

# =============================================================================
# MCTS Client (Vectorized)
# =============================================================================

class FastZeroCopyMCTSEngine:
    def __init__(self, worker_id, sh_in, sh_val, sh_pol, req_q, resp_event):
        self.worker_id = worker_id
        self.shared_in = sh_in[worker_id]
        self.shared_val = sh_val[worker_id]
        self.shared_pol = sh_pol[worker_id]
        self.req_q = req_q
        self.resp_event = resp_event

    def _forward_batch(self, boards, need_value, need_policy):
        n = len(boards)
        boards_to_tensors(boards, self.shared_in[:n])
        self.resp_event.clear()
        self.req_q.put((self.worker_id, n))
        self.resp_event.wait()
        return self.shared_val[:n].numpy() if need_value else None, self.shared_pol[:n].numpy() if need_policy else None

    def make_root(self, board): return MCTSNode(board=board.copy(stack=False))
    
    def advance_root(self, root, move, history):
        if move in root.children and root.children[move]:
            new_root = root.children[move]
        else:
            new_board = root.board.copy(stack=False); new_board.push(move)
            new_root = MCTSNode(board=new_board)
        new_root.parent = None; new_root.parent_move = None
        intrinsic = terminal_score_white(new_root.board)
        if intrinsic is not None: new_root.is_terminal = True; new_root.terminal_value = intrinsic
        else: _check_repetition_terminal(new_root, Counter(history[:-1]), {})
        return new_root

    def _expand_node(self, node, policy_logits_np):
        legal = list(node.board.legal_moves)
        if not legal: return
        
        flat_logits = policy_logits_np.reshape(-1)
        # Vectorized lookup of priors
        indices = [MOVE_FLAT_INDEX_TABLE[m.from_square, m.to_square, m.promotion or 0] for m in legal]
        indices = np.array(indices, dtype=np.int32)
        
        # Filter valid indices and compute softmax in one go
        valid_mask = indices >= 0
        priors = np.zeros(len(legal), dtype=np.float32)
        if valid_mask.any():
            logits = flat_logits[indices[valid_mask]].astype(np.float64)
            logits -= np.max(logits)
            probs = np.exp(logits)
            priors[valid_mask] = (probs / probs.sum()).astype(np.float32)
        else:
            priors.fill(1.0 / len(legal))
            
        node.child_moves = legal
        node.child_priors = priors
        node.child_N = np.zeros(len(legal), dtype=np.int32)
        node.child_W = np.zeros(len(legal), dtype=np.float32)
        node.child_vlosses = np.zeros(len(legal), dtype=np.int32)
        node.child_nodes = [None] * len(legal); node.children = {m: None for m in legal}
        node.expanded = True

    def _select_child_idx(self, node, cpuct, fpu):
        is_w = (node.turn == chess.WHITE)
        eff_N = node.child_N + node.child_vlosses
        sqrt_total = np.sqrt(eff_N.sum()) if eff_N.sum() > 0 else 1.0
        pqw = node.W / node.N if node.N > 0 else 0.0
        q_mover = np.full(len(eff_N), (pqw-fpu) if is_w else -(pqw+fpu), dtype=np.float32)
        mask = eff_N > 0
        if mask.any():
            vls = -1.0 if is_w else 1.0
            q_mover[mask] = (node.child_W[mask] + node.child_vlosses[mask]*vls) / eff_N[mask]
            if not is_w: q_mover[mask] *= -1
        return np.argmax(q_mover + (cpuct * sqrt_total) * node.child_priors / (1 + eff_N))

    def _descend(self, root, cpuct, fpu, history_counts, path_counts):
        node = root; path = [node]
        while node.expanded and not node.is_terminal:
            idx = self._select_child_idx(node, cpuct, fpu)
            child = node.child_nodes[idx]
            if not child:
                child = MCTSNode(board=None, parent=node, parent_move=node.child_moves[idx], prior=node.child_priors[idx])
                child.index_in_parent = idx; node.child_nodes[idx] = child; node.children[node.child_moves[idx]] = child
            if not child.board:
                child.board = node.board.copy(stack=False); child.board.push(child.parent_move)
                term = terminal_score_white(child.board)
                if term is not None: child.is_terminal = True; child.terminal_value = term
                else: _check_repetition_terminal(child, history_counts, path_counts)
            node = child; path.append(node)
            if not node.is_terminal:
                key = _position_key(node.board); path_counts[key] = path_counts.get(key, 0) + 1
        return node, path

    def choose_move(self, root, n_sims, batch_size, cpuct, fpu, history, early_exit=0):
        h_counts = Counter(history)
        if not root.expanded:
            _, p = self._forward_batch([root.board], False, True); self._expand_node(root, p[0])
        n_done = 0
        while n_done < n_sims:
            if early_exit > 0 and n_done >= early_exit:
                n_arr = root.child_N
                if len(n_arr) >= 2:
                    top_two = np.partition(n_arr, -2)[-2:]
                    if top_two[1] - top_two[0] > (n_sims - n_done): break
            target = min(batch_size, n_sims - n_done)
            batch_leaves, batch_paths = [], []
            while len(batch_leaves) < target:
                leaf, path = self._descend(root, cpuct, fpu, h_counts, {})
                if leaf.is_terminal:
                    for n in path:
                        n.N += 1; n.W += leaf.terminal_value
                        if n.parent: n.parent.child_N[n.index_in_parent] = n.N; n.parent.child_W[n.index_in_parent] = n.W
                    n_done += 1; continue
                for n in path:
                    n.virtual_loss += 1
                    if n.parent: n.parent.child_vlosses[n.index_in_parent] = n.virtual_loss
                batch_leaves.append(leaf); batch_paths.append(path)
            if not batch_leaves: continue
            unique, leaf_map = [], {}
            for l in batch_leaves:
                if id(l) not in leaf_map: leaf_map[id(l)] = len(unique); unique.append(l)
            v_arr, p_arr = self._forward_batch([l.board for l in unique], True, True)
            for i, l in enumerate(unique):
                if not l.expanded: self._expand_node(l, p_arr[i])
            for l, path in zip(batch_leaves, batch_paths):
                v_white = float(v_arr[leaf_map[id(l)]])
                for n in path:
                    n.virtual_loss -= 1; n.N += 1; n.W += v_white
                    if n.parent:
                        n.parent.child_vlosses[n.index_in_parent] = n.virtual_loss
                        n.parent.child_N[n.index_in_parent] = n.N; n.parent.child_W[n.index_in_parent] = n.W
                n_done += 1
        n_arr = root.child_N; best_idx = np.argmax(n_arr)
        qw = np.where(n_arr > 0, root.child_W / np.maximum(n_arr, 1), 0.0)
        top = [(root.child_moves[i], int(n_arr[i]), float(root.child_priors[i]), float(qw[i])) for i in np.argsort(-n_arr)[:10]]
        return root.child_moves[best_idx], {'top_moves': top}

# =============================================================================
# Main
# =============================================================================

def selfplay_worker(worker_id, sh_in, sh_val, sh_pol, req_q, resp_event, num_games, output_csv, play_params, base_seed, shared_moves):
    random.seed((base_seed + worker_id * 100003) & 0xFFFFFFFF); np.random.seed(random.getrandbits(32))
    engine = FastZeroCopyMCTSEngine(worker_id, sh_in, sh_val, sh_pol, req_q, resp_event)
    for i in range(num_games):
        start = time.time(); board = chess.Board(); mcts_root = engine.make_root(board)
        history, examples, ply = [_position_key(board)], [], 0
        while not board.is_game_over(claim_draw=True) and ply < play_params['max_plies']:
            if ply > 0 and play_params['reuse_tree']: mcts_root = engine.advance_root(mcts_root, last_move, history)
            else: mcts_root = engine.make_root(board)
            mv, stats = engine.choose_move(mcts_root, play_params['n_simulations'], play_params['mcts_batch_size'], play_params['cpuct'], play_params['fpu_reduction'], history, play_params['early_exit_min_sims'])
            mv_visits = {mv: v for mv, v, _, _ in stats['top_moves']}; legal = list(board.legal_moves)
            v_counts = [mv_visits.get(mv, 0) for mv in legal]
            examples.append({'fen': board.fen(), 'move_uci_list': [mv.uci() for mv in legal], 'visit_counts': v_counts, 'turn': board.turn})
            move = sample_move_with_temperature(legal, v_counts, 1.0 if ply < play_params['temperature_moves'] else 0.0)
            board.push(move); last_move = move; history.append(_position_key(board)); ply += 1
            with shared_moves.get_lock(): shared_moves.value += 1
        res = board.result(claim_draw=True); examples = assign_game_outcome_to_examples(examples, res)
        save_training_examples_to_csv(examples, output_csv, f"w{worker_id}_g{i}")
        elap = time.time() - start; print(f"[W{worker_id}] G{i+1}: {res}, {ply} plies, {elap:.1f}s, {ply/elap:.1f}m/s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True); parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=64); parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--output", default="selfplay_data.csv")
    args = parser.parse_args()
    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass
    ctx = mp.get_context("spawn")
    
    # Contiguous pinned memory for zero-latency DMA
    sh_in = torch.zeros(args.workers, args.batch_size, 18, 8, 8).pin_memory().share_memory_()
    sh_val = torch.zeros(args.workers, args.batch_size).pin_memory().share_memory_()
    sh_pol = torch.zeros(args.workers, args.batch_size, 73, 8, 8).pin_memory().share_memory_()
    
    req_q = ctx.Queue(maxsize=args.workers * 2); resp_evs = [ctx.Event() for _ in range(args.workers)]; stop = ctx.Event()
    shared_moves = mp.Value('L', 0)
    
    server = ctx.Process(target=inference_server_process, args=(args.model, args.device, sh_in, sh_val, sh_pol, req_q, resp_evs, args.max_batch_size, args.workers, args.batch_size, stop))
    server.start()
    
    params = {'n_simulations': args.simulations, 'mcts_batch_size': args.batch_size, 'cpuct': 2.0, 'fpu_reduction': 0.0, 'reuse_tree': True, 'max_plies': 300, 'temperature_moves': 30, 'early_exit_min_sims': 200}
    base = int(time.time()); g_per_w = (args.games + args.workers - 1) // args.workers
    
    start_time = time.time()
    workers = [ctx.Process(target=selfplay_worker, args=(i, sh_in, sh_val, sh_pol, req_q, resp_evs[i], g_per_w, args.output, params, base, shared_moves)) for i in range(args.workers)]
    for w in workers: w.start()
    
    try:
        last_log = time.time()
        while any(w.is_alive() for w in workers):
            time.sleep(5)
            if time.time() - last_log >= 60:
                elap = time.time() - start_time; mv = shared_moves.value
                print(f"\n[CUMULATIVE STATS] Elapsed: {elap/60:.1f}m, Moves: {mv:,}, Throughput: {mv/elap:.1f} moves/s")
                last_log = time.time()
        for w in workers: w.join()
    except KeyboardInterrupt: pass
    
    total_elap = time.time() - start_time; total_mv = shared_moves.value
    print(f"\nFINAL STATISTICS\nTotal Time: {total_elap:.1f}s, Moves: {total_mv:,}, Throughput: {total_mv/total_elap:.1f} moves/s")

    stop.set()
    server.terminate(); server.join(5)

if __name__ == "__main__": main()
