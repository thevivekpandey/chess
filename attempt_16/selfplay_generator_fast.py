#!/usr/bin/env python3
"""
Fast Multi-Process Self-Play Generator (Nuance-Preserving).
Optimized for H100 + 24 CPUs.

Cloned from selfplay_generator.py with surgical optimizations:
1. Vectorized bit-unpacking for board encoding.
2. bfloat16 inference for H100.
3. Periodic cumulative performance logging.
"""

import argparse
import csv
import multiprocessing as mp
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn
import numpy as np
import torch

# Import MCTS engine from play_games_mcts
from play_games_mcts import (
    MCTSEngine,
    MCTSNode,
    _position_key,
)

# =============================================================================
# SURGICAL OPTIMIZATION: Vectorized Board Encoding
# =============================================================================

def fast_boards_to_tensors(boards: List[chess.Board], buffer: torch.Tensor):
    """
    Ultra-optimized board encoding using NumPy bit-unpacking.
    Avoids expensive property calls in python-chess loop.
    """
    n = len(boards)
    # Using numpy view of the torch tensor for direct write
    buf_np = buffer.numpy()
    
    # 1. Piece planes (0-11)
    # Extract bitboards in bulk
    bbs = np.zeros((n, 12), dtype=np.uint64)
    for i, b in enumerate(boards):
        occ = b.occupied_co
        w = occ[chess.WHITE]
        bl = occ[chess.BLACK]
        bbs[i, 0] = w & b.pawns
        bbs[i, 1] = w & b.knights
        bbs[i, 2] = w & b.bishops
        bbs[i, 3] = w & b.rooks
        bbs[i, 4] = w & b.queens
        bbs[i, 5] = w & b.kings
        bbs[i, 6] = bl & b.pawns
        bbs[i, 7] = bl & b.knights
        bbs[i, 8] = bl & b.bishops
        bbs[i, 9] = bl & b.rooks
        bbs[i, 10] = bl & b.queens
        bbs[i, 11] = bl & b.kings

    # Bulk bit-unpacking (8x8 planes)
    # unpackbits returns uint8 [0, 1]
    bits = np.unpackbits(bbs.view(np.uint8), bitorder='little').reshape(n, 12, 8, 8)
    # bits is (n, 12, 8, 8). We need to flip the rank dimension to match FEN order.
    buf_np[:n, 0:12] = bits[:, :, ::-1, :]
    
    # 2. Vectorized Castling, Turn, EP (Minor loop remaining, but only 6 planes)
    buf_np[:n, 12:18] = 0.0
    for i, b in enumerate(boards):
        cr = b.castling_rights
        if cr & chess.BB_H1: buf_np[i, 12] = 1.0
        if cr & chess.BB_A1: buf_np[i, 13] = 1.0
        if cr & chess.BB_H8: buf_np[i, 14] = 1.0
        if cr & chess.BB_A8: buf_np[i, 15] = 1.0
        if b.turn == chess.WHITE: buf_np[i, 16] = 1.0
        if b.ep_square is not None:
            buf_np[i, 17, 7 - (b.ep_square // 8), b.ep_square % 8] = 1.0

# =============================================================================
# SURGICAL OPTIMIZATION: Faster MCTSEngine Wrapper
# =============================================================================

class FastMCTSEngine(MCTSEngine):
    """
    Subclass of MCTSEngine that replaces the slow board encoding 
    and uses bfloat16 for H100.
    """
    def __init__(self, model_path, device, eval_batch_size):
        super().__init__(model_path, device, eval_batch_size)
        
        # H100 Optimization: Move model to bfloat16
        if self.device.type == "cuda":
            self.model = self.model.to(dtype=torch.bfloat16)
            # Re-compile if possible for better fusion
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                except:
                    pass

    def _forward_batch(
        self,
        boards: List[chess.Board],
        need_value: bool,
        need_policy: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Override to use our fast encoder and bfloat16."""
        if not boards:
            return None, None
            
        n = len(boards)
        # Use our fast vectorized encoder
        fast_boards_to_tensors(boards, self.pinned_tensor_buffer[:n])
        
        # Transfer to device as bfloat16 (H100 native)
        input_tensor = self.pinned_tensor_buffer[:n].to(self.device, dtype=torch.bfloat16, non_blocking=True)
        
        with torch.no_grad():
            values, policies = self.model(input_tensor)
            
        # Convert back to float32 for CPU processing
        v_res = values.float().cpu().numpy() if need_value else None
        p_res = policies.float().cpu().numpy() if need_policy else None
        
        return v_res, p_res

# =============================================================================
# Original Logic (Preserved from selfplay_generator.py)
# =============================================================================

def sample_move_with_temperature(
    moves: List[chess.Move],
    visit_counts: List[int],
    temperature: float = 1.0,
) -> chess.Move:
    if temperature == 0 or len(moves) == 1:
        max_visits = max(visit_counts)
        best_moves = [moves[i] for i, v in enumerate(visit_counts) if v == max_visits]
        return random.choice(best_moves)
    visits_array = np.array(visit_counts, dtype=np.float64)
    if temperature != 1.0:
        visits_array = visits_array ** (1.0 / temperature)
    total = visits_array.sum()
    if total == 0:
        probs = np.ones(len(moves)) / len(moves)
    else:
        probs = visits_array / total
    return np.random.choice(moves, p=probs)


def play_selfplay_game(
    neural: MCTSEngine,
    n_simulations: int,
    mcts_batch_size: int,
    cpuct: float,
    fpu_reduction: float,
    reuse_tree: bool,
    max_plies: int,
    temperature_moves: int = 30,
    verbose: bool = False,
    early_exit_min_sims: int = 0,
) -> Tuple[chess.pgn.Game, List[Dict]]:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play Training"
    game.headers["White"] = "NeuralEngine"
    game.headers["Black"] = "NeuralEngine"

    mcts_root: Optional[MCTSNode] = neural.make_root(board) if reuse_tree else None
    position_history: List[str] = [_position_key(board)]
    training_examples = []
    pgn_node = game
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if not reuse_tree or mcts_root is None:
            mcts_root = neural.make_root(board)

        _, search_stats = neural.choose_move(
            mcts_root,
            n_sims=n_simulations,
            batch_size=mcts_batch_size,
            cpuct=cpuct,
            fpu_reduction=fpu_reduction,
            game_history=position_history,
            early_exit_min_sims=early_exit_min_sims,
        )

        top_moves = search_stats['top_moves']
        move_visits_dict = {mv: visits for mv, visits, _, _ in top_moves}
        legal_moves = list(board.legal_moves)
        move_uci_list = [mv.uci() for mv in legal_moves]
        visit_counts = [move_visits_dict.get(mv, 0) for mv in legal_moves]

        training_examples.append({
            'fen': board.fen(),
            'move_uci_list': move_uci_list,
            'visit_counts': visit_counts,
            'turn': board.turn,
        })

        temperature = 1.0 if ply < temperature_moves else 0.0
        move = sample_move_with_temperature(legal_moves, visit_counts, temperature)

        board.push(move)
        position_history.append(_position_key(board))
        if reuse_tree and mcts_root is not None:
            mcts_root = neural.advance_root(mcts_root, move, position_history)
        pgn_node = pgn_node.add_variation(move)
        ply += 1

    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    return game, training_examples


def assign_game_outcome_to_examples(training_examples: List[Dict], game_result: str) -> List[Dict]:
    # White-perspective value targets (matches M0's supervised training and MCTS in play_games_mcts.py).
    if game_result == "1-0": white_outcome = 1.0
    elif game_result == "0-1": white_outcome = -1.0
    else: white_outcome = 0.0
    for example in training_examples:
        example['value_target'] = white_outcome
    return training_examples


def save_training_examples_to_csv(training_examples: List[Dict], csv_path: str, game_id: str):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['fen', 'policy_moves', 'policy_visits', 'value_target', 'game_id'])
        for example in training_examples:
            policy_moves = ','.join(example['move_uci_list'])
            policy_visits = ','.join(str(v) for v in example['visit_counts'])
            writer.writerow([example['fen'], policy_moves, policy_visits, example['value_target'], game_id])


_WORKER_NEURAL: Optional[MCTSEngine] = None
_WORKER_PLAY_PARAMS: Optional[Dict] = None


def _format_moves_san(game: chess.pgn.Game) -> str:
    moves_san = []
    node = game
    while node.variations:
        next_node = node.variation(0)
        moves_san.append(node.board().san(next_node.move))
        node = next_node
    parts = []
    for i in range(0, len(moves_san), 2):
        move_num = (i // 2) + 1
        if i + 1 < len(moves_san): parts.append(f"{move_num}. {moves_san[i]} {moves_san[i+1]}")
        else: parts.append(f"{move_num}. {moves_san[i]}")
    return " ".join(parts)


def _init_selfplay_worker(model_path, device, eval_batch_size, play_params, base_seed):
    global _WORKER_NEURAL, _WORKER_PLAY_PARAMS
    torch.set_num_threads(1)
    seed = (base_seed * 2654435761 + os.getpid()) & 0xFFFFFFFF
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # Use our FastMCTSEngine subclass
    _WORKER_NEURAL = FastMCTSEngine(model_path, device=device, eval_batch_size=eval_batch_size)
    _WORKER_PLAY_PARAMS = play_params


def _run_one_selfplay_game_in_worker(game_idx: int):
    global _WORKER_NEURAL, _WORKER_PLAY_PARAMS
    start = time.time()
    game, training_examples = play_selfplay_game(neural=_WORKER_NEURAL, **_WORKER_PLAY_PARAMS)
    elapsed = time.time() - start
    result = game.headers["Result"]
    training_examples = assign_game_outcome_to_examples(training_examples, result)
    return game_idx, result, training_examples, elapsed, _format_moves_san(game)


def generate_selfplay_games(
    model_path: str, num_games: int, output_csv: str, n_simulations: int = 800,
    mcts_batch_size: int = 64, cpuct: float = 2.0, fpu_reduction: float = 0.0,
    reuse_tree: bool = True, max_plies: int = 300, temperature_moves: int = 30,
    device: str = "cpu", eval_batch_size: int = 256, verbose: bool = False,
    early_exit_min_sims: int = 0, parallel_games: int = 1,
):
    n_workers = max(1, min(parallel_games, num_games))
    print(f"Generating {num_games} self-play games (FAST MODE)...")
    print(f"H100 Optimized: bfloat16, Vectorized bit-unpacking")
    print()

    total_positions = 0
    start_time = time.time()
    last_log_time = time.time()

    play_params = dict(
        n_simulations=n_simulations, mcts_batch_size=mcts_batch_size,
        cpuct=cpuct, fpu_reduction=fpu_reduction, reuse_tree=reuse_tree,
        max_plies=max_plies, temperature_moves=temperature_moves,
        early_exit_min_sims=early_exit_min_sims,
    )

    if n_workers == 1:
        neural = FastMCTSEngine(model_path, device=device, eval_batch_size=eval_batch_size)
        for game_idx in range(num_games):
            game_start = time.time()
            game, training_examples = play_selfplay_game(neural=neural, verbose=verbose, **play_params)
            result = game.headers["Result"]
            assign_game_outcome_to_examples(training_examples, result)
            save_training_examples_to_csv(training_examples, output_csv, f"game_{game_idx:05d}")
            total_positions += len(training_examples)
            game_elapsed = time.time() - game_start
            print(f"Game {game_idx + 1}: {result}, {len(training_examples)} positions, {game_elapsed:.1f}s")
    else:
        ctx = mp.get_context("spawn")
        base_seed = int(time.time_ns() & 0xFFFFFFFF)
        completed = 0
        with ctx.Pool(processes=n_workers, initializer=_init_selfplay_worker,
                     initargs=(model_path, device, eval_batch_size, play_params, base_seed)) as pool:
            for game_idx, result, training_examples, game_elapsed, moves_str in \
                    pool.imap_unordered(_run_one_selfplay_game_in_worker, range(num_games)):
                save_training_examples_to_csv(training_examples, output_csv, f"game_{game_idx:05d}")
                completed += 1
                total_positions += len(training_examples)
                
                # Periodic cumulative logging
                now = time.time()
                if now - last_log_time >= 60:
                    wall = now - start_time
                    mps = total_positions / wall if wall > 0 else 0
                    print(f"\n[CUMULATIVE STATS] Elapsed: {wall/60:.1f} min, Moves: {total_positions:,}, Throughput: {mps:.1f} moves/s")
                    last_log_time = now

                print(f"[{completed}/{num_games}] Game {game_idx + 1}: {result}, {len(training_examples)} positions, {game_elapsed:.1f}s, {len(training_examples)/game_elapsed:.1f} m/s")

    total_elapsed = time.time() - start_time
    print(f"\nCompleted {num_games} games in {total_elapsed:.1f}s")
    print(f"Overall throughput: {total_positions/total_elapsed:.1f} moves/s")


def main():
    parser = argparse.ArgumentParser(description="Generate self-play games for training")
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--output", default="selfplay_data.csv")
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--parallel-games", type=int, default=24)
    parser.add_argument("--cpuct", type=float, default=2.0)
    parser.add_argument("--temperature-moves", type=int, default=30)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--early-exit-min-sims", type=int, default=200)
    args = parser.parse_args()

    generate_selfplay_games(
        model_path=args.model, num_games=args.games, output_csv=args.output,
        n_simulations=args.simulations, mcts_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, cpuct=args.cpuct,
        temperature_moves=args.temperature_moves, device=args.device,
        verbose=args.verbose, early_exit_min_sims=args.early_exit_min_sims,
        parallel_games=args.parallel_games,
    )

if __name__ == "__main__":
    main()
