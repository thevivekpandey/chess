#!/usr/bin/env python3
"""
Play games between Stockfish and the trained neural chess engine, using
AlphaZero-style MCTS (PUCT) with leaf-parallel batched NN evaluation.
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import random
import threading
import time
from collections import Counter, OrderedDict
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import cython_chess
import numpy as np
import torch
from numba import njit

from chess_engine import ChessNet, board_to_tensor, boards_to_tensors, MOVE_FLAT_INDEX_TABLE

# Flat 1D view of MOVE_FLAT_INDEX_TABLE so the legal-priors hot path can index
# with a single linear int instead of a 3-axis numpy lookup per move.
# Shape (64, 64, 7) → strides 64*7=448, 7, 1.
_MOVE_FLAT_INDEX_TABLE_FLAT = MOVE_FLAT_INDEX_TABLE.reshape(-1)


@njit(cache=True, fastmath=True)
def _puct_argmax_core(s, fpu_baseline, cpuct_sqrtN,
                      child_N, child_W, child_vlosses, child_priors):
    # Single-pass PUCT argmax with no array allocations. All inputs are
    # scalars or 1-D numpy arrays (child_N/vlosses int32, child_W/priors
    # float32). Returns the argmax index.
    n = child_N.shape[0]
    best_idx = 0
    best_score = -1e30
    for i in range(n):
        eff = child_N[i] + child_vlosses[i]
        if eff > 0:
            q_mover = (s * child_W[i] - child_vlosses[i]) / eff
        else:
            q_mover = fpu_baseline
        score = q_mover + cpuct_sqrtN * child_priors[i] / (1 + eff)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# Terminal values, in WHITE's perspective, bounded to match NN value head's [-1, 1].
TERMINAL_WIN_WHITE = 1.0
TERMINAL_LOSS_WHITE = -1.0
TERMINAL_DRAW = 0.0


def find_latest_model() -> str:
    candidates = glob.glob("attempt_*_epoch*.pt")
    if candidates:
        return max(candidates, key=os.path.getmtime)
    if os.path.exists("attempt_13.pt"):
        return "attempt_13.pt"
    raise FileNotFoundError(
        "No model checkpoint found. Pass --model, or put attempt_*_epoch*.pt here."
    )


def pick_engine_color(mode: str, game_idx: int) -> chess.Color:
    if mode == "white":
        return chess.WHITE
    if mode == "black":
        return chess.BLACK
    if mode == "both":
        return chess.WHITE if game_idx % 2 == 0 else chess.BLACK
    return random.choice([chess.WHITE, chess.BLACK])


def _terminal_value_given(
    board: chess.Board,
    legal_moves: List[chess.Move],
    # Remove in_check from signature, compute lazily
) -> Optional[float]:
    if not legal_moves:
        # Only call is_check() if it's actually needed for mate detection
        if board.is_check():
            return TERMINAL_LOSS_WHITE if board.turn == chess.WHITE else TERMINAL_WIN_WHITE
        return TERMINAL_DRAW  # stalemate
    
    # halfmove_clock is a simple integer access
    if board.halfmove_clock >= 100:
        return TERMINAL_DRAW
        
    # is_insufficient_material is expensive. Fast-path: if there are pawns, 
    # it's not insufficient material.
    if board.pawns or board.rooks or board.queens:
        return None
        
    if board.is_insufficient_material():
        return TERMINAL_DRAW
    return None


def terminal_score_white(board: chess.Board) -> Optional[float]:
    legal = list(cython_chess.generate_legal_moves(board, chess.BB_ALL, chess.BB_ALL))
    return _terminal_value_given(board, legal)


def _position_key(board: chess.Board):
    return board._transposition_key()


def _check_repetition_terminal(
    node: "MCTSNode",
    game_history_counts: Counter,
    path_counts: Dict[Tuple, int],
) -> None:
    key = _position_key(node.board)
    # Check if this position has appeared 3 times (including this one)
    # count in game history + count in current simulation path
    count = game_history_counts.get(key, 0) + path_counts.get(key, 0)
    if count + 1 >= 3:
        node.is_terminal = True
        node.terminal_value = TERMINAL_DRAW


# =============================================================================
# MCTS tree
# =============================================================================


class MCTSNode:
    __slots__ = (
        "board", "parent", "parent_move", "prior",
        "children", "N", "W",
        "is_terminal", "terminal_value", "expanded",
        "turn", "virtual_loss",
        # Vectorized selection support
        "child_N", "child_W", "child_priors", "child_vlosses",
        "child_nodes", "child_moves", "index_in_parent",
        "total_N_eff",  # incrementally maintained sum(child_N + child_vlosses)
        "_cached_legal_moves",
    )

    def __init__(
        self,
        board: Optional[chess.Board] = None,
        parent: Optional["MCTSNode"] = None,
        parent_move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ):
        if board is None and parent is None:
            raise ValueError("MCTSNode needs either an explicit board or a parent")
        self.board = board
        self.parent = parent
        self.parent_move = parent_move
        self.prior = prior
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.N = 0
        self.W = 0.0
        self.is_terminal = False
        self.terminal_value: Optional[float] = None
        self.expanded = False
        self.turn = board.turn if board is not None else (not parent.turn)
        self.virtual_loss = 0
        self.index_in_parent = -1
        self.child_N = None
        self.child_W = None
        self.child_priors = None
        self.child_vlosses = None
        self.child_nodes = None
        self.child_moves = None
        self.total_N_eff = 0
        self._cached_legal_moves: Optional[List[chess.Move]] = None


def _materialize(
    node: MCTSNode,
    game_history_counts: Counter,
    path_counts: Dict[Tuple, int],
    engine: Optional["MCTSEngine"] = None,
) -> None:
    if node.board is not None:
        return
    parent = node.parent
    new_board = parent.board.copy(stack=False)
    new_board.push(node.parent_move)
    node.board = new_board

    key = _position_key(new_board)
    if engine is not None:
        cache = engine.legal_move_cache
        legal = cache.get(key)
        if legal is not None:
            cache.move_to_end(key)
        else:
            legal = list(cython_chess.generate_legal_moves(new_board, chess.BB_ALL, chess.BB_ALL))
            cache[key] = legal
            if len(cache) > engine._legal_move_cache_max:
                cache.popitem(last=False)
    else:
        legal = list(cython_chess.generate_legal_moves(new_board, chess.BB_ALL, chess.BB_ALL))

    term = _terminal_value_given(new_board, legal)
    if term is not None:
        node.is_terminal = True
        node.terminal_value = term
        return
    node._cached_legal_moves = legal

    _check_repetition_terminal(node, game_history_counts, path_counts)


# =============================================================================
# MCTS engine (stateless w.r.t. tree; caller owns the root)
# =============================================================================


class MCTSEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        eval_batch_size: int = 4096,
        verbose: bool = True,
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.eval_batch_size = eval_batch_size
        self.channels_last = self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            
        self.model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=16)

        if verbose:
            print(f"Loading neural model from {model_path} on {self.device}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.model.eval()
        
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
            except Exception:
                pass

        # Per-engine LRU cache of legal-move lists keyed by transposition key.
        # The previous "flush at 50K" policy threw away the entire cache on
        # overflow; LRU eviction keeps the recently-used positions, which is
        # what MCTS actually needs (subtrees revisit the same positions).
        self.legal_move_cache: "OrderedDict[Any, List[chess.Move]]" = OrderedDict()
        self._legal_move_cache_max = 100000
        
        self.pinned_tensor_buffer = None
        if self.device.type == "cuda":
            self.pinned_tensor_buffer = torch.empty(
                (eval_batch_size, 18, 8, 8),
                dtype=torch.float32,
                pin_memory=True
            )
            if self.channels_last:
                self.pinned_tensor_buffer = self.pinned_tensor_buffer.to(
                    memory_format=torch.channels_last
                )

        self._reset_search_stats()

    def _reset_search_stats(self):
        self.batched_eval_calls = 0
        self.evaluated_positions = 0

    def _forward_batch(
        self,
        boards: List[chess.Board],
        need_value: bool,
        need_policy: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not boards:
            return None, None
            
        actual_size = len(boards)
        # Bucketing for torch.compile stability
        if actual_size == 1:
            pad_to = 1
        elif actual_size <= 8:
            pad_to = 8
        elif actual_size <= 16:
            pad_to = 16
        elif actual_size <= 32:
            pad_to = 32
        elif actual_size <= 64:
            pad_to = 64
        elif actual_size <= 128:
            pad_to = 128
        elif actual_size <= 256:
            pad_to = 256
        else:
            pad_to = self.eval_batch_size

        # Use bfloat16 for H100
        amp_dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=amp_dtype)
            if self.device.type in ['cuda', 'cpu'] else nullcontext()
        )
        
        val_chunks: Optional[List[np.ndarray]] = [] if need_value else None
        pol_chunks: Optional[List[np.ndarray]] = [] if need_policy else None
        
        with torch.inference_mode(), amp_ctx:
            for start in range(0, actual_size, self.eval_batch_size):
                self.batched_eval_calls += 1
                end = min(start + self.eval_batch_size, actual_size)
                chunk = boards[start:end]
                curr_actual = len(chunk)
                curr_pad_to = pad_to if actual_size <= self.eval_batch_size else curr_actual
                
                if self.pinned_tensor_buffer is not None:
                    # Optimized batch encoding directly into pinned buffer
                    boards_to_tensors(chunk, self.pinned_tensor_buffer[:curr_actual])
                    
                    if curr_actual < curr_pad_to:
                        # Pad with copies of the last board to maintain stable shape
                        last_tensor = self.pinned_tensor_buffer[curr_actual-1:curr_actual]
                        padding = last_tensor.expand(curr_pad_to - curr_actual, -1, -1, -1)
                        self.pinned_tensor_buffer[curr_actual:curr_pad_to].copy_(padding)
                        
                    tensors = self.pinned_tensor_buffer[:curr_pad_to].to(self.device, non_blocking=True)
                else:
                    # Fallback
                    tensors = torch.stack([board_to_tensor(b) for b in chunk])
                    if curr_actual < curr_pad_to:
                        padding = tensors[-1:].expand(curr_pad_to - curr_actual, -1, -1, -1)
                        tensors = torch.cat([tensors, padding], dim=0)
                    tensors = tensors.to(self.device, non_blocking=True)

                if self.device.type == 'cuda':
                    tensors = tensors.to(dtype=amp_dtype)

                values, policies = self.model(tensors)
                if need_value:
                    val_chunks.append(values[:curr_actual].float().cpu().numpy().reshape(-1))
                if need_policy:
                    pol_chunks.append(policies[:curr_actual].float().cpu().numpy())
                    
        return (
            np.concatenate(val_chunks) if need_value else None,
            np.concatenate(pol_chunks, axis=0) if need_policy else None,
        )

    @staticmethod
    def _move_flat_index(move: chess.Move) -> Optional[int]:
        flat = int(MOVE_FLAT_INDEX_TABLE[
            move.from_square, move.to_square, move.promotion or 0
        ])
        return flat if flat >= 0 else None

    def _legal_priors_from_logits(
        self,
        board: chess.Board,
        policy_logits_np: np.ndarray,
        legal_moves: Optional[List[chess.Move]] = None,
    ) -> List[Tuple[chess.Move, float]]:
        if legal_moves is None:
            legal_moves = list(cython_chess.generate_legal_moves(board, chess.BB_ALL, chess.BB_ALL))
        n = len(legal_moves)
        if n == 0:
            return []

        # Vectorized: extract move fields into numpy arrays, do one bulk
        # table lookup, then mask out unindexed moves.
        from_sq = np.empty(n, dtype=np.int64)
        to_sq = np.empty(n, dtype=np.int64)
        promo = np.empty(n, dtype=np.int64)
        for i, m in enumerate(legal_moves):
            from_sq[i] = m.from_square
            to_sq[i] = m.to_square
            p = m.promotion
            promo[i] = 0 if p is None else p

        flat_keys = from_sq * 448 + to_sq * 7 + promo
        flat_lookups = _MOVE_FLAT_INDEX_TABLE_FLAT[flat_keys]
        indexed_mask = flat_lookups >= 0

        if not indexed_mask.any():
            return [(m, 1.0 / n) for m in legal_moves]

        flat_indices = flat_lookups[indexed_mask]
        flat_logits = policy_logits_np.reshape(-1)
        legal_logits = flat_logits[flat_indices].astype(np.float64, copy=False)
        legal_logits -= legal_logits.max()
        np.exp(legal_logits, out=legal_logits)
        legal_logits /= legal_logits.sum()

        # Preserve the original output order: indexed moves first (in their
        # encounter order), then unindexed moves with 0.0 prior. Order affects
        # argmax tie-breaking in _select_child_idx for 0-prior children.
        priors = legal_logits.tolist()
        indexed_pairs = []
        unindexed_pairs = []
        prior_iter = iter(priors)
        for m, ok in zip(legal_moves, indexed_mask.tolist()):
            if ok:
                indexed_pairs.append((m, next(prior_iter)))
            else:
                unindexed_pairs.append((m, 0.0))
        return indexed_pairs + unindexed_pairs

    def make_root(self, board: chess.Board) -> MCTSNode:
        node = MCTSNode(board=board.copy(stack=False))
        # Set terminal status up front so callers can rely on `node.is_terminal`
        # without first running a search. advance_root already does this for
        # subsequent roots; doing it here too lets the self-play loop use a
        # single consistent terminal check.
        intrinsic = terminal_score_white(node.board)
        if intrinsic is not None:
            node.is_terminal = True
            node.terminal_value = intrinsic
        return node

    def advance_root(self, root: MCTSNode, move: chess.Move, game_history: List[str]) -> MCTSNode:
        history_excl_self = game_history[:-1]
        history_counts = Counter(history_excl_self)

        if move in root.children:
            new_root = root.children[move]
            if new_root is None:
                idx = root.child_moves.index(move)
                new_root = MCTSNode(
                    board=None, parent=root, parent_move=move,
                    prior=root.child_priors[idx]
                )
                new_root.index_in_parent = idx
                root.child_nodes[idx] = new_root
                root.children[move] = new_root

            if new_root.board is None:
                _materialize(new_root, history_counts, {}, engine=self)
            else:
                intrinsic = terminal_score_white(new_root.board)
                if intrinsic is not None:
                    new_root.is_terminal = True
                    new_root.terminal_value = intrinsic
                else:
                    new_root.is_terminal = False
                    new_root.terminal_value = None
                    _check_repetition_terminal(new_root, history_counts, {})
        else:
            new_board = root.board.copy(stack=False)
            new_board.push(move)
            new_root = MCTSNode(board=new_board)
            intrinsic = terminal_score_white(new_board)
            if intrinsic is not None:
                new_root.is_terminal = True
                new_root.terminal_value = intrinsic
            else:
                _check_repetition_terminal(new_root, history_counts, {})
        new_root.parent = None
        new_root.parent_move = None
        return new_root

    def _expand_node(self, node: MCTSNode, policy_logits_np: np.ndarray):
        ranked = self._legal_priors_from_logits(
            node.board, policy_logits_np, node._cached_legal_moves
        )
        node._cached_legal_moves = None

        n_children = len(ranked)
        node.child_moves = [r[0] for r in ranked]
        node.child_priors = np.array([r[1] for r in ranked], dtype=np.float32)
        node.child_N = np.zeros(n_children, dtype=np.int32)
        node.child_W = np.zeros(n_children, dtype=np.float32)
        node.child_vlosses = np.zeros(n_children, dtype=np.int32)
        node.child_nodes = [None] * n_children
        node.children = {m: None for m in node.child_moves}
        node.expanded = True

    def _select_child_idx(
        self,
        node: MCTSNode,
        cpuct: float,
        fpu_reduction: float,
    ) -> int:
        # Mover sign: +1 for white-to-move, -1 for black. Since s ∈ {-1, +1},
        # s*s = 1, so q_mover_visited = s*(W - s*vl)/eff_N simplifies to
        # (s*W - vl)/eff_N — one fewer multiply and avoids per-color branching.
        s = 1.0 if node.turn == chess.WHITE else -1.0

        parent_q_white = node.W / node.N if node.N > 0 else 0.0
        fpu_baseline = s * parent_q_white - fpu_reduction

        total_N_eff = node.total_N_eff
        sqrt_total = math.sqrt(total_N_eff) if total_N_eff > 0 else 1.0
        cpuct_sqrtN = cpuct * sqrt_total

        return _puct_argmax_core(
            s, fpu_baseline, cpuct_sqrtN,
            node.child_N, node.child_W, node.child_vlosses, node.child_priors,
        )

    def _descend(
        self,
        root: MCTSNode,
        cpuct: float,
        fpu_reduction: float,
        game_history_counts: Counter,
        path_counts: Dict[Tuple, int],
        board: chess.Board,
    ) -> Tuple[MCTSNode, List[MCTSNode], List[chess.Move]]:
        node = root
        path = [node]
        moves_made = []
        while node.expanded and not node.is_terminal:
            idx = self._select_child_idx(node, cpuct, fpu_reduction)
            child = node.child_nodes[idx]
            if child is None:
                move = node.child_moves[idx]
                prior = node.child_priors[idx]
                child = MCTSNode(board=None, parent=node, parent_move=move, prior=prior)
                child.index_in_parent = idx
                node.child_nodes[idx] = child
                node.children[move] = child

            move = child.parent_move
            board.push(move)
            moves_made.append(move)
            
            if child.board is None:
                # Lazy materialize: check terminal/repetition using scratch board
                key = _position_key(board)
                count = game_history_counts.get(key, 0) + path_counts.get(key, 0)
                if count + 1 >= 3:
                    child.is_terminal = True
                    child.terminal_value = TERMINAL_DRAW
                else:
                    cache = self.legal_move_cache
                    legal = cache.get(key)
                    if legal is not None:
                        cache.move_to_end(key)
                    else:
                        legal = list(cython_chess.generate_legal_moves(board, chess.BB_ALL, chess.BB_ALL))
                        cache[key] = legal
                        if len(cache) > self._legal_move_cache_max:
                            cache.popitem(last=False)
                    
                    term = _terminal_value_given(board, legal)
                    if term is not None:
                        child.is_terminal = True
                        child.terminal_value = term
                    else:
                        child._cached_legal_moves = legal
                
                # Leaf nodes (unexpanded) MUST have a board for NN evaluation
                if not child.expanded:
                    child.board = board.copy(stack=False)

            node = child
            path.append(node)
            
            if not node.is_terminal:
                key = _position_key(board)
                path_counts[key] = path_counts.get(key, 0) + 1
                
        return node, path, moves_made

    @staticmethod
    def _backprop(path: List[MCTSNode], v_white: float, undo_virtual_loss: bool):
        # total_N_eff = sum(child_N + child_vlosses). When undoing virtual loss,
        # child_N += 1 and child_vlosses -= 1 cancel — net 0. When no vloss was
        # applied (terminal leaf), only child_N += 1 — net +1.
        for n in path:
            if undo_virtual_loss:
                n.virtual_loss -= 1
            n.N += 1
            n.W += v_white
            p = n.parent
            if p:
                idx = n.index_in_parent
                p.child_N[idx] = n.N
                p.child_W[idx] = n.W
                if undo_virtual_loss:
                    p.child_vlosses[idx] = n.virtual_loss
                else:
                    p.total_N_eff += 1

    @staticmethod
    def _is_visit_dominated(
        root: MCTSNode, n_sims: int, n_done: int, min_sims: int
    ) -> bool:
        if min_sims <= 0 or n_done < min_sims or root.child_N is None:
            return False
        n_arr = root.child_N
        if len(n_arr) < 2:
            return False
        if len(n_arr) < 50:
            top_n = -1
            second_n = -1
            for val in n_arr:
                if val > top_n:
                    second_n = top_n
                    top_n = val
                elif val > second_n:
                    second_n = val
        else:
            top_two_idx = np.partition(n_arr, -2)[-2:]
            second_n = top_two_idx[0]
            top_n = top_two_idx[1]
        return top_n - second_n > (n_sims - n_done)

    def run_simulations(
        self,
        root: MCTSNode,
        n_sims: int,
        batch_size: int,
        cpuct: float,
        fpu_reduction: float,
        game_history: List[str],
        early_exit_min_sims: int = 0,
        dirichlet_alpha: float = 0.0,
        dirichlet_epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        t0 = time.time()
        self._reset_search_stats()
        history_counts = Counter(game_history)

        intrinsic = terminal_score_white(root.board)
        if intrinsic is not None:
            root.is_terminal = True
            root.terminal_value = intrinsic
            return {
                "elapsed": time.time() - t0,
                "simulations": 0,
                "evaluated_positions": 0,
                "batched_eval_calls": 0,
                "early_exit": False,
            }
        if root.is_terminal:
            root.is_terminal = False
            root.terminal_value = None

        if not root.expanded:
            _, p_arr = self._forward_batch(
                [root.board], need_value=False, need_policy=True
            )
            self._expand_node(root, p_arr[0])
            self.evaluated_positions += 1

        # Apply Dirichlet noise to root priors for exploration in self-play.
        # Even if already expanded (tree reuse), AlphaZero adds noise to the root of the current search.
        if dirichlet_epsilon > 0 and dirichlet_alpha > 0:
            n_moves = len(root.child_moves)
            if n_moves > 0:
                noise = np.random.dirichlet([dirichlet_alpha] * n_moves)
                root.child_priors = (1 - dirichlet_epsilon) * root.child_priors + \
                                    dirichlet_epsilon * noise.astype(np.float32)

        n_done = 0
        early_exit_triggered = False
        scratch_board = root.board.copy(stack=False)
        while n_done < n_sims:
            if self._is_visit_dominated(root, n_sims, n_done, early_exit_min_sims):
                early_exit_triggered = True
                break

            target = min(batch_size, n_sims - n_done)
            batch_leaves = []
            batch_paths = []

            while len(batch_leaves) < target and (n_done + len(batch_leaves)) < n_sims:
                path_counts = {} 
                leaf, path, moves_made = self._descend(root, cpuct, fpu_reduction, history_counts, path_counts, scratch_board)
                
                # Pop moves to return scratch_board to root state
                for _ in range(len(moves_made)):
                    scratch_board.pop()

                if leaf.is_terminal:
                    self._backprop(path, leaf.terminal_value, undo_virtual_loss=False)
                    n_done += 1
                    if self._is_visit_dominated(root, n_sims, n_done, early_exit_min_sims):
                        early_exit_triggered = True
                        break
                    continue
                    
                for n in path:
                    n.virtual_loss += 1
                    if n.parent:
                        n.parent.child_vlosses[n.index_in_parent] = n.virtual_loss
                        n.parent.total_N_eff += 1
                batch_leaves.append(leaf)
                batch_paths.append(path)

            if early_exit_triggered:
                break
            if not batch_leaves:
                break

            unique_leaves = []
            leaf_to_idx = {}
            for leaf in batch_leaves:
                if id(leaf) not in leaf_to_idx:
                    leaf_to_idx[id(leaf)] = len(unique_leaves)
                    unique_leaves.append(leaf)

            v_arr, p_arr = self._forward_batch(
                [leaf.board for leaf in unique_leaves], need_value=True, need_policy=True
            )
            self.evaluated_positions += len(unique_leaves)

            for i, leaf in enumerate(unique_leaves):
                if not leaf.expanded:
                    self._expand_node(leaf, p_arr[i])

            for leaf, path in zip(batch_leaves, batch_paths):
                v_white = float(v_arr[leaf_to_idx[id(leaf)]])
                self._backprop(path, v_white, undo_virtual_loss=True)
                n_done += 1

        return {
            "elapsed": time.time() - t0,
            "simulations": n_done,
            "evaluated_positions": self.evaluated_positions,
            "batched_eval_calls": self.batched_eval_calls,
            "early_exit": early_exit_triggered,
        }

    def choose_move(
        self,
        root: MCTSNode,
        n_sims: int,
        batch_size: int,
        cpuct: float,
        fpu_reduction: float,
        game_history: List[str],
        early_exit_min_sims: int = 0,
        dirichlet_alpha: float = 0.0,
        dirichlet_epsilon: float = 0.0,
    ) -> Tuple[chess.Move, Dict[str, Any]]:
        sim_stats = self.run_simulations(
            root, n_sims, batch_size, cpuct, fpu_reduction, game_history,
            early_exit_min_sims=early_exit_min_sims,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

        if root.child_N is None or len(root.child_moves) == 0:
            raise ValueError("Root has no children; no legal move available.")

        is_white_to_move = (root.turn == chess.WHITE)
        n_arr = root.child_N
        w_arr = root.child_W
        q_white_arr = np.where(n_arr > 0, w_arr / np.maximum(n_arr, 1), 0.0)
        mover_q = q_white_arr if is_white_to_move else -q_white_arr
        best_idx = np.lexsort((mover_q, n_arr))[-1]
        best_move = root.child_moves[best_idx]

        sort_indices = np.argsort(-(n_arr.astype(np.float64) * 1e6 + mover_q))
        top_moves = []
        for i in sort_indices[:10]:
            top_moves.append((
                root.child_moves[i],
                int(root.child_N[i]),
                float(root.child_priors[i]),
                float(q_white_arr[i])
            ))

        sim_stats.update({
            "q_for_mover": float(mover_q[best_idx]),
            "selected_prior": float(root.child_priors[best_idx]),
            "top_moves": top_moves,
            "root_visits": int(root.N),
        })
        return best_move, sim_stats


def format_top_moves(
    board: chess.Board,
    top_moves: List[Tuple[chess.Move, int, float, float]],
    k: int,
) -> str:
    parts = []
    for move, visits, prior, q_white in top_moves[:k]:
        parts.append(f"{board.san(move)} N={visits} P={prior:.1%} Qw={q_white:+.2f}")
    return ", ".join(parts)


def configure_stockfish(engine: chess.engine.SimpleEngine, skill_level: int):
    skill_level = max(0, min(20, skill_level))
    try:
        engine.configure({"Skill Level": skill_level})
    except chess.engine.EngineError as exc:
        print(f"Warning: could not configure Stockfish skill level: {exc}")


def stockfish_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    movetime: Optional[float],
    depth: Optional[int],
) -> chess.Move:
    limit = chess.engine.Limit(depth=depth) if depth is not None else chess.engine.Limit(time=movetime)
    return engine.play(board, limit).move


def play_one_game(
    game_idx: int,
    neural: MCTSEngine,
    stockfish: chess.engine.SimpleEngine,
    neural_color: chess.Color,
    n_simulations: int,
    mcts_batch_size: int,
    cpuct: float,
    fpu_reduction: float,
    reuse_tree: bool,
    stockfish_time: float,
    stockfish_depth: Optional[int],
    stockfish_level: int,
    max_plies: int,
    verbose: bool,
    progress_callback: Optional[Callable[[int, str, bool], None]] = None,
    early_exit_min_sims: int = 0,
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
) -> Tuple[chess.pgn.Game, float, float]:
    board = chess.Board()
    game = chess.pgn.Game()
    sf_name = f"Stockfish-L{stockfish_level}"
    game.headers["Event"] = f"NeuralEngine(MCTS) vs {sf_name}"
    game.headers["Date"] = datetime.utcnow().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_idx + 1)
    game.headers["White"] = "NeuralEngine" if neural_color == chess.WHITE else sf_name
    game.headers["Black"] = sf_name if neural_color == chess.WHITE else "NeuralEngine"

    mcts_root: Optional[MCTSNode] = neural.make_root(board) if reuse_tree else None
    position_history: List[str] = [_position_key(board)]

    pgn_node = game
    ply = 0
    nn_time_total = 0.0
    sf_time_total = 0.0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if board.turn == neural_color:
            if not reuse_tree or mcts_root is None:
                mcts_root = neural.make_root(board)
            move, search_stats = neural.choose_move(
                mcts_root,
                n_sims=n_simulations,
                batch_size=mcts_batch_size,
                cpuct=cpuct,
                fpu_reduction=fpu_reduction,
                game_history=position_history,
                early_exit_min_sims=early_exit_min_sims,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
            )
            nn_time_total += search_stats["elapsed"]
            ee_tag = "*" if search_stats.get("early_exit") else ""
            comment = f"Q {search_stats['q_for_mover']:+.2f} P {search_stats['selected_prior']:.3f} N {search_stats['root_visits']}{ee_tag}"
            if verbose:
                print(f"  Top: {format_top_moves(board, search_stats['top_moves'], 5)}")
        else:
            sf_t0 = time.time()
            move = stockfish_move(stockfish, board, stockfish_time, stockfish_depth)
            sf_time_total += time.time() - sf_t0
            comment = ""

        san = board.san(move)
        board.push(move)
        position_history.append(_position_key(board))

        if reuse_tree and mcts_root is not None:
            mcts_root = neural.advance_root(mcts_root, move, position_history)
        pgn_node = pgn_node.add_variation(move)
        if comment:
            pgn_node.comment = comment
        ply += 1

        if progress_callback is not None:
            progress_callback(ply, san, board.turn == chess.BLACK)

    game.headers["Result"] = board.result(claim_draw=True)
    return game, nn_time_total, sf_time_total


_WORKER_NEURAL: Optional[MCTSEngine] = None
_WORKER_STOCKFISH: Optional[chess.engine.SimpleEngine] = None
_WORKER_LAST_LEVEL: Optional[int] = None
_WORKER_PROGRESS: Optional[Any] = None


def _init_worker(
    model_path: str,
    device: str,
    eval_batch_size: int,
    stockfish_path: str,
    progress_dict: Optional[Any],
):
    global _WORKER_NEURAL, _WORKER_STOCKFISH, _WORKER_LAST_LEVEL, _WORKER_PROGRESS
    _WORKER_NEURAL = MCTSEngine(
        model_path,
        device=device,
        eval_batch_size=eval_batch_size,
    )
    _WORKER_STOCKFISH = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        _WORKER_STOCKFISH.configure({"Threads": 1})
    except Exception:
        pass
    _WORKER_LAST_LEVEL = None
    _WORKER_PROGRESS = progress_dict


def _run_one_game_in_worker(task: Tuple[int, int, bool, Dict[str, Any]]):
    global _WORKER_NEURAL, _WORKER_STOCKFISH, _WORKER_LAST_LEVEL, _WORKER_PROGRESS
    level, game_idx, neural_color, play_params = task

    if level != _WORKER_LAST_LEVEL:
        configure_stockfish(_WORKER_STOCKFISH, level)
        _WORKER_LAST_LEVEL = level

    pid = os.getpid()
    progress_cb = None
    if _WORKER_PROGRESS is not None:
        _WORKER_PROGRESS[pid] = (level, game_idx, neural_color, 0, "", False, time.time())
        def progress_cb(ply, san, white_just_moved):
            _WORKER_PROGRESS[pid] = (level, game_idx, neural_color, ply, san, white_just_moved, time.time())

    start = time.time()
    try:
        game, nn_time, sf_time = play_one_game(
            game_idx=game_idx,
            neural=_WORKER_NEURAL,
            stockfish=_WORKER_STOCKFISH,
            neural_color=neural_color,
            stockfish_level=level,
            verbose=False,
            progress_callback=progress_cb,
            **play_params,
        )
    finally:
        if _WORKER_PROGRESS is not None:
            _WORKER_PROGRESS.pop(pid, None)
    elapsed = time.time() - start
    return (level, game_idx, neural_color, str(game), game.headers["Result"], elapsed, nn_time, sf_time)


def _format_progress_line(state: tuple) -> str:
    level, game_idx, neural_color, ply, san, white_moved, ts = state
    elapsed = time.time() - ts
    full_move = (ply + 1) // 2
    move_str = "(starting)" if ply == 0 else (f"{full_move}.{san}" if white_moved else f"{full_move}...{san}")
    side_to_move_now = chess.BLACK if white_moved and ply > 0 else chess.WHITE
    thinking = "neural" if (ply == 0 and neural_color == chess.WHITE) or (ply > 0 and side_to_move_now == neural_color) else "stockfish"
    n_color = "W" if neural_color == chess.WHITE else "B"
    return f"  L{level:>2} g{game_idx + 1} (n={n_color}) ply {ply} {move_str:<10} — {thinking} thinking {elapsed:.1f}s"


def _progress_printer_loop(progress_dict, stop_event, interval, total_tasks, games_done_ref, print_lock):
    while not stop_event.wait(interval):
        try:
            snapshot = list(progress_dict.items())
        except Exception:
            continue
        if not snapshot:
            continue
        snapshot.sort(key=lambda kv: (kv[1][0], kv[1][1]))
        with print_lock:
            done = games_done_ref[0]
            print(f"\n--- progress ({len(snapshot)} active, {done}/{total_tasks} done) ---")
            for _pid, state in snapshot:
                print(_format_progress_line(state))


def _classify_result(result: str, neural_color: chess.Color) -> str:
    if result == "1/2-1/2": return "draw"
    if (result == "1-0" and neural_color == chess.WHITE) or (result == "0-1" and neural_color == chess.BLACK): return "win"
    return "loss"


def main():
    parser = argparse.ArgumentParser(description="Play Stockfish vs the trained neural engine using MCTS.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--stockfish-level-min", type=int, default=5)
    parser.add_argument("--stockfish-level-max", type=int, default=None)
    parser.add_argument("--stockfish-time", type=float, default=0.1)
    parser.add_argument("--stockfish-depth", type=int, default=None)
    parser.add_argument("--mcts-simulations", type=int, default=800)
    parser.add_argument("--mcts-batch-size", type=int, default=16)
    parser.add_argument("--mcts-cpuct", type=float, default=2.0)
    parser.add_argument("--mcts-fpu", type=float, default=0.0)
    parser.add_argument("--mcts-no-reuse-tree", action="store_true")
    parser.add_argument("--mcts-early-exit-min-sims", type=int, default=200)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.0)
    parser.add_argument("--engine-color", choices=["white", "black", "both", "random"], default="both")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--pgn-out", default="neural_vs_stockfish_mcts.pgn")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--parallel-games", type=int, default=1)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    args = parser.parse_args()

    level_min = args.stockfish_level_min
    level_max = args.stockfish_level_max if args.stockfish_level_max is not None else level_min
    levels = list(range(level_min, level_max + 1))
    model_path = args.model or find_latest_model()
    mode = "a" if args.append else "w"

    play_params = {
        "n_simulations": args.mcts_simulations, "mcts_batch_size": args.mcts_batch_size,
        "cpuct": args.mcts_cpuct, "fpu_reduction": args.mcts_fpu,
        "reuse_tree": not args.mcts_no_reuse_tree, "stockfish_time": args.stockfish_time,
        "stockfish_depth": args.stockfish_depth, "max_plies": args.max_plies,
        "early_exit_min_sims": args.mcts_early_exit_min_sims,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_epsilon": args.dirichlet_epsilon,
    }

    if args.parallel_games == 1:
        _run_sequential(args, levels, model_path, mode, play_params)
    else:
        _run_parallel(args, levels, model_path, mode, play_params, args.parallel_games)


def _run_sequential(args, levels, model_path, mode, play_params):
    neural = MCTSEngine(model_path, device=args.device, eval_batch_size=args.eval_batch_size)
    stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    summary = []
    try:
        with open(args.pgn_out, mode, encoding="utf-8") as pgn_file:
            for level in levels:
                configure_stockfish(stockfish, level)
                wins = losses = draws = 0
                level_nn = level_sf = 0.0
                for game_idx in range(args.games):
                    neural_color = pick_engine_color(args.engine_color, game_idx)
                    game, nn_time, sf_time = play_one_game(game_idx=game_idx, neural=neural, stockfish=stockfish, neural_color=neural_color, stockfish_level=level, verbose=not args.quiet, **play_params)
                    level_nn += nn_time; level_sf += sf_time
                    print(game, file=pgn_file, end="\n\n"); pgn_file.flush()
                    outcome = _classify_result(game.headers["Result"], neural_color)
                    if outcome == "win": wins += 1
                    elif outcome == "loss": losses += 1
                    else: draws += 1
                summary.append((level, wins, losses, draws, level_nn, level_sf))
    finally:
        stockfish.quit()
    for s in summary: print(f"L{s[0]:>2}: +{s[1]} -{s[2]} ={s[3]} nn={s[4]:.1f}s sf={s[5]:.1f}s")


def _run_parallel(args, levels, model_path, mode, play_params, parallel):
    tasks = []
    for level in levels:
        for game_idx in range(args.games):
            tasks.append((level, game_idx, pick_engine_color(args.engine_color, game_idx), play_params))
    n_workers = min(parallel, len(tasks))
    ctx = mp.get_context("spawn")
    progress_dict = None; progress_manager = None; progress_thread = None
    progress_stop = threading.Event(); print_lock = threading.Lock(); games_done_ref = [0]
    if args.progress_interval > 0:
        progress_manager = ctx.Manager(); progress_dict = progress_manager.dict()
        progress_thread = threading.Thread(target=_progress_printer_loop, args=(progress_dict, progress_stop, args.progress_interval, len(tasks), games_done_ref, print_lock), daemon=True)
        progress_thread.start()
    summary_per_level = {level: [0, 0, 0, 0.0, 0.0] for level in levels}
    overall_start = time.time()
    try:
        with open(args.pgn_out, mode, encoding="utf-8") as pgn_file:
            with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(model_path, args.device, args.eval_batch_size, args.stockfish, progress_dict)) as pool:
                for level, game_idx, neural_color, pgn_str, result, game_secs, nn_time, sf_time in pool.imap_unordered(_run_one_game_in_worker, tasks):
                    pgn_file.write(pgn_str); pgn_file.write("\n\n"); pgn_file.flush()
                    outcome = _classify_result(result, neural_color)
                    if outcome == "win": summary_per_level[level][0] += 1
                    elif outcome == "loss": summary_per_level[level][1] += 1
                    else: summary_per_level[level][2] += 1
                    summary_per_level[level][3] += nn_time; summary_per_level[level][4] += sf_time
                    games_done_ref[0] += 1
                    with print_lock: print(f"[{games_done_ref[0]}/{len(tasks)}] L{level} game {game_idx+1}: {result} in {game_secs:.1f}s")
    finally:
        progress_stop.set()
    print(f"Wall time: {time.time()-overall_start:.1f}s")


if __name__ == "__main__":
    main()
