#!/usr/bin/env python3
"""
Play games between Stockfish and the trained neural chess engine, using
AlphaZero-style MCTS (PUCT) with leaf-parallel batched NN evaluation.

Mirrors play_games.py (Stockfish levels, parallel-games pool, progress
snapshots, NN/SF time accounting) but replaces the BFS-pruned minimax search
with a real MCTS:
  - Node values and Q stored from WHITE's perspective (matches the value head).
  - Selection uses PUCT: argmax_c [ Q_for_player(c) + cpuct * P(c) * sqrt(N) / (1 + N(c)) ].
  - Leaf parallelism via virtual loss: each batch collects up to `batch_size`
    leaves with virtual loss applied along their paths, runs ONE batched
    forward pass for value+policy, expands all leaves, then backprops and
    undoes virtual loss.
  - Subtree reuse across consecutive moves in a game (engine and opponent).

Usage example:
  ~/myvenv/bin/python play_games_mcts.py \
      --model attempt_14_epoch077.pt \
      --stockfish-level-min 0 --stockfish-level-max 8 \
      --games 5 --engine-color both \
      --mcts-simulations 800 --mcts-batch-size 16 --mcts-cpuct 2.0 \
      --stockfish-time 2 --parallel-games 16
"""

import argparse
import glob
import math
import multiprocessing as mp
import os
import random
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from chess_engine import ChessNet, fen_to_tensor, move_to_policy_index


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


def terminal_score_white(board: chess.Board) -> Optional[float]:
    """Terminal value in WHITE's perspective, or None if not terminal.

    Note: 3-fold repetition is NOT detected here, because the board passed in
    was built via copy(stack=False).push(move) — its move stack is empty, so
    python-chess can't see prior occurrences. Repetition is handled separately
    by `_check_repetition_terminal`, which uses the explicit game/path history.
    """
    if board.is_checkmate():
        # Side to move is checkmated -> they lose.
        return TERMINAL_LOSS_WHITE if board.turn == chess.WHITE else TERMINAL_WIN_WHITE
    if board.is_game_over(claim_draw=True):
        return TERMINAL_DRAW
    return None


def _position_key(board: chess.Board) -> str:
    """FIDE-correct repetition key: pieces + side-to-move + castling + en-passant.

    Drops the halfmove and fullmove counters from the FEN. Two positions are
    "the same" for repetition purposes iff their `_position_key` matches.
    Using `board_fen()` (piece placement only) is too loose: it would consider
    a post-castling position equal to the start position even though castling
    rights differ.
    """
    return board.fen().rsplit(" ", 2)[0]


def _check_repetition_terminal(
    node: "MCTSNode",
    game_history: List[str],
    path_below_root: List["MCTSNode"],
) -> None:
    """Set node.is_terminal/terminal_value=DRAW iff its position is the 3rd occurrence.

    Caller guarantees node.board is set, and that node's own position is NOT
    represented in either `game_history` or `path_below_root` (we add +1 for
    it here). Specifically:
      - `game_history` is the list of played-position keys; for descents below
        root, this includes root's key as the last entry, which is OK because
        root sits at path[0], not in path_below_root.
      - `path_below_root` is the MCTS path from root's first child down to
        (but not including) `node`. Root MUST be excluded — otherwise root
        would be double-counted (it's already the tail of `game_history`).
    """
    key = _position_key(node.board)
    game_count = sum(1 for k in game_history if k == key)
    path_count = sum(
        1 for n in path_below_root
        if n.board is not None and _position_key(n.board) == key
    )
    if game_count + path_count + 1 >= 3:
        node.is_terminal = True
        node.terminal_value = TERMINAL_DRAW


# =============================================================================
# MCTS tree
# =============================================================================


class MCTSNode:
    """Node in the MCTS tree.

    Value bookkeeping is in WHITE's perspective:
      W = sum of v_white over visits ; Q_white = W / N.
    At a white-to-move node we want max Q_white; at black-to-move, min Q_white.

    Virtual loss is an unsigned counter; when computing effective W we add a
    signed shift per virtual visit:
      vl_shift_white = -1 if parent.turn == WHITE else +1
    so OTHER concurrent descents see this child as "worse for the parent's
    player" and pick siblings.

    Lazy boards: child nodes are created with `board=None` (only prior + parent
    pointer + parent_move are set). The board is materialized lazily on the
    first descent through the child via `_materialize`. Most children at any
    expanded node receive 0 visits, so we save ~70% of board copies.
    """

    __slots__ = (
        "board", "parent", "parent_move", "prior",
        "children", "N", "W",
        "is_terminal", "terminal_value", "expanded",
        "turn", "virtual_loss",
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
        # Turn is known eagerly: if board is given, read from it; otherwise
        # flip parent's turn (parent's turn is always set, recursively).
        self.turn = board.turn if board is not None else (not parent.turn)
        self.virtual_loss = 0


def _materialize(
    node: MCTSNode,
    game_history: List[str],
    path_to_parent: List[MCTSNode],
) -> None:
    """Build node.board from its parent if it's a lazy stub, and detect terminal status.

    No-op if the node is already materialized. Caller must guarantee
    node.parent.board is set; this is true by construction whenever we are
    descending from root, since each step materializes the child before
    moving into it.

    Args:
        node: The node to materialize.
        game_history: Position keys from the actual played game. The last entry
            is expected to be root's key (the position before the upcoming move).
            Node's own key must NOT be present.
        path_to_parent: MCTS path from root through node's parent (i.e. everything
            traversed during this descent so far, excluding `node`). The
            repetition counter skips path_to_parent[0] (the root) because root
            is already represented in `game_history`'s tail.
    """
    if node.board is not None:
        return
    parent = node.parent
    new_board = parent.board.copy(stack=False)
    new_board.push(node.parent_move)
    node.board = new_board

    term = terminal_score_white(new_board)
    if term is not None:
        node.is_terminal = True
        node.terminal_value = term
        return

    # Repetition: skip path[0] (root) — already counted in game_history's tail.
    _check_repetition_terminal(node, game_history, path_to_parent[1:])


# =============================================================================
# MCTS engine (stateless w.r.t. tree; caller owns the root)
# =============================================================================


class MCTSEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        eval_batch_size: int = 4096,
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
        self.model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=16)

        print(f"Loading neural model from {model_path} on {self.device}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._reset_search_stats()

    def _reset_search_stats(self):
        self.batched_eval_calls = 0
        self.evaluated_positions = 0

    # --- forward pass and policy decoding ---

    def _forward_batch(
        self,
        fens: List[str],
        need_value: bool,
        need_policy: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not fens:
            return None, None
        val_chunks: Optional[List[np.ndarray]] = [] if need_value else None
        pol_chunks: Optional[List[np.ndarray]] = [] if need_policy else None
        with torch.no_grad():
            for start in range(0, len(fens), self.eval_batch_size):
                self.batched_eval_calls += 1
                chunk = fens[start:start + self.eval_batch_size]
                tensors = torch.stack([fen_to_tensor(f) for f in chunk]).to(self.device)
                values, policies = self.model(tensors)
                if need_value:
                    val_chunks.append(values.float().cpu().numpy().reshape(-1))
                if need_policy:
                    pol_chunks.append(policies.float().cpu().numpy())
        return (
            np.concatenate(val_chunks) if need_value else None,
            np.concatenate(pol_chunks, axis=0) if need_policy else None,
        )

    @staticmethod
    def _move_flat_index(move: chess.Move) -> Optional[int]:
        idx = move_to_policy_index(move.uci())
        if idx is None:
            return None
        src_row, src_col, plane = idx
        return plane * 64 + src_row * 8 + src_col

    def _legal_priors_from_logits(
        self,
        board: chess.Board,
        policy_logits_np: np.ndarray,
    ) -> List[Tuple[chess.Move, float]]:
        """Softmax over legal moves only, using the (73,8,8) policy logits."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []
        flat_indices: List[int] = []
        indexed_moves: List[chess.Move] = []
        unindexed_moves: List[chess.Move] = []
        for m in legal_moves:
            flat = self._move_flat_index(m)
            if flat is not None:
                flat_indices.append(flat)
                indexed_moves.append(m)
            else:
                unindexed_moves.append(m)
        if not indexed_moves:
            n = len(legal_moves)
            return [(m, 1.0 / n) for m in legal_moves]
        flat_logits = policy_logits_np.reshape(-1)
        legal_logits = flat_logits[flat_indices].astype(np.float64, copy=False)
        legal_logits = legal_logits - legal_logits.max()
        exp_logits = np.exp(legal_logits)
        probs = exp_logits / exp_logits.sum()
        result = list(zip(indexed_moves, probs.tolist()))
        # Encoding edge cases (rare): give unindexed legal moves a tiny prior.
        result.extend((m, 0.0) for m in unindexed_moves)
        return result

    # --- root management ---

    def make_root(self, board: chess.Board) -> MCTSNode:
        return MCTSNode(board=board.copy(stack=False))

    def advance_root(self, root: MCTSNode, move: chess.Move, game_history: List[str]) -> MCTSNode:
        """Return the child for `move` as the new root, preserving its subtree.

        If `move` isn't in the tree (unvisited branch), build a fresh root.

        Args:
            root: Current root node.
            move: Move to advance by.
            game_history: Played-position keys. Caller appends the post-move key
                BEFORE calling us, so `game_history[-1]` IS the new root's key.
                We strip that tail before counting to avoid double-counting.
        """
        # The caller has just appended the new root's position to game_history.
        # All terminal checks below operate on history WITHOUT that tail entry,
        # since `_check_repetition_terminal` accounts for the new node with +1.
        history_excl_self = game_history[:-1]

        if move in root.children:
            new_root = root.children[move]
            if new_root.board is None:
                # Stub: materialize (will recompute terminal flags from scratch).
                _materialize(new_root, history_excl_self, [root])
            else:
                # Already materialized in a prior search. Refresh terminal status:
                # repetition is history-dependent, so a previously non-terminal
                # node may now be a 3-fold (or vice versa). Intrinsic terminals
                # (mate/stalemate/insufficient material/50-move) are stable, so
                # we re-derive them too rather than trust the stored flag.
                intrinsic = terminal_score_white(new_root.board)
                if intrinsic is not None:
                    new_root.is_terminal = True
                    new_root.terminal_value = intrinsic
                else:
                    new_root.is_terminal = False
                    new_root.terminal_value = None
                    _check_repetition_terminal(new_root, history_excl_self, [])
        else:
            new_board = root.board.copy(stack=False)
            new_board.push(move)
            new_root = MCTSNode(board=new_board)
            intrinsic = terminal_score_white(new_board)
            if intrinsic is not None:
                new_root.is_terminal = True
                new_root.terminal_value = intrinsic
            else:
                _check_repetition_terminal(new_root, history_excl_self, [])
        new_root.parent = None
        new_root.parent_move = None
        return new_root

    # --- MCTS internals ---

    def _expand_node(self, node: MCTSNode, policy_logits_np: np.ndarray):
        """Create children as lazy stubs with priors only.

        Boards (and terminal flags) are materialized on the first descent
        into each child. With ~30 legal moves but only ~5-10 actually visited
        per leaf, this avoids ~70% of board copies vs eager creation.
        """
        ranked = self._legal_priors_from_logits(node.board, policy_logits_np)
        for move, prior in ranked:
            node.children[move] = MCTSNode(
                board=None,
                parent=node,
                parent_move=move,
                prior=prior,
            )
        node.expanded = True

    def _select_child(
        self,
        node: MCTSNode,
        cpuct: float,
        fpu_reduction: float,
    ) -> Optional[MCTSNode]:
        """PUCT selection at an expanded internal node."""
        is_white = (node.turn == chess.WHITE)
        # Effective parent-visits for the explore term.
        total_N_eff = sum(c.N + c.virtual_loss for c in node.children.values())
        sqrt_total = math.sqrt(total_N_eff) if total_N_eff > 0 else 1.0

        # FPU baseline = parent's empirical Q (white's perspective).
        if node.N > 0:
            parent_q_white = node.W / node.N
        else:
            parent_q_white = 0.0
        # FPU reduction shifts unvisited child's Q toward "worse for parent's player".
        unvisited_q_white = (
            parent_q_white - fpu_reduction if is_white else parent_q_white + fpu_reduction
        )

        # Virtual-loss W-shift in WHITE's perspective.
        vl_shift_white = -1.0 if is_white else 1.0

        best_child = None
        best_score = -math.inf
        for child in node.children.values():
            eff_N = child.N + child.virtual_loss
            if eff_N > 0:
                eff_W = child.W + child.virtual_loss * vl_shift_white
                q_white = eff_W / eff_N
            else:
                q_white = unvisited_q_white
            explore = cpuct * child.prior * sqrt_total / (1 + eff_N)
            # Maximize (Q+explore) for white; minimize (Q-explore) for black,
            # equivalently maximize (-Q+explore) for black.
            score = (q_white if is_white else -q_white) + explore
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _descend(
        self,
        root: MCTSNode,
        cpuct: float,
        fpu_reduction: float,
        game_history: List[str],
    ) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Descend by PUCT until we hit an unexpanded or terminal node.

        Each selected child is materialized before continuing — this is where
        a stub child gets its board (and terminal status) for the first time.
        """
        node = root
        path = [node]
        while node.expanded and not node.is_terminal:
            child = self._select_child(node, cpuct, fpu_reduction)
            if child is None:
                break
            _materialize(child, game_history, path)
            node = child
            path.append(node)
        return node, path

    @staticmethod
    def _backprop(path: List[MCTSNode], v_white: float, undo_virtual_loss: bool):
        for n in path:
            if undo_virtual_loss:
                n.virtual_loss -= 1
            n.N += 1
            n.W += v_white

    @staticmethod
    def _is_visit_dominated(
        root: MCTSNode, n_sims: int, n_done: int, min_sims: int
    ) -> bool:
        """Return True iff the argmax-by-visits root child can no longer change.

        Cheap check: scan children once for top and second visit counts. If the
        top child's lead exceeds the sims still to run, no allocation of those
        sims to the runner-up could overtake — so we can bail.

        `min_sims` is a floor: the check is suppressed before that many sims
        complete in this call, to avoid bailing on a tiny early N-disparity.
        """
        if min_sims <= 0 or n_done < min_sims or not root.children:
            return False
        top_n = -1
        second_n = -1
        for c in root.children.values():
            if c.N > top_n:
                second_n = top_n
                top_n = c.N
            elif c.N > second_n:
                second_n = c.N
        if second_n < 0:
            second_n = 0
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
    ) -> Dict[str, Any]:
        """Run up to `n_sims` new simulations from `root` (preserving any prior visits).

        Args:
            root: Root node for the search.
            n_sims: Number of simulations to run.
            batch_size: Batch size for leaf evaluation.
            cpuct: PUCT exploration constant.
            fpu_reduction: First-play urgency reduction.
            game_history: List of `_position_key(board)` strings from the actual
                game history (last entry == root's position).
            early_exit_min_sims: If > 0, after at least this many sims have
                completed, bail out as soon as the most-visited root child's
                lead exceeds the remaining sim budget — i.e. as soon as no
                possible distribution of remaining visits could change the
                argmax-by-N move. Provably doesn't change the chosen move.
                Pass 0 to disable.
        """
        t0 = time.time()
        self._reset_search_stats()

        # Handle terminal root. Distinguish two flavors:
        #   - intrinsic (mate/stalemate/insufficient material/50-move): terminal
        #     no matter the history; we can safely return without expanding.
        #   - stale `is_terminal` flag (e.g. set by a prior repetition check that
        #     no longer applies): the outer game loop already verified the game
        #     isn't over via `board.is_game_over(claim_draw=True)`, so if MCTS
        #     thinks otherwise the flag is stale — clear it and proceed.
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
            # Stale terminal flag (rep check no longer true given current history).
            root.is_terminal = False
            root.terminal_value = None

        # Expand root if needed (single forward pass, policy only).
        if not root.expanded:
            _, p_arr = self._forward_batch(
                [root.board.fen()], need_value=False, need_policy=True
            )
            self._expand_node(root, p_arr[0])
            self.evaluated_positions += 1

        n_done = 0
        early_exit_triggered = False
        while n_done < n_sims:
            # Visit-domination early exit (top of every iteration). Picks up the
            # post-batch state from the previous iteration and any pre-existing
            # tree-reuse imbalance.
            if self._is_visit_dominated(root, n_sims, n_done, early_exit_min_sims):
                early_exit_triggered = True
                break

            target = min(batch_size, n_sims - n_done)
            batch_leaves: List[MCTSNode] = []
            batch_paths: List[List[MCTSNode]] = []

            # Phase 1: collect a batch of non-terminal leaves; resolve terminal
            # leaves immediately (no NN call needed).
            while len(batch_leaves) < target and (n_done + len(batch_leaves)) < n_sims:
                leaf, path = self._descend(root, cpuct, fpu_reduction, game_history)
                if leaf.is_terminal:
                    self._backprop(path, leaf.terminal_value, undo_virtual_loss=False)
                    n_done += 1
                    # Check after each terminal-leaf sim too: forced-move positions
                    # (one legal move that leads to a terminal) hit only this path
                    # and would otherwise burn through every sim before exiting.
                    if self._is_visit_dominated(
                        root, n_sims, n_done, early_exit_min_sims
                    ):
                        early_exit_triggered = True
                        break
                    continue
                # Apply virtual loss along the path so concurrent descents diverge.
                for n in path:
                    n.virtual_loss += 1
                batch_leaves.append(leaf)
                batch_paths.append(path)

            if early_exit_triggered:
                break
            if not batch_leaves:
                # All remaining sims terminated immediately. Done.
                break

            # Phase 2: dedupe leaves (different descents may target the same one
            # despite virtual loss in extreme cases) and run ONE forward pass.
            unique_leaves: List[MCTSNode] = []
            leaf_to_idx: Dict[int, int] = {}
            for leaf in batch_leaves:
                if id(leaf) not in leaf_to_idx:
                    leaf_to_idx[id(leaf)] = len(unique_leaves)
                    unique_leaves.append(leaf)

            leaf_fens = [leaf.board.fen() for leaf in unique_leaves]
            v_arr, p_arr = self._forward_batch(
                leaf_fens, need_value=True, need_policy=True
            )
            self.evaluated_positions += len(unique_leaves)

            # Phase 3: expand each unique leaf.
            for i, leaf in enumerate(unique_leaves):
                if not leaf.expanded:
                    self._expand_node(leaf, p_arr[i])

            # Phase 4: backprop value and undo virtual loss for each batched sim.
            # The visit-domination early-exit check fires at the top of the next
            # iteration of this outer loop, so it sees the post-Phase-4 state.
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

    # --- top-level move selection ---

    def choose_move(
        self,
        root: MCTSNode,
        n_sims: int,
        batch_size: int,
        cpuct: float,
        fpu_reduction: float,
        game_history: List[str],
        early_exit_min_sims: int = 0,
    ) -> Tuple[chess.Move, Dict[str, Any]]:
        """Run sims and return the most-visited root move plus stats.

        Args:
            root: Root node for the search.
            n_sims: Number of simulations.
            batch_size: Batch size for evaluation.
            cpuct: PUCT exploration constant.
            fpu_reduction: First-play urgency reduction.
            game_history: List of `_position_key(board)` strings from the actual
                game history (last entry == root's position).
            early_exit_min_sims: See `run_simulations`. 0 disables.

        Returns:
            Tuple of (best_move, stats_dict)

        stats contains:
          elapsed, simulations, evaluated_positions, batched_eval_calls,
          early_exit (bool: True iff the visit-domination early-exit fired),
          q_for_mover (Q from the side-to-move's perspective on selected child),
          selected_prior, top_moves: List[(move, visits, prior, Q_white)].
        """
        sim_stats = self.run_simulations(
            root, n_sims, batch_size, cpuct, fpu_reduction, game_history,
            early_exit_min_sims=early_exit_min_sims,
        )

        if not root.children:
            raise ValueError("Root has no children; no legal move available.")

        is_white_to_move = (root.turn == chess.WHITE)

        def child_q_white(c: MCTSNode) -> float:
            return (c.W / c.N) if c.N > 0 else 0.0

        # Tie-break by Q from mover's perspective.
        def sort_key(c: MCTSNode):
            qw = child_q_white(c)
            return (c.N, qw if is_white_to_move else -qw)

        best_child = max(root.children.values(), key=sort_key)
        q_white = child_q_white(best_child)
        q_for_mover = q_white if is_white_to_move else -q_white

        sorted_children = sorted(
            root.children.values(),
            key=lambda c: (c.N, c.prior),
            reverse=True,
        )
        top_moves = [
            (c.parent_move, c.N, c.prior, child_q_white(c))
            for c in sorted_children
        ]

        sim_stats.update({
            "q_for_mover": q_for_mover,
            "selected_prior": best_child.prior,
            "top_moves": top_moves,
            "root_visits": root.N,
        })
        return best_child.parent_move, sim_stats


def format_top_moves(
    board: chess.Board,
    top_moves: List[Tuple[chess.Move, int, float, float]],
    k: int,
) -> str:
    parts = []
    for move, visits, prior, q_white in top_moves[:k]:
        parts.append(f"{board.san(move)} N={visits} P={prior:.1%} Qw={q_white:+.2f}")
    return ", ".join(parts)


# =============================================================================
# Stockfish helpers
# =============================================================================


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
    if depth is not None:
        limit = chess.engine.Limit(depth=depth)
    else:
        limit = chess.engine.Limit(time=movetime)
    return engine.play(board, limit).move


# =============================================================================
# One-game driver
# =============================================================================


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

    # Track position history for repetition detection.
    # Use `_position_key` (FEN minus halfmove/fullmove counters): captures pieces,
    # side-to-move, castling rights, and en-passant target — exactly the FIDE-rule
    # equality criterion. `board_fen()` would be wrong: it includes only piece
    # placement, so e.g. a post-castling king/rook return to e1/h1 would be
    # falsely considered equal to the starting position despite different
    # castling rights.
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
            )
            nn_time_total += search_stats["elapsed"]
            ee_tag = "*" if search_stats.get("early_exit") else ""
            comment = (
                f"Q {search_stats['q_for_mover']:+.2f} "
                f"P {search_stats['selected_prior']:.3f} "
                f"N {search_stats['root_visits']}{ee_tag}"
            )
            if verbose:
                print(f"  Top: {format_top_moves(board, search_stats['top_moves'], 5)}")
                print(
                    f"  search: sims={search_stats['simulations']}"
                    f"{' (early exit)' if search_stats.get('early_exit') else ''}, "
                    f"evaluated={search_stats['evaluated_positions']}, "
                    f"batches={search_stats['batched_eval_calls']}, "
                    f"time={search_stats['elapsed']:.2f}s"
                )
        else:
            sf_t0 = time.time()
            move = stockfish_move(stockfish, board, stockfish_time, stockfish_depth)
            sf_time_total += time.time() - sf_t0
            comment = ""

        san = board.san(move)
        move_prefix = (
            f"{board.fullmove_number}. " if board.turn == chess.WHITE
            else f"{board.fullmove_number}..."
        )
        if verbose:
            print(f"{move_prefix}{san}")

        board.push(move)

        # Add new position to history AFTER the move.
        position_history.append(_position_key(board))

        if reuse_tree and mcts_root is not None:
            mcts_root = neural.advance_root(mcts_root, move, position_history)
        pgn_node = pgn_node.add_variation(move)
        if comment:
            pgn_node.comment = comment
        ply += 1

        if progress_callback is not None:
            # board.turn just flipped after push; white-just-moved iff black is now to move.
            progress_callback(ply, san, board.turn == chess.BLACK)

    game.headers["Result"] = board.result(claim_draw=True)
    return game, nn_time_total, sf_time_total


# =============================================================================
# Multiprocessing worker
# =============================================================================


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
    except chess.engine.EngineError:
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
    progress_cb: Optional[Callable[[int, str, bool], None]] = None
    if _WORKER_PROGRESS is not None:
        _WORKER_PROGRESS[pid] = (level, game_idx, neural_color, 0, "", False, time.time())

        def progress_cb(ply: int, san: str, white_just_moved: bool):
            _WORKER_PROGRESS[pid] = (
                level, game_idx, neural_color, ply, san, white_just_moved, time.time()
            )

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
    return (
        level,
        game_idx,
        neural_color,
        str(game),
        game.headers["Result"],
        elapsed,
        nn_time,
        sf_time,
    )


# =============================================================================
# Progress printer (parallel mode)
# =============================================================================


def _format_progress_line(state: tuple) -> str:
    level, game_idx, neural_color, ply, san, white_moved, ts = state
    elapsed = time.time() - ts
    full_move = (ply + 1) // 2
    if ply == 0:
        move_str = "(starting)"
    elif white_moved:
        move_str = f"{full_move}.{san}"
    else:
        move_str = f"{full_move}...{san}"
    side_to_move_now = chess.BLACK if white_moved and ply > 0 else chess.WHITE
    if ply == 0:
        thinking = "neural" if neural_color == chess.WHITE else "stockfish"
    else:
        thinking = "neural" if side_to_move_now == neural_color else "stockfish"
    n_color = "W" if neural_color == chess.WHITE else "B"
    return (
        f"  L{level:>2} g{game_idx + 1} (n={n_color}) ply {ply} {move_str:<10} "
        f"— {thinking} thinking {elapsed:.1f}s"
    )


def _progress_printer_loop(
    progress_dict: Any,
    stop_event: threading.Event,
    interval: float,
    total_tasks: int,
    games_done_ref: List[int],
    print_lock: threading.Lock,
):
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
            print(
                f"\n--- progress ({len(snapshot)} active, "
                f"{done}/{total_tasks} done) ---"
            )
            for _pid, state in snapshot:
                print(_format_progress_line(state))


# =============================================================================
# Main
# =============================================================================


def _classify_result(result: str, neural_color: chess.Color) -> str:
    if result == "1/2-1/2":
        return "draw"
    if (result == "1-0" and neural_color == chess.WHITE) or \
       (result == "0-1" and neural_color == chess.BLACK):
        return "win"
    return "loss"


def main():
    parser = argparse.ArgumentParser(
        description="Play Stockfish vs the trained neural engine using MCTS."
    )
    parser.add_argument("--model", default=None,
                        help="Path to model checkpoint. Defaults to latest attempt_*_epoch*.pt.")
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--stockfish-level-min", type=int, default=5)
    parser.add_argument("--stockfish-level-max", type=int, default=None,
                        help="Defaults to --stockfish-level-min.")
    parser.add_argument("--stockfish-time", type=float, default=0.1)
    parser.add_argument("--stockfish-depth", type=int, default=None)

    # MCTS-specific
    parser.add_argument("--mcts-simulations", type=int, default=800,
                        help="Simulations per move.")
    parser.add_argument("--mcts-batch-size", type=int, default=16,
                        help="Leaves to bundle per batched NN forward pass.")
    parser.add_argument("--mcts-cpuct", type=float, default=2.0,
                        help="PUCT exploration constant.")
    parser.add_argument("--mcts-fpu", type=float, default=0.0,
                        help="First-play urgency reduction (subtracted from parent Q for unvisited children).")
    parser.add_argument("--mcts-no-reuse-tree", action="store_true",
                        help="Disable subtree reuse across moves in a game.")
    parser.add_argument("--mcts-early-exit-min-sims", type=int, default=200,
                        help="Visit-domination early exit: bail once the most-visited "
                             "root child's lead exceeds the remaining sim budget. The "
                             "value is the minimum sims that must complete before the "
                             "check is allowed to fire (prevents bailing on noise). "
                             "0 disables. Default 200.")

    parser.add_argument("--engine-color", choices=["white", "black", "both", "random"], default="both")
    parser.add_argument("--games", type=int, default=2, help="Games per Stockfish level.")
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--eval-batch-size", type=int, default=4096,
                        help="Hard cap on inner forward-pass chunk size (rarely needs raising).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--pgn-out", default="neural_vs_stockfish_mcts.pgn")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--parallel-games", type=int, default=1,
                        help="Run this many games concurrently in worker processes.")
    parser.add_argument("--progress-interval", type=float, default=10.0,
                        help="Snapshot interval in parallel mode (s). 0 disables.")
    args = parser.parse_args()

    print(
        "MCTS settings: "
        f"sims={args.mcts_simulations}, batch={args.mcts_batch_size}, "
        f"cpuct={args.mcts_cpuct}, fpu={args.mcts_fpu}, "
        f"reuse_tree={not args.mcts_no_reuse_tree}, "
        f"early_exit_min_sims={args.mcts_early_exit_min_sims}"
    )

    level_min = args.stockfish_level_min
    level_max = args.stockfish_level_max if args.stockfish_level_max is not None else level_min
    if level_max < level_min:
        parser.error("--stockfish-level-max must be >= --stockfish-level-min")
    levels = list(range(level_min, level_max + 1))

    model_path = args.model or find_latest_model()
    mode = "a" if args.append else "w"

    play_params: Dict[str, Any] = {
        "n_simulations": args.mcts_simulations,
        "mcts_batch_size": args.mcts_batch_size,
        "cpuct": args.mcts_cpuct,
        "fpu_reduction": args.mcts_fpu,
        "reuse_tree": not args.mcts_no_reuse_tree,
        "stockfish_time": args.stockfish_time,
        "stockfish_depth": args.stockfish_depth,
        "max_plies": args.max_plies,
        "early_exit_min_sims": args.mcts_early_exit_min_sims,
    }

    parallel = max(1, args.parallel_games)
    if parallel == 1:
        _run_sequential(args, levels, model_path, mode, play_params)
    else:
        _run_parallel(args, levels, model_path, mode, play_params, parallel)


def _run_sequential(args, levels, model_path, mode, play_params):
    neural = MCTSEngine(model_path, device=args.device, eval_batch_size=args.eval_batch_size)
    print(f"Starting Stockfish: {args.stockfish}")
    stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    summary: List[Tuple[int, int, int, int, float, float]] = []
    try:
        with open(args.pgn_out, mode, encoding="utf-8") as pgn_file:
            for level in levels:
                configure_stockfish(stockfish, level)
                print(f"\n=== Stockfish Skill Level {level} ===")
                wins = losses = draws = 0
                level_nn = 0.0
                level_sf = 0.0
                for game_idx in range(args.games):
                    neural_color = pick_engine_color(args.engine_color, game_idx)
                    print(
                        f"Game {game_idx + 1}/{args.games} (L{level}): neural plays "
                        f"{'white' if neural_color == chess.WHITE else 'black'}"
                    )
                    game, nn_time, sf_time = play_one_game(
                        game_idx=game_idx,
                        neural=neural,
                        stockfish=stockfish,
                        neural_color=neural_color,
                        stockfish_level=level,
                        verbose=not args.quiet,
                        **play_params,
                    )
                    level_nn += nn_time
                    level_sf += sf_time
                    print(game, file=pgn_file, end="\n\n")
                    pgn_file.flush()
                    result = game.headers["Result"]
                    print(f"  Result: {result}  [nn={nn_time:.1f}s sf={sf_time:.1f}s]")

                    outcome = _classify_result(result, neural_color)
                    if outcome == "win":
                        wins += 1
                    elif outcome == "loss":
                        losses += 1
                    else:
                        draws += 1

                print(
                    f"  L{level} record: +{wins} -{losses} ={draws} (of {args.games})"
                    f"  nn={level_nn:.1f}s sf={level_sf:.1f}s"
                )
                summary.append((level, wins, losses, draws, level_nn, level_sf))
    finally:
        stockfish.quit()

    print(f"\nWrote PGN to {args.pgn_out}")
    if len(summary) > 1:
        print("\nSummary across levels:")
        for level, wins, losses, draws, nn_t, sf_t in summary:
            print(f"  L{level:>2}: +{wins} -{losses} ={draws}  nn={nn_t:.1f}s sf={sf_t:.1f}s")
        total_nn = sum(s[4] for s in summary)
        total_sf = sum(s[5] for s in summary)
        print(f"  total: nn={total_nn:.1f}s sf={total_sf:.1f}s")


def _run_parallel(args, levels, model_path, mode, play_params, parallel):
    tasks: List[Tuple[int, int, bool, Dict[str, Any]]] = []
    for level in levels:
        for game_idx in range(args.games):
            neural_color = pick_engine_color(args.engine_color, game_idx)
            tasks.append((level, game_idx, neural_color, play_params))

    n_workers = min(parallel, len(tasks))
    print(
        f"Parallel mode: {n_workers} worker process(es) for {len(tasks)} game(s) "
        f"across {len(levels)} level(s). Per-move logging is suppressed."
    )

    ctx = mp.get_context("spawn")

    progress_dict: Optional[Any] = None
    progress_manager = None
    progress_thread: Optional[threading.Thread] = None
    progress_stop = threading.Event()
    print_lock = threading.Lock()
    games_done_ref = [0]

    if args.progress_interval > 0:
        progress_manager = ctx.Manager()
        progress_dict = progress_manager.dict()
        progress_thread = threading.Thread(
            target=_progress_printer_loop,
            args=(
                progress_dict, progress_stop, args.progress_interval,
                len(tasks), games_done_ref, print_lock,
            ),
            daemon=True,
        )
        progress_thread.start()
        print(f"Progress snapshots every {args.progress_interval:g}s.")

    summary_per_level: Dict[int, List[float]] = {level: [0, 0, 0, 0.0, 0.0] for level in levels}
    overall_start = time.time()

    try:
        with open(args.pgn_out, mode, encoding="utf-8") as pgn_file:
            with ctx.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(
                    model_path, args.device, args.eval_batch_size,
                    args.stockfish, progress_dict,
                ),
            ) as pool:
                for level, game_idx, neural_color, pgn_str, result, game_secs, nn_time, sf_time in \
                        pool.imap_unordered(_run_one_game_in_worker, tasks):
                    pgn_file.write(pgn_str)
                    pgn_file.write("\n\n")
                    pgn_file.flush()

                    outcome = _classify_result(result, neural_color)
                    if outcome == "win":
                        summary_per_level[level][0] += 1
                    elif outcome == "loss":
                        summary_per_level[level][1] += 1
                    else:
                        summary_per_level[level][2] += 1
                    summary_per_level[level][3] += nn_time
                    summary_per_level[level][4] += sf_time

                    games_done_ref[0] += 1
                    color = "white" if neural_color == chess.WHITE else "black"
                    with print_lock:
                        print(
                            f"[{games_done_ref[0]}/{len(tasks)}] L{level} game {game_idx + 1} "
                            f"(neural {color}): {result} in {game_secs:.1f}s "
                            f"[nn={nn_time:.1f}s sf={sf_time:.1f}s]"
                        )
    finally:
        progress_stop.set()
        if progress_thread is not None:
            progress_thread.join(timeout=2)
        if progress_manager is not None:
            progress_manager.shutdown()

    overall_elapsed = time.time() - overall_start
    total_nn = sum(s[3] for s in summary_per_level.values())
    total_sf = sum(s[4] for s in summary_per_level.values())
    print(f"\nWrote PGN to {args.pgn_out}")
    print(f"Wall time for {len(tasks)} games: {overall_elapsed:.1f}s "
          f"({overall_elapsed / len(tasks):.1f}s/game avg)")
    print(f"Aggregate engine time across workers: nn={total_nn:.1f}s sf={total_sf:.1f}s")

    print("\nResults per level:")
    for level in levels:
        w, l, d, nn_t, sf_t = summary_per_level[level]
        print(
            f"  L{level:>2}: +{int(w)} -{int(l)} ={int(d)}  (of {args.games})"
            f"  nn={nn_t:.1f}s sf={sf_t:.1f}s"
        )


if __name__ == "__main__":
    main()
