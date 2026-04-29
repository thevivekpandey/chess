#!/usr/bin/env python3
"""
Play chess interactively against the trained neural chess engine using
AlphaZero-style MCTS (PUCT) with leaf-parallel batched NN evaluation.

The human player enters moves in SAN (Standard Algebraic Notation) on the console.
The engine displays the board, top move candidates, and evaluation after each move.

Usage example:
  ~/myvenv/bin/python play_games_manual_mcts.py \
      --model attempt_14_epoch077.pt \
      --engine-color black \
      --mcts-simulations 800 --mcts-batch-size 16 --mcts-cpuct 2.0
"""

import argparse
import glob
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chess
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


def terminal_score_white(board: chess.Board) -> Optional[float]:
    """Terminal value in WHITE's perspective, or None if not terminal."""
    if board.is_checkmate():
        # Side to move is checkmated -> they lose.
        return TERMINAL_LOSS_WHITE if board.turn == chess.WHITE else TERMINAL_WIN_WHITE
    if board.is_game_over(claim_draw=True):
        return TERMINAL_DRAW
    return None


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


def _materialize(node: MCTSNode) -> None:
    """Build node.board from its parent if it's a lazy stub, and detect terminal status.

    No-op if the node is already materialized. Caller must guarantee
    node.parent.board is set; this is true by construction whenever we are
    descending from root, since each step materializes the child before
    moving into it.
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

    def advance_root(self, root: MCTSNode, move: chess.Move) -> MCTSNode:
        """Return the child for `move` as the new root, preserving its subtree.

        If `move` isn't in the tree (unvisited branch), build a fresh root.
        """
        if move in root.children:
            new_root = root.children[move]
            # Must materialize before clearing parent — _materialize needs parent.board.
            _materialize(new_root)
        else:
            new_board = root.board.copy(stack=False)
            new_board.push(move)
            new_root = MCTSNode(board=new_board)
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
            _materialize(child)
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

    def run_simulations(
        self,
        root: MCTSNode,
        n_sims: int,
        batch_size: int,
        cpuct: float,
        fpu_reduction: float,
    ) -> Dict[str, Any]:
        """Run up to `n_sims` new simulations from `root` (preserving any prior visits)."""
        t0 = time.time()
        self._reset_search_stats()

        # Handle terminal root.
        if root.is_terminal or terminal_score_white(root.board) is not None:
            term = terminal_score_white(root.board)
            if term is not None:
                root.is_terminal = True
                root.terminal_value = term
            return {
                "elapsed": time.time() - t0,
                "simulations": 0,
                "evaluated_positions": 0,
                "batched_eval_calls": 0,
            }

        # Expand root if needed (single forward pass, policy only).
        if not root.expanded:
            _, p_arr = self._forward_batch(
                [root.board.fen()], need_value=False, need_policy=True
            )
            self._expand_node(root, p_arr[0])
            self.evaluated_positions += 1

        n_done = 0
        while n_done < n_sims:
            target = min(batch_size, n_sims - n_done)
            batch_leaves: List[MCTSNode] = []
            batch_paths: List[List[MCTSNode]] = []

            # Phase 1: collect a batch of non-terminal leaves; resolve terminal
            # leaves immediately (no NN call needed).
            while len(batch_leaves) < target and (n_done + len(batch_leaves)) < n_sims:
                leaf, path = self._descend(root, cpuct, fpu_reduction)
                if leaf.is_terminal:
                    self._backprop(path, leaf.terminal_value, undo_virtual_loss=False)
                    n_done += 1
                    continue
                # Apply virtual loss along the path so concurrent descents diverge.
                for n in path:
                    n.virtual_loss += 1
                batch_leaves.append(leaf)
                batch_paths.append(path)

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
            for leaf, path in zip(batch_leaves, batch_paths):
                v_white = float(v_arr[leaf_to_idx[id(leaf)]])
                self._backprop(path, v_white, undo_virtual_loss=True)
                n_done += 1

        return {
            "elapsed": time.time() - t0,
            "simulations": n_done,
            "evaluated_positions": self.evaluated_positions,
            "batched_eval_calls": self.batched_eval_calls,
        }

    # --- top-level move selection ---

    def choose_move(
        self,
        root: MCTSNode,
        n_sims: int,
        batch_size: int,
        cpuct: float,
        fpu_reduction: float,
    ) -> Tuple[chess.Move, Dict[str, Any]]:
        """Run sims and return the most-visited root move plus stats.

        stats contains:
          elapsed, simulations, evaluated_positions, batched_eval_calls,
          q_for_mover (Q from the side-to-move's perspective on selected child),
          selected_prior, top_moves: List[(move, visits, prior, Q_white)].
        """
        sim_stats = self.run_simulations(root, n_sims, batch_size, cpuct, fpu_reduction)

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
# Board display
# =============================================================================


def display_board(board: chess.Board, flip: bool = False):
    """Display the board in ASCII with coordinates."""
    print()
    ranks = range(7, -1, -1) if not flip else range(8)
    for rank in ranks:
        print(f" {rank + 1} ", end="")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                symbol = piece.symbol()
            else:
                symbol = "."
            print(f" {symbol}", end="")
        print()
    print("    a b c d e f g h")
    print()


# =============================================================================
# Human move input
# =============================================================================


def get_human_move(board: chess.Board) -> Optional[chess.Move]:
    """Prompt the human player for a move in SAN notation.

    Returns None if the user wants to quit.
    """
    while True:
        move_str = input("Your move (in SAN, e.g., 'e4', 'Nf3', 'O-O', or 'quit'): ").strip()

        if move_str.lower() in ["quit", "q", "exit"]:
            return None

        if not move_str:
            continue

        try:
            # Try to parse as SAN
            move = board.parse_san(move_str)
            if move in board.legal_moves:
                return move
            else:
                print(f"Illegal move: {move_str}")
        except ValueError:
            print(f"Invalid move notation: {move_str}")
            print("Please use SAN notation (e.g., e4, Nf3, Bxc6, O-O-O, etc.)")


# =============================================================================
# One-game driver
# =============================================================================


def play_one_game(
    neural: MCTSEngine,
    engine_color: chess.Color,
    n_simulations: int,
    mcts_batch_size: int,
    cpuct: float,
    fpu_reduction: float,
    reuse_tree: bool,
    max_plies: int,
    show_engine_analysis: bool,
) -> chess.pgn.Game:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Human vs NeuralEngine(MCTS)"
    game.headers["Date"] = datetime.utcnow().strftime("%Y.%m.%d")
    game.headers["White"] = "NeuralEngine" if engine_color == chess.WHITE else "Human"
    game.headers["Black"] = "Human" if engine_color == chess.WHITE else "NeuralEngine"

    mcts_root: Optional[MCTSNode] = neural.make_root(board) if reuse_tree else None

    pgn_node = game
    ply = 0

    print("\n" + "="*60)
    print("GAME START")
    print("="*60)
    print(f"Engine plays: {'WHITE' if engine_color == chess.WHITE else 'BLACK'}")
    print(f"Human plays:  {'WHITE' if engine_color == chess.BLACK else 'BLACK'}")
    print("="*60)

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        # Display board from human's perspective (flipped if human is black)
        flip_display = (engine_color == chess.WHITE)
        display_board(board, flip=flip_display)

        move_number = board.fullmove_number
        side_name = "White" if board.turn == chess.WHITE else "Black"
        print(f"Move {move_number}, {side_name} to move")

        if board.turn == engine_color:
            # Engine's turn
            print("\nEngine is thinking...")
            if not reuse_tree or mcts_root is None:
                mcts_root = neural.make_root(board)

            move, search_stats = neural.choose_move(
                mcts_root,
                n_sims=n_simulations,
                batch_size=mcts_batch_size,
                cpuct=cpuct,
                fpu_reduction=fpu_reduction,
            )

            san = board.san(move)
            print(f"\nEngine plays: {san}")
            print(f"  Evaluation: Q={search_stats['q_for_mover']:+.2f}")
            print(f"  Search time: {search_stats['elapsed']:.2f}s")
            print(f"  Simulations: {search_stats['simulations']}")

            if show_engine_analysis:
                print(f"  Top moves: {format_top_moves(board, search_stats['top_moves'], 5)}")

            comment = (
                f"Q {search_stats['q_for_mover']:+.2f} "
                f"P {search_stats['selected_prior']:.3f} "
                f"N {search_stats['root_visits']}"
            )
        else:
            # Human's turn
            print("\nLegal moves:")
            legal_sans = [board.san(m) for m in board.legal_moves]
            for i, san_move in enumerate(sorted(legal_sans)):
                print(f"  {san_move}", end="")
                if (i + 1) % 8 == 0:
                    print()
            print("\n")

            move = get_human_move(board)
            if move is None:
                print("\nGame abandoned by human.")
                game.headers["Result"] = "*"
                return game

            san = board.san(move)
            print(f"\nYou play: {san}")
            comment = ""

        board.push(move)
        if reuse_tree and mcts_root is not None:
            mcts_root = neural.advance_root(mcts_root, move)
        pgn_node = pgn_node.add_variation(move)
        if comment:
            pgn_node.comment = comment
        ply += 1

    # Game over
    display_board(board, flip=flip_display)
    result = board.result(claim_draw=True)
    game.headers["Result"] = result

    print("\n" + "="*60)
    print("GAME OVER")
    print("="*60)
    print(f"Result: {result}")

    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    elif board.can_claim_draw():
        print("Draw (by repetition or 50-move rule)")

    print("="*60)

    return game


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Play interactively against the trained neural engine using MCTS."
    )
    parser.add_argument("--model", default=None,
                        help="Path to model checkpoint. Defaults to latest attempt_*_epoch*.pt.")

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

    parser.add_argument("--engine-color", choices=["white", "black"], default="black",
                        help="Which color the engine plays (human plays the opposite).")
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--eval-batch-size", type=int, default=4096,
                        help="Hard cap on inner forward-pass chunk size (rarely needs raising).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--pgn-out", default="human_vs_neural_mcts.pgn")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--no-engine-analysis", action="store_true",
                        help="Hide engine's top move analysis.")
    args = parser.parse_args()

    print(
        "MCTS settings: "
        f"sims={args.mcts_simulations}, batch={args.mcts_batch_size}, "
        f"cpuct={args.mcts_cpuct}, fpu={args.mcts_fpu}, "
        f"reuse_tree={not args.mcts_no_reuse_tree}"
    )

    model_path = args.model or find_latest_model()
    engine_color = chess.WHITE if args.engine_color == "white" else chess.BLACK

    neural = MCTSEngine(model_path, device=args.device, eval_batch_size=args.eval_batch_size)

    game = play_one_game(
        neural=neural,
        engine_color=engine_color,
        n_simulations=args.mcts_simulations,
        mcts_batch_size=args.mcts_batch_size,
        cpuct=args.mcts_cpuct,
        fpu_reduction=args.mcts_fpu,
        reuse_tree=not args.mcts_no_reuse_tree,
        max_plies=args.max_plies,
        show_engine_analysis=not args.no_engine_analysis,
    )

    mode = "a" if args.append else "w"
    with open(args.pgn_out, mode, encoding="utf-8") as pgn_file:
        print(game, file=pgn_file, end="\n\n")

    print(f"\nGame saved to {args.pgn_out}")


if __name__ == "__main__":
    main()
