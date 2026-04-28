#!/usr/bin/env python3
"""
Play games between Stockfish and the trained neural chess engine.

The neural engine chooses moves by searching legal continuations and using the
policy head to select candidate moves plus the value head to evaluate leaf
positions. Stockfish strength is controlled through its UCI "Skill Level" option.
"""

import argparse
import glob
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

from chess_engine import ChessNet, denormalize_eval, fen_to_tensor, move_to_policy_index


MATE_SCORE = 1000.0


def find_latest_model() -> str:
    candidates = glob.glob("attempt_13_epoch*.pt")
    if candidates:
        return max(candidates, key=os.path.getmtime)
    if os.path.exists("attempt_13.pt"):
        return "attempt_13.pt"
    raise FileNotFoundError(
        "No model checkpoint found. Pass --model, or put attempt_13_epoch*.pt / attempt_13.pt here."
    )


def pick_engine_color(mode: str, game_idx: int) -> chess.Color:
    if mode == "white":
        return chess.WHITE
    if mode == "black":
        return chess.BLACK
    if mode == "both":
        return chess.WHITE if game_idx % 2 == 0 else chess.BLACK
    return random.choice([chess.WHITE, chess.BLACK])


class _SearchNode:
    """Node in the BFS minimax search tree.

    `board` is freed (set to None) once the node has been expanded or
    evaluated, so deep trees don't hold ~thousands of chess.Board copies.
    `turn` is cached at construction so backprop doesn't need the board.
    """

    __slots__ = ("board", "parent_move", "children", "value", "terminal",
                 "policy_prob", "turn")

    def __init__(
        self,
        board: chess.Board,
        parent_move: Optional[chess.Move] = None,
        policy_prob: float = 0.0,
        terminal: bool = False,
        value: Optional[float] = None,
    ):
        self.board = board
        self.parent_move = parent_move
        self.children: List["_SearchNode"] = []
        self.value = value
        self.terminal = terminal
        self.policy_prob = policy_prob
        self.turn = board.turn


class NeuralPolicyEngine:
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
        self.reset_search_stats()

    def reset_search_stats(self):
        self.policy_nodes = 0
        self.leaf_positions = 0
        self.batched_leaf_calls = 0

    @staticmethod
    def terminal_score(board: chess.Board) -> Optional[float]:
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_game_over(claim_draw=True):
            return 0.0
        return None

    def _forward_batch(
        self,
        fens: List[str],
        need_value: bool = False,
        need_policy: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run forward pass(es) over `fens`, chunked by `self.eval_batch_size`.

        Returns (values, policy_logits) where each is None if not requested.
        - values: shape (n,), normalized to [-1, 1]
        - policy_logits: shape (n, 73, 8, 8)
        """
        if not fens:
            return None, None

        val_chunks: Optional[List[np.ndarray]] = [] if need_value else None
        pol_chunks: Optional[List[np.ndarray]] = [] if need_policy else None

        with torch.no_grad():
            for start in range(0, len(fens), self.eval_batch_size):
                self.batched_leaf_calls += 1
                chunk = fens[start:start + self.eval_batch_size]
                tensors = torch.stack([fen_to_tensor(f) for f in chunk]).to(self.device)
                values, policies = self.model(tensors)
                if need_value:
                    val_chunks.append(values.float().cpu().numpy().reshape(-1))
                if need_policy:
                    pol_chunks.append(policies.float().cpu().numpy())

        final_values = np.concatenate(val_chunks) if need_value else None
        final_policies = np.concatenate(pol_chunks, axis=0) if need_policy else None
        return final_values, final_policies

    @staticmethod
    def move_flat_index(move: chess.Move) -> Optional[int]:
        idx = move_to_policy_index(move.uci())
        if idx is None:
            return None
        src_row, src_col, plane = idx
        return plane * 64 + src_row * 8 + src_col

    def _rank_legal_moves_from_logits(
        self,
        board: chess.Board,
        policy_logits_np: np.ndarray,
    ) -> List[Tuple[chess.Move, float]]:
        """Rank `board`'s legal moves using a precomputed (73, 8, 8) policy logits array."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        flat_indices: List[int] = []
        indexed_moves: List[chess.Move] = []
        for move in legal_moves:
            flat_idx = self.move_flat_index(move)
            if flat_idx is not None:
                flat_indices.append(flat_idx)
                indexed_moves.append(move)

        if not indexed_moves:
            return [(move, 0.0) for move in legal_moves]

        flat_logits = policy_logits_np.reshape(-1)
        legal_logits = flat_logits[flat_indices].astype(np.float64, copy=False)
        legal_logits = legal_logits - legal_logits.max()
        exp_logits = np.exp(legal_logits)
        probs = exp_logits / exp_logits.sum()

        return sorted(zip(indexed_moves, probs.tolist()), key=lambda item: item[1], reverse=True)

    def policy_candidates(
        self,
        ranked: List[Tuple[chess.Move, float]],
        top_k: int,
        threshold: float,
        min_moves: int,
    ) -> List[Tuple[chess.Move, float]]:
        if not ranked:
            return []

        selected = []
        cumulative = 0.0
        hard_limit = max(1, min(top_k, len(ranked)))
        min_moves = max(1, min(min_moves, hard_limit))
        threshold = max(0.0, min(1.0, threshold))

        for move, prob in ranked:
            selected.append((move, prob))
            cumulative += prob
            if len(selected) >= min_moves and cumulative >= threshold:
                break
            if len(selected) >= hard_limit:
                break

        return selected

    def _backprop(self, node: _SearchNode) -> None:
        if not node.children:
            return  # value already set (terminal or NN-evaluated leaf)
        for child in node.children:
            self._backprop(child)
        child_values = [c.value for c in node.children]
        node.value = max(child_values) if node.turn == chess.WHITE else min(child_values)

    def choose_move(
        self,
        board: chess.Board,
        depth: int,
        policy_topk: int,
        policy_threshold: float,
        min_policy_moves: int,
    ) -> Tuple[chess.Move, float, float, List[Tuple[chess.Move, float]], dict]:
        """Level-synchronous BFS minimax search.

        Builds the tree level by level. At each internal level, runs ONE
        batched forward pass over all open nodes to get policy logits, then
        expands each node's policy-trimmed children. After the deepest level,
        runs ONE batched value-head pass over the leaves and backpropagates
        minimax up the tree.

        This collapses what was ~k^d sequential single-board policy calls
        into d+1 batched forward passes per move.
        """
        self.reset_search_stats()
        start_time = time.time()
        depth = max(1, depth)
        use_policy = policy_topk > 0

        root = _SearchNode(board=board.copy(stack=False))
        current_level: List[_SearchNode] = [root]
        root_ranked: List[Tuple[chess.Move, float]] = []

        # Internal expansion: build levels 1..depth
        for _ in range(depth):
            eval_nodes = [n for n in current_level if not n.terminal]
            if not eval_nodes:
                break

            policy_logits_np: Optional[np.ndarray] = None
            if use_policy:
                fens = [n.board.fen() for n in eval_nodes]
                _, policy_logits_np = self._forward_batch(
                    fens, need_value=False, need_policy=True
                )
                self.policy_nodes += len(eval_nodes)

            next_level: List[_SearchNode] = []
            for i, n in enumerate(eval_nodes):
                if use_policy:
                    ranked = self._rank_legal_moves_from_logits(n.board, policy_logits_np[i])
                    candidates = self.policy_candidates(
                        ranked, policy_topk, policy_threshold, min_policy_moves
                    )
                    if not candidates:
                        candidates = [(m, 0.0) for m in n.board.legal_moves]
                else:
                    ranked = [(m, 0.0) for m in n.board.legal_moves]
                    candidates = ranked

                if n is root:
                    root_ranked = ranked

                for move, prob in candidates:
                    child_board = n.board.copy(stack=False)
                    child_board.push(move)
                    term = self.terminal_score(child_board)
                    child = _SearchNode(
                        board=child_board,
                        parent_move=move,
                        policy_prob=prob,
                        terminal=term is not None,
                        value=term,
                    )
                    n.children.append(child)
                    next_level.append(child)

            # Internal-node boards no longer needed after expansion
            for n in eval_nodes:
                n.board = None

            current_level = next_level

        # Leaf-level value evaluation for non-terminal leaves
        leaf_eval_nodes = [n for n in current_level if not n.terminal]
        if leaf_eval_nodes:
            leaf_fens = [n.board.fen() for n in leaf_eval_nodes]
            leaf_values, _ = self._forward_batch(
                leaf_fens, need_value=True, need_policy=False
            )
            self.leaf_positions += len(leaf_eval_nodes)
            for n, v in zip(leaf_eval_nodes, leaf_values):
                n.value = float(denormalize_eval(v))
        for n in current_level:
            n.board = None

        if not root.children:
            raise ValueError("No legal moves available")

        # Backprop minimax
        self._backprop(root)

        # Pick best at root (max for white-to-move, min for black-to-move)
        if root.turn == chess.WHITE:
            best_child = max(root.children, key=lambda c: c.value)
        else:
            best_child = min(root.children, key=lambda c: c.value)

        elapsed = time.time() - start_time
        stats = {
            "elapsed": elapsed,
            "root_candidates": len(root.children),
            "policy_nodes": self.policy_nodes,
            "leaf_positions": self.leaf_positions,
            "batched_leaf_calls": self.batched_leaf_calls,
        }

        ranked_for_print = root_ranked[:policy_topk] if use_policy else root_ranked
        return (
            best_child.parent_move,
            best_child.value,
            best_child.policy_prob,
            ranked_for_print,
            stats,
        )


def format_policy_moves(board: chess.Board, ranked_moves: List[Tuple[chess.Move, float]]) -> str:
    parts = [f"{board.san(move)} {prob:.1%}" for move, prob in ranked_moves]
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
    if depth is not None:
        limit = chess.engine.Limit(depth=depth)
    else:
        limit = chess.engine.Limit(time=movetime)
    result = engine.play(board, limit)
    return result.move


def play_one_game(
    game_idx: int,
    neural: NeuralPolicyEngine,
    stockfish: chess.engine.SimpleEngine,
    neural_color: chess.Color,
    neural_depth: int,
    policy_topk: int,
    policy_threshold: float,
    min_policy_moves: int,
    stockfish_time: float,
    stockfish_depth: Optional[int],
    stockfish_level: int,
    max_plies: int,
    verbose: bool,
    progress_callback: Optional[Callable[[int, str, bool], None]] = None,
) -> Tuple[chess.pgn.Game, float, float]:
    board = chess.Board()
    game = chess.pgn.Game()
    sf_name = f"Stockfish-L{stockfish_level}"
    game.headers["Event"] = f"NeuralEngine vs {sf_name}"
    game.headers["Date"] = datetime.utcnow().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_idx + 1)
    game.headers["White"] = "NeuralEngine" if neural_color == chess.WHITE else sf_name
    game.headers["Black"] = sf_name if neural_color == chess.WHITE else "NeuralEngine"

    node = game
    ply = 0
    nn_time_total = 0.0
    sf_time_total = 0.0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if board.turn == neural_color:
            move, score, policy_prob, ranked_policy_moves, search_stats = neural.choose_move(
                board,
                neural_depth,
                policy_topk,
                policy_threshold,
                min_policy_moves,
            )
            nn_time_total += search_stats["elapsed"]
            actor = "Neural"
            comment = f"eval {score:+.2f}, policy {policy_prob:.3f}"
        else:
            sf_t0 = time.time()
            move = stockfish_move(stockfish, board, stockfish_time, stockfish_depth)
            sf_time_total += time.time() - sf_t0
            actor = "Stockfish"
            comment = ""

        move_prefix = f"{board.fullmove_number}. " if board.turn == chess.WHITE else f"{board.fullmove_number}..."
        san = board.san(move)
        if verbose:
            if board.turn == neural_color and ranked_policy_moves:
                print(f"Top {len(ranked_policy_moves)}: {format_policy_moves(board, ranked_policy_moves)}")
            print(f"{move_prefix}{san}")
            if board.turn == neural_color:
                print(
                    "  search: "
                    f"depth={neural_depth}, root_candidates={search_stats['root_candidates']}, "
                    f"policy_nodes={search_stats['policy_nodes']}, leaves={search_stats['leaf_positions']}, "
                    f"time={search_stats['elapsed']:.2f}s"
                )

        board.push(move)
        node = node.add_variation(move)
        if comment:
            node.comment = comment
        ply += 1

        if progress_callback is not None:
            # board.turn flipped after push, so white just moved iff it's now black to move
            progress_callback(ply, san, board.turn == chess.BLACK)

    game.headers["Result"] = board.result(claim_draw=True)
    return game, nn_time_total, sf_time_total


# =============================================================================
# Multiprocessing worker (used when --parallel-games > 1)
#
# Each pool worker loads its own model on the GPU and spawns its own Stockfish
# subprocess. Worker globals persist across tasks, so model + Stockfish are
# initialized once per worker.
# =============================================================================

_WORKER_NEURAL: Optional[NeuralPolicyEngine] = None
_WORKER_STOCKFISH: Optional[chess.engine.SimpleEngine] = None
_WORKER_LAST_LEVEL: Optional[int] = None
_WORKER_PROGRESS: Optional[Any] = None  # Manager().dict() proxy or None


def _init_worker(
    model_path: str,
    device: str,
    eval_batch_size: int,
    stockfish_path: str,
    progress_dict: Optional[Any],
):
    global _WORKER_NEURAL, _WORKER_STOCKFISH, _WORKER_LAST_LEVEL, _WORKER_PROGRESS
    _WORKER_NEURAL = NeuralPolicyEngine(
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
    """Pool worker: play one game and return its PGN string + metadata."""
    global _WORKER_NEURAL, _WORKER_STOCKFISH, _WORKER_LAST_LEVEL, _WORKER_PROGRESS
    level, game_idx, neural_color, play_params = task

    if level != _WORKER_LAST_LEVEL:
        configure_stockfish(_WORKER_STOCKFISH, level)
        _WORKER_LAST_LEVEL = level

    pid = os.getpid()
    progress_cb: Optional[Callable[[int, str, bool], None]] = None
    if _WORKER_PROGRESS is not None:
        # Initial slot so the snapshot can show this game from ply 0.
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

    # After the move, the side that just moved is white_moved; the side now
    # thinking is the opposite side.
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
            continue  # manager may be shutting down
        if not snapshot:
            continue
        snapshot.sort(key=lambda kv: (kv[1][0], kv[1][1]))  # by (level, game_idx)
        with print_lock:
            done = games_done_ref[0]
            print(
                f"\n--- progress ({len(snapshot)} active, "
                f"{done}/{total_tasks} done) ---"
            )
            for _pid, state in snapshot:
                print(_format_progress_line(state))


def main():
    parser = argparse.ArgumentParser(description="Play Stockfish vs the trained neural engine.")
    parser.add_argument("--model", default=None, help="Path to model checkpoint. Defaults to latest attempt_13_epoch*.pt.")
    parser.add_argument("--stockfish", default="/usr/games/stockfish", help="Path to Stockfish binary.")
    parser.add_argument("--stockfish-level-min", type=int, default=5, help="Minimum Stockfish Skill Level, 0-20.")
    parser.add_argument("--stockfish-level-max", type=int, default=None, help="Maximum Stockfish Skill Level (defaults to min).")
    parser.add_argument("--stockfish-time", type=float, default=0.1, help="Stockfish seconds per move.")
    parser.add_argument("--stockfish-depth", type=int, default=None, help="Optional fixed Stockfish search depth.")
    parser.add_argument("--engine-depth", type=int, default=1, help="Neural engine minimax depth in plies.")
    parser.add_argument("--policy-topk", type=int, default=8, help="Max legal policy moves searched by neural engine. Use 0 for all legal moves.")
    parser.add_argument("--policy-threshold", type=float, default=0.95, help="Stop adding policy moves after this cumulative probability.")
    parser.add_argument("--min-policy-moves", type=int, default=3, help="Minimum policy candidates to search.")
    parser.add_argument("--engine-color", choices=["white", "black", "both", "random"], default="both")
    parser.add_argument("--games", type=int, default=2, help="Games per Stockfish level.")
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--pgn-out", default="neural_vs_stockfish.pgn")
    parser.add_argument("--append", action="store_true", help="Append to PGN file instead of overwriting.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=1,
        help="Run this many games concurrently in worker processes. 1 = sequential (verbose). >1 disables per-move logging.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="In parallel mode, print a snapshot of all active games every N seconds. 0 to disable.",
    )
    args = parser.parse_args()

    effective_min_policy_moves = args.min_policy_moves
    if args.policy_topk > 0:
        effective_min_policy_moves = min(args.min_policy_moves, args.policy_topk)

    print(
        "Neural settings: "
        f"depth={max(1, args.engine_depth)}, policy_topk={args.policy_topk}, "
        f"policy_threshold={args.policy_threshold}, min_policy_moves={args.min_policy_moves}"
    )
    if args.policy_topk > 0 and effective_min_policy_moves != args.min_policy_moves:
        print(
            "Warning: min_policy_moves is capped by policy_topk, so the effective "
            f"minimum is {effective_min_policy_moves}."
        )
    if args.policy_topk == 1:
        print("Warning: policy_topk=1 creates a single-line search; larger depths will be much cheaper.")

    level_min = args.stockfish_level_min
    level_max = args.stockfish_level_max if args.stockfish_level_max is not None else level_min
    if level_max < level_min:
        parser.error("--stockfish-level-max must be >= --stockfish-level-min")
    levels = list(range(level_min, level_max + 1))

    model_path = args.model or find_latest_model()
    mode = "a" if args.append else "w"

    play_params: Dict[str, Any] = {
        "neural_depth": max(1, args.engine_depth),
        "policy_topk": args.policy_topk,
        "policy_threshold": args.policy_threshold,
        "min_policy_moves": args.min_policy_moves,
        "stockfish_time": args.stockfish_time,
        "stockfish_depth": args.stockfish_depth,
        "max_plies": args.max_plies,
    }

    parallel = max(1, args.parallel_games)

    if parallel == 1:
        _run_sequential(args, levels, model_path, mode, play_params)
    else:
        _run_parallel(args, levels, model_path, mode, play_params, parallel)


def _classify_result(result: str, neural_color: chess.Color) -> str:
    """Returns 'win', 'loss', or 'draw' from the neural engine's perspective."""
    if result == "1/2-1/2":
        return "draw"
    if (result == "1-0" and neural_color == chess.WHITE) or \
       (result == "0-1" and neural_color == chess.BLACK):
        return "win"
    return "loss"


def _run_sequential(args, levels, model_path, mode, play_params):
    neural = NeuralPolicyEngine(model_path, device=args.device, eval_batch_size=args.eval_batch_size)

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
    # Build all (level, game_idx) tasks. Each level rotates colors via game_idx
    # so --engine-color both stays balanced per level.
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

    # 'spawn' avoids the well-known fork-after-CUDA-init pitfalls.
    ctx = mp.get_context("spawn")

    progress_dict: Optional[Any] = None
    progress_manager = None
    progress_thread: Optional[threading.Thread] = None
    progress_stop = threading.Event()
    print_lock = threading.Lock()
    games_done_ref = [0]  # mutable list cell so printer thread sees updates

    if args.progress_interval > 0:
        progress_manager = ctx.Manager()
        progress_dict = progress_manager.dict()
        progress_thread = threading.Thread(
            target=_progress_printer_loop,
            args=(
                progress_dict,
                progress_stop,
                args.progress_interval,
                len(tasks),
                games_done_ref,
                print_lock,
            ),
            daemon=True,
        )
        progress_thread.start()
        print(f"Progress snapshots every {args.progress_interval:g}s.")

    # Per level: [wins, losses, draws, nn_time_total, sf_time_total]
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
