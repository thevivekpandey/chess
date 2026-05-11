#!/usr/bin/env python3
"""
Play games between two neural chess models using AlphaZero-style MCTS.

Both models share the architecture defined in chess_engine.ChessNet and use the
*exact same* MCTS search (PUCT, leaf-parallel virtual-loss batching, subtree
reuse, visit-domination early exit) — only the weights differ. The search
machinery is reused verbatim from play_games_mcts.py: this file just swaps the
Stockfish opponent for a second MCTSEngine and bookkeeps the head-to-head score.

Subtree reuse: each model keeps its own tree and advances *its* root through
every played move (its own and the opponent's), so reuse works for both sides.

Note on GPU memory: every worker process loads BOTH models, so with
--parallel-games N you have 2*N copies of the net resident. Dial down
--parallel-games if you OOM.

Usage example:
  ~/myvenv/bin/python play_models_mcts.py \
      --model-a attempt_14_epoch077.pt \
      --model-b attempt_14b_epoch032.pt \
      --games 20 --model-a-color both \
      --mcts-simulations 6000 --mcts-batch-size 16 --mcts-cpuct 2.0 \
      --parallel-games 8 --pgn-out a_vs_b.pgn
"""

import argparse
import multiprocessing as mp
import os
import random
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import chess.pgn

from play_games_mcts import (
    MCTSEngine,
    MCTSNode,
    _classify_result,
    _position_key,
    format_top_moves,
)


def _model_label(path: str, override: Optional[str]) -> str:
    if override:
        return override
    base = os.path.basename(path)
    return base[:-3] if base.endswith(".pt") else base


def pick_a_color(mode: str, game_idx: int) -> chess.Color:
    """Color for model A in this game."""
    if mode == "white":
        return chess.WHITE
    if mode == "black":
        return chess.BLACK
    if mode == "both":
        return chess.WHITE if game_idx % 2 == 0 else chess.BLACK
    return random.choice([chess.WHITE, chess.BLACK])


def _sample_move_by_visits(
    top_moves: List[Tuple[chess.Move, int, float, float]],
    temperature: float,
    rng: random.Random,
) -> Tuple[chess.Move, int, float, float]:
    """Sample one entry from `top_moves` with probability proportional to N**(1/T).

    `top_moves` is MCTSEngine.choose_move's stats list of (move, N, prior, Q_white),
    already sorted by (N, prior) descending. Returns the chosen tuple. Falls back
    to the most-visited move if every visit count is zero or temperature ~= 0.
    """
    if not top_moves:
        raise ValueError("top_moves is empty; no legal move to sample.")
    if temperature <= 1e-6:
        return top_moves[0]
    counts = [n for (_m, n, _p, _q) in top_moves]
    if sum(counts) <= 0:
        return top_moves[0]
    inv_t = 1.0 / temperature
    weights = [float(n) ** inv_t for n in counts]
    return rng.choices(top_moves, weights=weights, k=1)[0]


# =============================================================================
# One-game driver
# =============================================================================


def play_one_game(
    game_idx: int,
    engine_a: MCTSEngine,
    engine_b: MCTSEngine,
    a_color: chess.Color,
    name_a: str,
    name_b: str,
    n_simulations: int,
    mcts_batch_size: int,
    cpuct: float,
    fpu_reduction: float,
    reuse_tree: bool,
    max_plies: int,
    verbose: bool,
    progress_callback: Optional[Callable[[int, str, bool], None]] = None,
    early_exit_min_sims: int = 0,
    root_temperature: float = 1.0,
    root_temp_plies: int = 0,
    seed: Optional[int] = None,
) -> Tuple[chess.pgn.Game, float, float]:
    rng = random.Random(seed)
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"{name_a} vs {name_b} (MCTS)"
    game.headers["Date"] = datetime.utcnow().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_idx + 1)
    game.headers["White"] = name_a if a_color == chess.WHITE else name_b
    game.headers["Black"] = name_b if a_color == chess.WHITE else name_a

    # Each model owns a tree; both advance through every played move.
    root_a: Optional[MCTSNode] = engine_a.make_root(board) if reuse_tree else None
    root_b: Optional[MCTSNode] = engine_b.make_root(board) if reuse_tree else None

    # Position-key history for FIDE-correct repetition detection (see play_games_mcts).
    position_history: List[str] = [_position_key(board)]

    pgn_node = game
    ply = 0
    a_time_total = 0.0
    b_time_total = 0.0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        a_to_move = (board.turn == a_color)
        engine = engine_a if a_to_move else engine_b
        root = root_a if a_to_move else root_b
        if not reuse_tree or root is None:
            root = engine.make_root(board)
            if reuse_tree:
                if a_to_move:
                    root_a = root
                else:
                    root_b = root

        # Opening-phase root sampling: choose the move ~ N**(1/T) instead of
        # argmax-by-visits. While sampling, disable the visit-domination early
        # exit so the visit distribution we sample from reflects the full budget.
        sampling = (ply < root_temp_plies) and (root_temperature > 1e-6)
        eemin = 0 if sampling else early_exit_min_sims

        move, search_stats = engine.choose_move(
            root,
            n_sims=n_simulations,
            batch_size=mcts_batch_size,
            cpuct=cpuct,
            fpu_reduction=fpu_reduction,
            game_history=position_history,
            early_exit_min_sims=eemin,
        )

        sel_prior = search_stats["selected_prior"]
        q_for_mover = search_stats["q_for_mover"]
        if sampling:
            move, _n, sel_prior, q_white = _sample_move_by_visits(
                search_stats["top_moves"], root_temperature, rng
            )
            q_for_mover = q_white if board.turn == chess.WHITE else -q_white

        if a_to_move:
            a_time_total += search_stats["elapsed"]
        else:
            b_time_total += search_stats["elapsed"]

        mover = name_a if a_to_move else name_b
        ee_tag = "*" if search_stats.get("early_exit") else ""
        temp_tag = f" ~T{root_temperature:g}" if sampling else ""
        comment = (
            f"{mover}{temp_tag} Q {q_for_mover:+.2f} "
            f"P {sel_prior:.3f} "
            f"N {search_stats['root_visits']}{ee_tag}"
        )
        if verbose:
            print(f"  Top: {format_top_moves(board, search_stats['top_moves'], 5)}")
            print(
                f"  {mover} search: sims={search_stats['simulations']}"
                f"{' (early exit)' if search_stats.get('early_exit') else ''}"
                f"{' (sampled)' if sampling else ''}, "
                f"evaluated={search_stats['evaluated_positions']}, "
                f"batches={search_stats['batched_eval_calls']}, "
                f"time={search_stats['elapsed']:.2f}s"
            )

        san = board.san(move)
        move_prefix = (
            f"{board.fullmove_number}. " if board.turn == chess.WHITE
            else f"{board.fullmove_number}..."
        )
        if verbose:
            print(f"{move_prefix}{san}")

        board.push(move)
        position_history.append(_position_key(board))

        if reuse_tree:
            if root_a is not None:
                root_a = engine_a.advance_root(root_a, move, position_history)
            if root_b is not None:
                root_b = engine_b.advance_root(root_b, move, position_history)

        pgn_node = pgn_node.add_variation(move)
        if comment:
            pgn_node.comment = comment
        ply += 1

        if progress_callback is not None:
            # board.turn flipped on push; white-just-moved iff black is now to move.
            progress_callback(ply, san, board.turn == chess.BLACK)

    game.headers["Result"] = board.result(claim_draw=True)
    return game, a_time_total, b_time_total


# =============================================================================
# Multiprocessing worker
# =============================================================================


_WORKER_A: Optional[MCTSEngine] = None
_WORKER_B: Optional[MCTSEngine] = None
_WORKER_PROGRESS: Optional[Any] = None


def _init_worker(
    model_a_path: str,
    model_b_path: str,
    device: str,
    eval_batch_size: int,
    progress_dict: Optional[Any],
):
    global _WORKER_A, _WORKER_B, _WORKER_PROGRESS
    _WORKER_A = MCTSEngine(model_a_path, device=device, eval_batch_size=eval_batch_size)
    _WORKER_B = MCTSEngine(model_b_path, device=device, eval_batch_size=eval_batch_size)
    _WORKER_PROGRESS = progress_dict


def _run_one_game_in_worker(task: Tuple[int, bool, Optional[int], str, str, Dict[str, Any]]):
    global _WORKER_A, _WORKER_B, _WORKER_PROGRESS
    game_idx, a_color, seed, name_a, name_b, play_params = task

    pid = os.getpid()
    progress_cb: Optional[Callable[[int, str, bool], None]] = None
    if _WORKER_PROGRESS is not None:
        _WORKER_PROGRESS[pid] = (game_idx, a_color, 0, "", False, time.time())

        def progress_cb(ply: int, san: str, white_just_moved: bool):
            _WORKER_PROGRESS[pid] = (
                game_idx, a_color, ply, san, white_just_moved, time.time()
            )

    start = time.time()
    try:
        game, a_time, b_time = play_one_game(
            game_idx=game_idx,
            engine_a=_WORKER_A,
            engine_b=_WORKER_B,
            a_color=a_color,
            name_a=name_a,
            name_b=name_b,
            verbose=False,
            progress_callback=progress_cb,
            seed=seed,
            **play_params,
        )
    finally:
        if _WORKER_PROGRESS is not None:
            _WORKER_PROGRESS.pop(pid, None)
    elapsed = time.time() - start
    return (
        game_idx,
        a_color,
        str(game),
        game.headers["Result"],
        elapsed,
        a_time,
        b_time,
    )


# =============================================================================
# Progress printer (parallel mode)
# =============================================================================


def _format_progress_line(state: tuple, name_a: str, name_b: str) -> str:
    game_idx, a_color, ply, san, white_moved, ts = state
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
        thinking = name_a if a_color == chess.WHITE else name_b
    else:
        thinking = name_a if side_to_move_now == a_color else name_b
    a_side = "W" if a_color == chess.WHITE else "B"
    return (
        f"  g{game_idx + 1} (A={a_side}) ply {ply} {move_str:<10} "
        f"— {thinking} thinking {elapsed:.1f}s"
    )


def _progress_printer_loop(
    progress_dict: Any,
    stop_event: threading.Event,
    interval: float,
    total_tasks: int,
    games_done_ref: List[int],
    print_lock: threading.Lock,
    name_a: str,
    name_b: str,
):
    while not stop_event.wait(interval):
        try:
            snapshot = list(progress_dict.items())
        except Exception:
            continue
        if not snapshot:
            continue
        snapshot.sort(key=lambda kv: kv[1][0])  # by game_idx
        with print_lock:
            done = games_done_ref[0]
            print(
                f"\n--- progress ({len(snapshot)} active, "
                f"{done}/{total_tasks} done) ---"
            )
            for _pid, state in snapshot:
                print(_format_progress_line(state, name_a, name_b))


# =============================================================================
# Main
# =============================================================================


def _print_match_summary(name_a, name_b, a_wins, b_wins, draws, n_games,
                          a_time_total, b_time_total, time_label):
    score_a = a_wins + 0.5 * draws
    print(f"\nMatch result ({n_games} games):")
    print(f"  {name_a} wins : {a_wins}")
    print(f"  {name_b} wins : {b_wins}")
    print(f"  draws        : {draws}")
    print(f"  score        : {name_a} {score_a:.1f} - {n_games - score_a:.1f} {name_b}")
    print(f"  {time_label}: {name_a}={a_time_total:.1f}s {name_b}={b_time_total:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Play two neural chess models against each other using MCTS."
    )
    parser.add_argument("--model-a", required=True, help="Path to model A checkpoint.")
    parser.add_argument("--model-b", required=True, help="Path to model B checkpoint.")
    parser.add_argument("--name-a", default=None,
                        help="Label for model A in PGN/output. Default: checkpoint filename.")
    parser.add_argument("--name-b", default=None,
                        help="Label for model B in PGN/output. Default: checkpoint filename.")

    # MCTS-specific (shared by both models — same search algo).
    parser.add_argument("--mcts-simulations", type=int, default=800,
                        help="Simulations per move (both models).")
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

    # Opening-phase randomization (so a 20-game match isn't 1 game replayed 20x).
    parser.add_argument("--root-temp-plies", type=int, default=0,
                        help="Number of opening plies (half-moves, counted from move 1, "
                             "both sides) during which the played move is SAMPLED with "
                             "probability ~ N**(1/temperature) over root visit counts, "
                             "instead of argmax. 0 (default) = always argmax = fully "
                             "deterministic. During these plies the visit-domination "
                             "early exit is disabled so the sampled distribution uses "
                             "the full sim budget.")
    parser.add_argument("--root-temperature", type=float, default=1.0,
                        help="Softmax temperature for the sampling above. T=1: sample "
                             "proportional to visits; T>1: flatter; T->0: argmax. "
                             "Only active when --root-temp-plies > 0. Default 1.0.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base RNG seed for reproducible sampling. Game i uses "
                             "seed+i. Omit for nondeterministic sampling.")

    parser.add_argument("--model-a-color", choices=["white", "black", "both", "random"],
                        default="both", help="Which side model A takes (default both: alternates).")
    parser.add_argument("--games", type=int, default=10, help="Total games to play.")
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--eval-batch-size", type=int, default=4096,
                        help="Hard cap on inner forward-pass chunk size (rarely needs raising).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--pgn-out", default="model_a_vs_model_b_mcts.pgn")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--parallel-games", type=int, default=1,
                        help="Run this many games concurrently in worker processes "
                             "(each worker loads BOTH models).")
    parser.add_argument("--progress-interval", type=float, default=10.0,
                        help="Snapshot interval in parallel mode (s). 0 disables.")
    args = parser.parse_args()

    name_a = _model_label(args.model_a, args.name_a)
    name_b = _model_label(args.model_b, args.name_b)
    if name_a == name_b:
        name_a, name_b = f"{name_a}#A", f"{name_b}#B"

    sampling_on = args.root_temp_plies > 0 and args.root_temperature > 1e-6
    print(
        "MCTS settings: "
        f"sims={args.mcts_simulations}, batch={args.mcts_batch_size}, "
        f"cpuct={args.mcts_cpuct}, fpu={args.mcts_fpu}, "
        f"reuse_tree={not args.mcts_no_reuse_tree}, "
        f"early_exit_min_sims={args.mcts_early_exit_min_sims}"
    )
    if sampling_on:
        print(
            f"Root sampling: first {args.root_temp_plies} plies ~ N^(1/{args.root_temperature:g}); "
            f"seed={args.seed if args.seed is not None else 'random'}"
        )
    else:
        print("Root sampling: disabled (deterministic argmax-by-visits) "
              "-> games will repeat; pass --root-temp-plies to diversify")
    print(f"Model A: {args.model_a}  (label: {name_a})")
    print(f"Model B: {args.model_b}  (label: {name_b})")

    mode = "a" if args.append else "w"
    play_params: Dict[str, Any] = {
        "n_simulations": args.mcts_simulations,
        "mcts_batch_size": args.mcts_batch_size,
        "cpuct": args.mcts_cpuct,
        "fpu_reduction": args.mcts_fpu,
        "reuse_tree": not args.mcts_no_reuse_tree,
        "max_plies": args.max_plies,
        "early_exit_min_sims": args.mcts_early_exit_min_sims,
        "root_temperature": args.root_temperature,
        "root_temp_plies": args.root_temp_plies,
    }

    parallel = max(1, args.parallel_games)
    if parallel == 1:
        _run_sequential(args, name_a, name_b, mode, play_params)
    else:
        _run_parallel(args, name_a, name_b, mode, play_params, parallel)


def _run_sequential(args, name_a, name_b, mode, play_params):
    engine_a = MCTSEngine(args.model_a, device=args.device, eval_batch_size=args.eval_batch_size)
    engine_b = MCTSEngine(args.model_b, device=args.device, eval_batch_size=args.eval_batch_size)

    a_wins = b_wins = draws = 0
    a_time_total = b_time_total = 0.0

    with open(args.pgn_out, mode, encoding="utf-8") as pgn_file:
        for game_idx in range(args.games):
            a_color = pick_a_color(args.model_a_color, game_idx)
            game_seed = None if args.seed is None else args.seed + game_idx
            print(
                f"\nGame {game_idx + 1}/{args.games}: {name_a} plays "
                f"{'white' if a_color == chess.WHITE else 'black'}"
            )
            game, a_time, b_time = play_one_game(
                game_idx=game_idx,
                engine_a=engine_a,
                engine_b=engine_b,
                a_color=a_color,
                name_a=name_a,
                name_b=name_b,
                verbose=not args.quiet,
                seed=game_seed,
                **play_params,
            )
            a_time_total += a_time
            b_time_total += b_time
            print(game, file=pgn_file, end="\n\n")
            pgn_file.flush()
            result = game.headers["Result"]
            print(f"  Result: {result}  [{name_a}={a_time:.1f}s {name_b}={b_time:.1f}s]")

            outcome = _classify_result(result, a_color)  # "win" => A won
            if outcome == "win":
                a_wins += 1
            elif outcome == "loss":
                b_wins += 1
            else:
                draws += 1
            print(f"  Running: {name_a} +{a_wins} ={draws} -{b_wins} {name_b}")

    print(f"\nWrote PGN to {args.pgn_out}")
    _print_match_summary(name_a, name_b, a_wins, b_wins, draws, args.games,
                         a_time_total, b_time_total, "think time")


def _run_parallel(args, name_a, name_b, mode, play_params, parallel):
    tasks: List[Tuple[int, bool, Optional[int], str, str, Dict[str, Any]]] = [
        (
            game_idx,
            pick_a_color(args.model_a_color, game_idx),
            None if args.seed is None else args.seed + game_idx,
            name_a,
            name_b,
            play_params,
        )
        for game_idx in range(args.games)
    ]
    n_workers = min(parallel, len(tasks))
    print(
        f"Parallel mode: {n_workers} worker process(es) for {len(tasks)} game(s). "
        "Each worker loads BOTH models; per-move logging is suppressed."
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
                len(tasks), games_done_ref, print_lock, name_a, name_b,
            ),
            daemon=True,
        )
        progress_thread.start()
        print(f"Progress snapshots every {args.progress_interval:g}s.")

    a_wins = b_wins = draws = 0
    a_time_total = b_time_total = 0.0
    overall_start = time.time()

    try:
        with open(args.pgn_out, mode, encoding="utf-8") as pgn_file:
            with ctx.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(
                    args.model_a, args.model_b, args.device,
                    args.eval_batch_size, progress_dict,
                ),
            ) as pool:
                for game_idx, a_color, pgn_str, result, game_secs, a_time, b_time in \
                        pool.imap_unordered(_run_one_game_in_worker, tasks):
                    pgn_file.write(pgn_str)
                    pgn_file.write("\n\n")
                    pgn_file.flush()

                    outcome = _classify_result(result, a_color)  # "win" => A won
                    if outcome == "win":
                        a_wins += 1
                    elif outcome == "loss":
                        b_wins += 1
                    else:
                        draws += 1
                    a_time_total += a_time
                    b_time_total += b_time

                    games_done_ref[0] += 1
                    color = "white" if a_color == chess.WHITE else "black"
                    with print_lock:
                        print(
                            f"[{games_done_ref[0]}/{len(tasks)}] game {game_idx + 1} "
                            f"({name_a} {color}): {result} in {game_secs:.1f}s "
                            f"[{name_a}={a_time:.1f}s {name_b}={b_time:.1f}s] "
                            f"| running {name_a} +{a_wins} ={draws} -{b_wins} {name_b}"
                        )
    finally:
        progress_stop.set()
        if progress_thread is not None:
            progress_thread.join(timeout=2)
        if progress_manager is not None:
            progress_manager.shutdown()

    overall_elapsed = time.time() - overall_start
    print(f"\nWrote PGN to {args.pgn_out}")
    print(f"Wall time for {len(tasks)} games: {overall_elapsed:.1f}s "
          f"({overall_elapsed / len(tasks):.1f}s/game avg)")
    _print_match_summary(name_a, name_b, a_wins, b_wins, draws, len(tasks),
                         a_time_total, b_time_total, "aggregate think time")


if __name__ == "__main__":
    main()
