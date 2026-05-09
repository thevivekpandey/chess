#!/usr/bin/env python3
"""
Evaluate a new model against the current best model.

Plays N games between the two models and determines if the new model
should replace the best model based on win rate.
"""

import argparse
import multiprocessing as mp
import os
import random
import time
from typing import Dict, Optional, Tuple

import chess
import chess.pgn
import numpy as np
import torch

from play_games_mcts import MCTSEngine, MCTSNode, _position_key
from selfplay_generator import sample_move_with_temperature


def play_evaluation_game(
    model1: MCTSEngine,
    model2: MCTSEngine,
    model1_color: chess.Color,
    n_simulations: int,
    mcts_batch_size: int,
    cpuct: float,
    fpu_reduction: float,
    reuse_tree: bool,
    max_plies: int,
    opening_temperature_plies: int = 10,
    opening_temperature: float = 0.5,
    verbose: bool = False,
) -> str:
    """
    Play one game between two models.

    Args:
        model1: First model
        model2: Second model
        model1_color: Color for model1 (WHITE or BLACK)
        n_simulations: MCTS simulations per move
        mcts_batch_size: MCTS batch size
        cpuct: PUCT exploration constant
        fpu_reduction: First-play urgency reduction
        reuse_tree: Whether to reuse MCTS tree
        max_plies: Maximum plies per game
        verbose: Print move-by-move output

    Returns:
        Game result: "1-0", "0-1", or "1/2-1/2"
    """
    board = chess.Board()

    # Initialize roots
    root1: MCTSNode = model1.make_root(board) if reuse_tree else None
    root2: MCTSNode = model2.make_root(board) if reuse_tree else None

    position_history = [_position_key(board)]
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        # Determine which model's turn
        if board.turn == model1_color:
            current_model = model1
            current_root = root1
        else:
            current_model = model2
            current_root = root2

        # Make root if needed
        if not reuse_tree or current_root is None:
            current_root = current_model.make_root(board)

        # Choose move
        move, search_stats = current_model.choose_move(
            current_root,
            n_sims=n_simulations,
            batch_size=mcts_batch_size,
            cpuct=cpuct,
            fpu_reduction=fpu_reduction,
            game_history=position_history,
        )

        # Re-sample first N plies from visit distribution with temperature so
        # parallel "same colors" pairings don't all collapse to the same game.
        if ply < opening_temperature_plies:
            top_moves = search_stats['top_moves']
            if len(top_moves) > 1:
                moves_only = [mv for mv, _, _, _ in top_moves]
                visit_counts = [v for _, v, _, _ in top_moves]
                move = sample_move_with_temperature(
                    moves_only, visit_counts, temperature=opening_temperature
                )

        if verbose:
            san = board.san(move)
            move_prefix = (
                f"{board.fullmove_number}. " if board.turn == chess.WHITE
                else f"{board.fullmove_number}..."
            )
            print(f"{move_prefix}{san}")

        # Make move
        board.push(move)
        position_history.append(_position_key(board))

        # Advance both trees to the new position
        if reuse_tree:
            root1 = model1.advance_root(root1, move, position_history)
            root2 = model2.advance_root(root2, move, position_history)

        ply += 1

    return board.result(claim_draw=True)


# =============================================================================
# Multiprocessing workers for parallel evaluation
# =============================================================================
#
# Each worker loads BOTH the new and best models (its own copies on the target
# device) and plays one full game per task. spawn context is required for CUDA.

_WORKER_NEW_MODEL: Optional[MCTSEngine] = None
_WORKER_BEST_MODEL: Optional[MCTSEngine] = None
_WORKER_PLAY_PARAMS: Optional[Dict] = None


def _init_eval_worker(
    new_model_path: str,
    best_model_path: str,
    device: str,
    eval_batch_size: int,
    play_params: Dict,
    base_seed: int,
):
    """Per-worker initialization for parallel evaluation."""
    global _WORKER_NEW_MODEL, _WORKER_BEST_MODEL, _WORKER_PLAY_PARAMS

    # Cap torch intra-op threads so N workers don't all grab all 24 cores.
    torch.set_num_threads(1)

    # Decorrelate temperature sampling across workers.
    pid = os.getpid()
    seed = (base_seed * 2654435761 + pid) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _WORKER_NEW_MODEL = MCTSEngine(
        new_model_path, device=device, eval_batch_size=eval_batch_size, verbose=False,
    )
    _WORKER_BEST_MODEL = MCTSEngine(
        best_model_path, device=device, eval_batch_size=eval_batch_size, verbose=False,
    )
    _WORKER_PLAY_PARAMS = play_params


def _run_one_eval_game_in_worker(game_idx: int):
    """Play one evaluation game; main process tallies the result."""
    global _WORKER_NEW_MODEL, _WORKER_BEST_MODEL, _WORKER_PLAY_PARAMS

    new_model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK

    start = time.time()
    result = play_evaluation_game(
        model1=_WORKER_NEW_MODEL,
        model2=_WORKER_BEST_MODEL,
        model1_color=new_model_color,
        verbose=False,
        **_WORKER_PLAY_PARAMS,
    )
    elapsed = time.time() - start
    return game_idx, result, new_model_color, elapsed


def evaluate_models(
    new_model_path: str,
    best_model_path: str,
    num_games: int = 100,
    n_simulations: int = 400,  # Lower than self-play for faster evaluation
    mcts_batch_size: int = 16,
    cpuct: float = 2.0,
    fpu_reduction: float = 0.0,
    reuse_tree: bool = True,
    max_plies: int = 300,
    device: str = "cpu",
    eval_batch_size: int = 16,
    opening_temperature_plies: int = 10,
    opening_temperature: float = 0.5,
    verbose: bool = False,
    win_threshold: float = 0.50,
    parallel_games: int = 1,
) -> Tuple[bool, dict]:
    """
    Evaluate new model against best model.

    Args:
        new_model_path: Path to new model checkpoint
        best_model_path: Path to best model checkpoint
        num_games: Number of games to play
        n_simulations: MCTS simulations per move (reduced for speed)
        mcts_batch_size: MCTS batch size
        cpuct: PUCT exploration constant
        fpu_reduction: First-play urgency reduction
        reuse_tree: Whether to reuse MCTS tree
        max_plies: Maximum plies per game
        device: "cpu" or "cuda"
        eval_batch_size: Batch size for NN evaluation
        verbose: Print detailed output
        win_threshold: Minimum win rate to promote new model (default: 55%)

    Returns:
        Tuple of (should_promote, statistics)
    """
    n_workers = max(1, min(parallel_games, num_games))

    print(f"\nPlaying {num_games} evaluation games...")
    print(f"MCTS: simulations={n_simulations}, batch={mcts_batch_size}, cpuct={cpuct}")
    print(f"Parallel workers: {n_workers}")
    print(f"Win threshold: {win_threshold * 100:.0f}%")
    print()

    results = {'new_wins': 0, 'best_wins': 0, 'draws': 0}
    start_time = time.time()

    play_params = dict(
        n_simulations=n_simulations,
        mcts_batch_size=mcts_batch_size,
        cpuct=cpuct,
        fpu_reduction=fpu_reduction,
        reuse_tree=reuse_tree,
        max_plies=max_plies,
        opening_temperature_plies=opening_temperature_plies,
        opening_temperature=opening_temperature,
    )

    def _tally(game_idx: int, result: str, new_model_color: chess.Color):
        if result == "1-0":
            if new_model_color == chess.WHITE:
                results['new_wins'] += 1
                return "New model wins"
            results['best_wins'] += 1
            return "Best model wins"
        if result == "0-1":
            if new_model_color == chess.BLACK:
                results['new_wins'] += 1
                return "New model wins"
            results['best_wins'] += 1
            return "Best model wins"
        results['draws'] += 1
        return "Draw"

    if n_workers == 1:
        # Sequential path: load models in main process and play games one by one.
        print(f"Loading new model from {new_model_path}...")
        new_model = MCTSEngine(new_model_path, device=device, eval_batch_size=eval_batch_size)

        print(f"Loading best model from {best_model_path}...")
        best_model = MCTSEngine(best_model_path, device=device, eval_batch_size=eval_batch_size)

        for game_idx in range(num_games):
            new_model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK

            if verbose:
                print(f"\n=== Game {game_idx + 1}/{num_games} ===")
                print(f"New model: {'White' if new_model_color == chess.WHITE else 'Black'}")

            result = play_evaluation_game(
                model1=new_model,
                model2=best_model,
                model1_color=new_model_color,
                verbose=verbose,
                **play_params,
            )

            outcome = _tally(game_idx, result, new_model_color)
            print(f"Game {game_idx + 1}/{num_games}: {outcome} ({result})")
    else:
        # Parallel path: each worker loads both models and plays one game per
        # task. spawn (not fork) is required for CUDA.
        ctx = mp.get_context("spawn")
        base_seed = int(time.time_ns() & 0xFFFFFFFF)

        completed = 0
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_eval_worker,
            initargs=(new_model_path, best_model_path, device, eval_batch_size,
                      play_params, base_seed),
        ) as pool:
            for game_idx, result, new_model_color, game_elapsed in \
                    pool.imap_unordered(
                        _run_one_eval_game_in_worker, range(num_games)
                    ):
                outcome = _tally(game_idx, result, new_model_color)
                completed += 1
                print(f"[{completed}/{num_games}] Game {game_idx + 1}: {outcome} "
                      f"({result}, {game_elapsed:.1f}s)")

    elapsed = time.time() - start_time

    # Calculate statistics
    total_games = num_games
    new_win_rate = results['new_wins'] / total_games
    best_win_rate = results['best_wins'] / total_games
    draw_rate = results['draws'] / total_games

    # Score = wins + 0.5 * draws (standard chess convention). This is what we
    # gate promotion on, so a draw-heavy improvement still counts.
    new_score = (results['new_wins'] + 0.5 * results['draws']) / total_games

    # Decide if new model should be promoted (strictly greater than threshold)
    should_promote = new_score > win_threshold

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total games: {total_games}")
    print(f"New model wins:  {results['new_wins']:3d} ({new_win_rate * 100:.1f}%)")
    print(f"Best model wins: {results['best_wins']:3d} ({best_win_rate * 100:.1f}%)")
    print(f"Draws:           {results['draws']:3d} ({draw_rate * 100:.1f}%)")
    print(f"New model score: {new_score * 100:.1f}% (wins + 0.5 * draws)")
    print(f"Time: {elapsed:.1f}s ({elapsed / num_games:.1f}s/game)")
    print()

    if should_promote:
        print(f"✓ NEW MODEL PROMOTED (score {new_score * 100:.1f}% > {win_threshold * 100:.0f}%)")
    else:
        print(f"✗ New model NOT promoted (score {new_score * 100:.1f}% <= {win_threshold * 100:.0f}%)")
    print("=" * 60)

    statistics = {
        'total_games': total_games,
        'new_wins': results['new_wins'],
        'best_wins': results['best_wins'],
        'draws': results['draws'],
        'new_win_rate': new_win_rate,
        'best_win_rate': best_win_rate,
        'draw_rate': draw_rate,
        'new_score': new_score,
        'elapsed_time': elapsed,
    }

    return should_promote, statistics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate new model against best model"
    )
    parser.add_argument(
        "--new-model",
        required=True,
        help="Path to new model checkpoint"
    )
    parser.add_argument(
        "--best-model",
        required=True,
        help="Path to current best model checkpoint"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of evaluation games (default: 100)"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="MCTS simulations per move (default: 400)"
    )
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.50,
        help="Minimum win rate to promote (default: 0.50, strictly greater than)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for neural network (default: cpu)"
    )
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=1,
        help="Worker processes for concurrent evaluation games (default: 1)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print move-by-move output"
    )

    args = parser.parse_args()

    should_promote, stats = evaluate_models(
        new_model_path=args.new_model,
        best_model_path=args.best_model,
        num_games=args.games,
        n_simulations=args.simulations,
        device=args.device,
        verbose=args.verbose,
        win_threshold=args.win_threshold,
        parallel_games=args.parallel_games,
    )

    # Exit code: 0 if promoted, 1 if not
    exit(0 if should_promote else 1)


if __name__ == "__main__":
    main()
