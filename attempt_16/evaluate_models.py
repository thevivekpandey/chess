#!/usr/bin/env python3
"""
Evaluate a new model against the current best model.

Plays N games between the two models and determines if the new model
should replace the best model based on win rate.
"""

import argparse
import time
from typing import Tuple

import chess
import chess.pgn

from play_games_mcts import MCTSEngine, MCTSNode, _position_key


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
    verbose: bool = False,
    win_threshold: float = 0.55,
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
    print(f"Loading new model from {new_model_path}...")
    new_model = MCTSEngine(new_model_path, device=device, eval_batch_size=eval_batch_size)

    print(f"Loading best model from {best_model_path}...")
    best_model = MCTSEngine(best_model_path, device=device, eval_batch_size=eval_batch_size)

    print(f"\nPlaying {num_games} evaluation games...")
    print(f"MCTS: simulations={n_simulations}, batch={mcts_batch_size}, cpuct={cpuct}")
    print(f"Win threshold: {win_threshold * 100:.0f}%")
    print()

    results = {'new_wins': 0, 'best_wins': 0, 'draws': 0}
    start_time = time.time()

    for game_idx in range(num_games):
        # Alternate colors
        new_model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK

        if verbose:
            print(f"\n=== Game {game_idx + 1}/{num_games} ===")
            print(f"New model: {'White' if new_model_color == chess.WHITE else 'Black'}")

        # Play game
        result = play_evaluation_game(
            model1=new_model,
            model2=best_model,
            model1_color=new_model_color,
            n_simulations=n_simulations,
            mcts_batch_size=mcts_batch_size,
            cpuct=cpuct,
            fpu_reduction=fpu_reduction,
            reuse_tree=reuse_tree,
            max_plies=max_plies,
            verbose=verbose,
        )

        # Update results
        if result == "1-0":
            if new_model_color == chess.WHITE:
                results['new_wins'] += 1
                outcome = "New model wins"
            else:
                results['best_wins'] += 1
                outcome = "Best model wins"
        elif result == "0-1":
            if new_model_color == chess.BLACK:
                results['new_wins'] += 1
                outcome = "New model wins"
            else:
                results['best_wins'] += 1
                outcome = "Best model wins"
        else:
            results['draws'] += 1
            outcome = "Draw"

        print(f"Game {game_idx + 1}/{num_games}: {outcome} ({result})")

    elapsed = time.time() - start_time

    # Calculate statistics
    total_games = num_games
    new_win_rate = results['new_wins'] / total_games
    best_win_rate = results['best_wins'] / total_games
    draw_rate = results['draws'] / total_games

    # Decide if new model should be promoted
    should_promote = new_win_rate >= win_threshold

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total games: {total_games}")
    print(f"New model wins:  {results['new_wins']:3d} ({new_win_rate * 100:.1f}%)")
    print(f"Best model wins: {results['best_wins']:3d} ({best_win_rate * 100:.1f}%)")
    print(f"Draws:           {results['draws']:3d} ({draw_rate * 100:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed / num_games:.1f}s/game)")
    print()

    if should_promote:
        print(f"✓ NEW MODEL PROMOTED (win rate {new_win_rate * 100:.1f}% >= {win_threshold * 100:.0f}%)")
    else:
        print(f"✗ New model NOT promoted (win rate {new_win_rate * 100:.1f}% < {win_threshold * 100:.0f}%)")
    print("=" * 60)

    statistics = {
        'total_games': total_games,
        'new_wins': results['new_wins'],
        'best_wins': results['best_wins'],
        'draws': results['draws'],
        'new_win_rate': new_win_rate,
        'best_win_rate': best_win_rate,
        'draw_rate': draw_rate,
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
        default=0.55,
        help="Minimum win rate to promote (default: 0.55)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for neural network (default: cpu)"
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
    )

    # Exit code: 0 if promoted, 1 if not
    exit(0 if should_promote else 1)


if __name__ == "__main__":
    main()
