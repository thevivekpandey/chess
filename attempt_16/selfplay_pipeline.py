#!/usr/bin/env python3
"""
Self-play training pipeline for chess engine.

Orchestrates the complete AlphaZero-style training loop:
1. Generate self-play games (parallel)
2. Add to replay buffer
3. Train for N batches
4. Evaluate new model vs best model
5. Promote if win rate >= threshold
6. Repeat

Usage:
    python selfplay_pipeline.py \\
        --best-model ~/chess/attempt_13/attempt_14b_epoch032.pt \\
        --iterations 100 \\
        --device cuda
"""

import argparse
import os
import shutil
import time
from datetime import datetime

from selfplay_generator import generate_selfplay_games
from replay_buffer import ReplayBuffer
from train_selfplay import train
from evaluate_models import evaluate_models


def run_pipeline(
    best_model_path: str,
    work_dir: str = "selfplay_work",
    iterations: int = 100,
    games_per_iteration: int = 100,
    train_batches: int = 1000,
    eval_games: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = "cpu",
    selfplay_simulations: int = 800,
    eval_simulations: int = 400,
    win_threshold: float = 0.55,
    skip_first_selfplay: bool = False,
    verbose: bool = False,
):
    """
    Run the complete self-play training pipeline.

    Args:
        best_model_path: Path to initial best model
        work_dir: Working directory for pipeline outputs
        iterations: Number of training iterations
        games_per_iteration: Self-play games to generate per iteration
        train_batches: Training batches per iteration
        eval_games: Evaluation games per iteration
        batch_size: Training batch size
        learning_rate: Learning rate
        device: "cpu" or "cuda"
        selfplay_simulations: MCTS simulations for self-play
        eval_simulations: MCTS simulations for evaluation
        win_threshold: Win rate threshold for model promotion
        verbose: Verbose output
    """
    # Create working directories
    os.makedirs(work_dir, exist_ok=True)

    selfplay_dir = os.path.join(work_dir, "selfplay_games")
    training_dir = os.path.join(work_dir, "training_checkpoints")
    models_dir = os.path.join(work_dir, "models")

    os.makedirs(selfplay_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Paths
    replay_buffer_path = os.path.join(work_dir, "replay_buffer.csv")
    current_best_path = os.path.join(models_dir, "best_model.pt")

    # Copy initial model as best model
    if not os.path.exists(current_best_path):
        print(f"Initializing best model from {best_model_path}")
        shutil.copy(best_model_path, current_best_path)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_path, max_positions=500000)

    print("\n" + "=" * 80)
    print("SELF-PLAY TRAINING PIPELINE")
    print("=" * 80)
    print(f"Work directory: {work_dir}")
    print(f"Best model: {current_best_path}")
    print(f"Replay buffer: {replay_buffer_path}")
    print(f"Device: {device}")
    print(f"\nConfiguration:")
    print(f"  Iterations: {iterations}")
    print(f"  Games per iteration: {games_per_iteration}")
    print(f"  Training batches: {train_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Evaluation games: {eval_games}")
    print(f"  Win threshold: {win_threshold * 100:.0f}%")
    print("=" * 80)
    print()

    pipeline_start = time.time()
    promotions = 0

    for iteration in range(iterations):
        iteration_start = time.time()

        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration + 1}/{iterations}")
        print(f"{'=' * 80}\n")

        # Step 1: Generate self-play games
        selfplay_csv = os.path.join(selfplay_dir, f"iteration_{iteration:04d}.csv")

        # Skip self-play generation on first iteration if requested and data exists
        if iteration == 0 and skip_first_selfplay and os.path.exists(selfplay_csv):
            print(f"[1/4] Skipping self-play generation (using existing {selfplay_csv})...")
            selfplay_elapsed = 0.0
        else:
            print(f"[1/4] Generating {games_per_iteration} self-play games...")
            selfplay_start = time.time()
            generate_selfplay_games(
                model_path=current_best_path,
                num_games=games_per_iteration,
                output_csv=selfplay_csv,
                n_simulations=selfplay_simulations,
                temperature_moves=30,  # Use temperature sampling for first 30 moves
                device=device,
                verbose=verbose,
            )
            selfplay_elapsed = time.time() - selfplay_start

        # Step 2: Add to replay buffer
        print(f"\n[2/4] Adding to replay buffer...")
        buffer_start = time.time()
        replay_buffer.add_from_csv(selfplay_csv)
        replay_buffer.print_statistics()
        buffer_elapsed = time.time() - buffer_start

        # Step 3: Train on replay buffer
        print(f"\n[3/4] Training for {train_batches} batches...")
        training_output = os.path.join(training_dir, f"iteration_{iteration:04d}")
        train_start = time.time()

        train(
            model_path=current_best_path,
            replay_buffer_path=replay_buffer_path,
            output_dir=training_output,
            num_batches=train_batches,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            save_every=train_batches,  # Save only at end
            print_every=100,
        )

        new_model_path = os.path.join(training_output, "model_final.pt")
        train_elapsed = time.time() - train_start

        # Step 4: Evaluate new model
        print(f"\n[4/4] Evaluating new model...")
        eval_start = time.time()

        should_promote, eval_stats = evaluate_models(
            new_model_path=new_model_path,
            best_model_path=current_best_path,
            num_games=eval_games,
            n_simulations=eval_simulations,
            device=device,
            win_threshold=win_threshold,
            verbose=verbose,
        )

        eval_elapsed = time.time() - eval_start

        # Promote if better
        if should_promote:
            print(f"\n✓ Promoting new model as best model!")
            shutil.copy(new_model_path, current_best_path)

            # Also save a dated backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(models_dir, f"best_model_{timestamp}.pt")
            shutil.copy(new_model_path, backup_path)
            print(f"  Backup saved to: {backup_path}")

            promotions += 1
        else:
            print(f"\n✗ Keeping current best model")

        # Iteration summary
        iteration_elapsed = time.time() - iteration_start
        print(f"\n{'─' * 80}")
        print(f"Iteration {iteration + 1} complete:")
        print(f"  Self-play: {selfplay_elapsed:.1f}s")
        print(f"  Buffer update: {buffer_elapsed:.1f}s")
        print(f"  Training: {train_elapsed:.1f}s")
        print(f"  Evaluation: {eval_elapsed:.1f}s")
        print(f"  Total: {iteration_elapsed:.1f}s ({iteration_elapsed / 60:.1f} min)")
        print(f"  Win rate: {eval_stats['new_win_rate'] * 100:.1f}%")
        print(f"  Promotions so far: {promotions}/{iteration + 1}")
        print(f"{'─' * 80}")

    # Final summary
    pipeline_elapsed = time.time() - pipeline_start

    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total iterations: {iterations}")
    print(f"Total promotions: {promotions} ({100 * promotions / iterations:.1f}%)")
    print(f"Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed / 3600:.2f} hours)")
    print(f"Average time per iteration: {pipeline_elapsed / iterations:.1f}s")
    print(f"\nBest model: {current_best_path}")
    replay_buffer.print_statistics()
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Self-play training pipeline for chess engine"
    )
    parser.add_argument(
        "--best-model",
        required=True,
        help="Path to initial best model checkpoint"
    )
    parser.add_argument(
        "--work-dir",
        default="selfplay_work",
        help="Working directory for pipeline outputs (default: selfplay_work)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)"
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=100,
        help="Self-play games per iteration (default: 100)"
    )
    parser.add_argument(
        "--train-batches",
        type=int,
        default=1000,
        help="Training batches per iteration (default: 1000)"
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=100,
        help="Evaluation games per iteration (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training and evaluation (default: cpu)"
    )
    parser.add_argument(
        "--selfplay-simulations",
        type=int,
        default=800,
        help="MCTS simulations for self-play (default: 800)"
    )
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=400,
        help="MCTS simulations for evaluation (default: 400)"
    )
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.55,
        help="Win rate threshold for promotion (default: 0.55)"
    )
    parser.add_argument(
        "--skip-first-selfplay",
        action="store_true",
        help="Skip self-play generation on first iteration (use existing data for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Expand ~ in path
    best_model_path = os.path.expanduser(args.best_model)

    if not os.path.exists(best_model_path):
        print(f"ERROR: Best model not found: {best_model_path}")
        exit(1)

    run_pipeline(
        best_model_path=best_model_path,
        work_dir=args.work_dir,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        train_batches=args.train_batches,
        eval_games=args.eval_games,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        selfplay_simulations=args.selfplay_simulations,
        eval_simulations=args.eval_simulations,
        win_threshold=args.win_threshold,
        skip_first_selfplay=args.skip_first_selfplay,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
