"""
Supervised Learning Pipeline with Stockfish

Orchestrates the full training pipeline:
1. Generate training data (policy-only games + SF labeling)
2. Split into train/val sets
3. Train model on SF labels
4. Evaluate against Stockfish
5. Repeat for multiple iterations

This implements "Plan 2" from the discussion - learning from Stockfish as a teacher.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List
import time

from generate_sf_training_data import generate_training_data
from train_supervised import train_supervised
from evaluate_vs_stockfish import evaluate_model
from mix_datasets import mix_datasets


def run_pipeline(
    base_dir: str,
    initial_model_path: str,
    stockfish_path: str,
    foundation_data_path: str = None,
    foundation_ratio: float = 0.3,
    num_iterations: int = 100,
    games_per_iteration: int = 100,
    training_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    sf_depth: int = 20,
    sf_multipv: int = 5,
    eval_games: int = 100,
    initial_sf_level: int = 0,
    num_workers: int = 8,
    device: str = "mps"
):
    """
    Run the full supervised learning pipeline with adaptive Stockfish difficulty.

    ADAPTIVE DIFFICULTY SYSTEM:
    - Starts at initial_sf_level (default: 0)
    - After each iteration, evaluates model at current level
    - If win rate > 60%, automatically increases SF level by 1
    - Continues training indefinitely (or until num_iterations reached)
    - Never stops - progressively harder until model plateaus

    Args:
        base_dir: Base directory for all outputs
        initial_model_path: Path to starting model checkpoint
        stockfish_path: Path to Stockfish binary
        foundation_data_path: Path to foundation dataset (e.g., 3.6M examples). If None, no mixing.
        foundation_ratio: Ratio of foundation data in training mix (default: 0.3 = 30%)
        num_iterations: Maximum number of training iterations (default: 100)
        games_per_iteration: Games to generate per iteration
        training_epochs: Training epochs per iteration
        batch_size: Training batch size
        learning_rate: Learning rate
        sf_depth: Stockfish search depth for labeling
        sf_multipv: Stockfish multi-PV count
        eval_games: Number of evaluation games
        initial_sf_level: Starting Stockfish level (default: 0, auto-advances)
        num_workers: Number of parallel workers
        device: Device for training/inference
    """
    print("=" * 80)
    print("SUPERVISED LEARNING PIPELINE WITH STOCKFISH")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Initial model: {initial_model_path}")
    print(f"Stockfish: {stockfish_path}")
    print(f"Foundation data: {foundation_data_path if foundation_data_path else 'None'}")
    print(f"Foundation ratio: {foundation_ratio if foundation_data_path else 'N/A'}")
    print(f"Iterations: {num_iterations}")
    print(f"Games/iteration: {games_per_iteration}")
    print(f"Training epochs: {training_epochs}")
    print(f"SF depth: {sf_depth}, multi-PV: {sf_multipv}")
    print(f"Eval games: {eval_games}")
    print(f"Starting SF level: {initial_sf_level} (auto-advances when >60%)")
    print()

    # Create directory structure
    os.makedirs(base_dir, exist_ok=True)
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Copy initial model to models directory
    current_model_path = os.path.join(models_dir, "model_M0.pth")
    shutil.copy(initial_model_path, current_model_path)
    print(f"✓ Copied initial model to {current_model_path}")

    # Best model tracking (promotion system)
    best_model_path = os.path.join(models_dir, "model_best.pth")
    shutil.copy(initial_model_path, best_model_path)

    # Adaptive difficulty tracking
    current_sf_level = initial_sf_level
    baseline_score = None  # Will be set in first iteration at each level
    level_history = []  # Track level progressions

    # Log results
    results_log = []

    # Training loop
    for iteration in range(1, num_iterations + 1):
        print("\n" + "=" * 80)
        print(f"ITERATION {iteration}/{num_iterations}")
        print("=" * 80)

        iter_start_time = time.time()

        # Step 1: Generate training data
        print(f"\n[Step 1/4] Generating training data...")
        raw_data_path = os.path.join(data_dir, f"iter{iteration}_raw.csv")

        generate_training_data(
            model_path=current_model_path,
            output_path=raw_data_path,
            stockfish_path=stockfish_path,
            num_games=games_per_iteration,
            num_workers=num_workers,
            temperature=0.8,
            sf_depth=sf_depth,
            sf_multipv=sf_multipv,
            device=device
        )

        # Step 2: Mix with foundation data (if provided)
        train_data_path = os.path.join(data_dir, f"iter{iteration}_train.csv")
        val_data_path = os.path.join(data_dir, f"iter{iteration}_val.csv")

        if foundation_data_path:
            print(f"\n[Step 2/4] Mixing with foundation data ({foundation_ratio:.0%} foundation)...")
            mix_datasets(
                foundation_path=foundation_data_path,
                new_data_path=raw_data_path,
                output_train_path=train_data_path,
                output_val_path=val_data_path,
                foundation_ratio=foundation_ratio,
                val_split=0.1
            )
        else:
            print(f"\n[Step 2/4] Splitting train/val (no foundation mixing)...")
            # Simple split without mixing
            import random
            with open(raw_data_path, 'r') as f:
                header = f.readline()
                rows = f.readlines()

            random.shuffle(rows)
            split_idx = int(len(rows) * 0.9)

            with open(train_data_path, 'w') as f:
                f.write(header)
                f.writelines(rows[:split_idx])

            with open(val_data_path, 'w') as f:
                f.write(header)
                f.writelines(rows[split_idx:])

            print(f"  Train: {split_idx} examples")
            print(f"  Val: {len(rows) - split_idx} examples")

        # Step 3: Train model
        print(f"\n[Step 3/4] Training model...")
        iter_models_dir = os.path.join(models_dir, f"iter{iteration}")

        train_supervised(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            model_path=current_model_path,
            output_path=iter_models_dir,
            num_epochs=training_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            policy_weight=1.0,
            value_weight=1.0,
            device=device,
            save_every=training_epochs  # Only save final
        )

        # Update current model to latest trained
        new_model_path = os.path.join(iter_models_dir, f"model_epoch_{training_epochs}.pth")
        candidate_model_path = os.path.join(models_dir, f"model_M{iteration}.pth")
        shutil.copy(new_model_path, candidate_model_path)

        # Step 4: Evaluate candidate model at current SF level
        print(f"\n[Step 4/4] Evaluating model vs Stockfish level {current_sf_level}...")

        # If baseline not set (first iteration or level just increased), establish it
        if baseline_score is None:
            print(f"  Establishing baseline at level {current_sf_level}...")
            baseline_stats = evaluate_model(
                model_path=best_model_path,  # Current best
                stockfish_path=stockfish_path,
                num_games=eval_games,
                stockfish_level=current_sf_level,
                stockfish_time=0.1,
                device=device
            )
            baseline_score = baseline_stats['score']
            print(f"  Baseline score: {baseline_score:.3f} ({baseline_score:.1%})")

            # Check if already > 60% (skip training, advance level immediately)
            if baseline_score > 0.60:
                print(f"\n  🚀 Baseline already exceeds 60%! Skipping training, advancing to level {current_sf_level + 1}")
                level_history.append({
                    'level': current_sf_level,
                    'final_score': baseline_score,
                    'iterations': 0,
                    'skipped': True
                })
                current_sf_level += 1
                baseline_score = None  # Reset for next level
                continue  # Skip to next iteration
            print()

        # Evaluate candidate
        print(f"  Evaluating candidate (M{iteration})...")
        eval_stats = evaluate_model(
            model_path=candidate_model_path,
            stockfish_path=stockfish_path,
            num_games=eval_games,
            stockfish_level=current_sf_level,
            stockfish_time=0.1,
            device=device
        )
        candidate_score = eval_stats['score']

        # Promotion decision
        print(f"\n{'='*80}")
        print(f"PROMOTION DECISION (Level {current_sf_level})")
        print(f"{'='*80}")
        print(f"  Baseline:          {baseline_score:.3f} ({baseline_score:.1%})")
        print(f"  Candidate (M{iteration}): {candidate_score:.3f} ({candidate_score:.1%})")
        print(f"  Improvement:       {candidate_score - baseline_score:+.3f} ({(candidate_score - baseline_score):.1%})")

        promoted = False
        level_advanced = False

        if candidate_score > baseline_score:
            print(f"\n✓ PROMOTED! Candidate beats baseline.")
            shutil.copy(candidate_model_path, best_model_path)

            # CRITICAL: Advance training trunk to promoted model
            current_model_path = best_model_path
            promoted = True

            # Update baseline for next iteration
            baseline_score = candidate_score

            # Check if exceeds 60% - advance to next level
            if baseline_score > 0.60:
                print(f"\n🎯 Baseline exceeds 60%! Advancing to Stockfish level {current_sf_level + 1}")
                level_history.append({
                    'level': current_sf_level,
                    'final_score': baseline_score,
                    'iterations': iteration if iteration == 1 else len([r for r in results_log if r.get('sf_level') == current_sf_level]),
                    'skipped': False
                })
                current_sf_level += 1
                baseline_score = None  # Reset baseline for new level
                level_advanced = True
        else:
            print(f"\n✗ REJECTED. Candidate does not beat baseline.")
            print(f"  Training trunk reset to best model (discarding M{iteration})")

            # CRITICAL: Reset trunk to best model (don't compound degradation)
            current_model_path = best_model_path

        # Log results
        iter_time = time.time() - iter_start_time
        result = {
            'iteration': iteration,
            'model': f"M{iteration}",
            'sf_level': current_sf_level if not level_advanced else current_sf_level - 1,
            'candidate_score': candidate_score,
            'baseline_score': baseline_score if baseline_score is not None else candidate_score,
            'improvement': candidate_score - (baseline_score if baseline_score is not None else candidate_score),
            'wins': eval_stats['wins'],
            'draws': eval_stats['draws'],
            'losses': eval_stats['losses'],
            'promoted': promoted,
            'level_advanced': level_advanced,
            'time_seconds': iter_time
        }
        results_log.append(result)

        # Save results log
        log_path = os.path.join(logs_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(results_log, f, indent=2)

        print(f"\n✓ Iteration {iteration} complete in {iter_time/60:.1f} minutes")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Status: Completed {iteration} iterations")
    print(f"\nAdaptive Difficulty Progression:")
    if level_history:
        for level_info in level_history:
            status = "SKIPPED (already >60%)" if level_info['skipped'] else f"{level_info['iterations']} iterations"
            print(f"  Level {level_info['level']}: {level_info['final_score']:.1%} ({status})")
    print(f"  Current Level: {current_sf_level}")
    if baseline_score:
        print(f"  Current Baseline: {baseline_score:.1%}")

    print()
    print(f"Results summary:")
    print(f"{'Iter':<6} {'Model':<8} {'Level':<6} {'Score':<8} {'Baseline':<10} {'Δ':<8} {'W':<4} {'D':<4} {'L':<4} {'Status':<15}")
    print("-" * 90)
    for result in results_log:
        status = "✓ PROMOTED" if result['promoted'] else "✗ REJECTED"
        if result.get('level_advanced'):
            status += " [LVL+]"
        print(f"{result['iteration']:<6} {result['model']:<8} {result['sf_level']:<6} {result['candidate_score']:<8.3f} "
              f"{result['baseline_score']:<10.3f} {result['improvement']:<+8.3f} "
              f"{result['wins']:<4} {result['draws']:<4} {result['losses']:<4} {status:<15}")

    print(f"\nBest model: {best_model_path}")
    print(f"Reached Stockfish level: {current_sf_level}")
    if baseline_score:
        print(f"Current baseline at L{current_sf_level}: {baseline_score:.1%}")

    print(f"\nAll results saved to: {logs_dir}")

    # Save level history
    level_log_path = os.path.join(logs_dir, "level_history.json")
    with open(level_log_path, 'w') as f:
        json.dump(level_history, f, indent=2)
    print(f"Level progression saved to: {level_log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run supervised learning pipeline with adaptive Stockfish difficulty")
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory for outputs")
    parser.add_argument("--model", type=str, required=True, help="Initial model checkpoint")
    parser.add_argument("--stockfish", type=str, required=True, help="Path to Stockfish binary")
    parser.add_argument("--foundation-data", type=str, default=None, help="Path to foundation dataset (CSV). If provided, will mix with new data.")
    parser.add_argument("--foundation-ratio", type=float, default=0.3, help="Ratio of foundation data in mix (default: 0.3 = 30%%)")
    parser.add_argument("--iterations", type=int, default=100, help="Max number of iterations (default: 100)")
    parser.add_argument("--games", type=int, default=100, help="Games per iteration")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--sf-depth", type=int, default=20, help="Stockfish depth for labeling")
    parser.add_argument("--sf-multipv", type=int, default=5, help="Stockfish multi-PV count")
    parser.add_argument("--eval-games", type=int, default=100, help="Evaluation games")
    parser.add_argument("--initial-level", type=int, default=0, help="Starting Stockfish level (default: 0, auto-advances)")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    run_pipeline(
        base_dir=args.base_dir,
        initial_model_path=args.model,
        stockfish_path=args.stockfish,
        foundation_data_path=args.foundation_data,
        foundation_ratio=args.foundation_ratio,
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        training_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sf_depth=args.sf_depth,
        sf_multipv=args.sf_multipv,
        eval_games=args.eval_games,
        initial_sf_level=args.initial_level,
        num_workers=args.workers,
        device=args.device
    )
