#!/usr/bin/env python3
"""
Train neural network on self-play data using AlphaZero-style training.

Loss function (matches supervised training in chess_engine.py):
  - Policy loss: Cross-entropy between MCTS visit distribution and NN policy
  - Value loss: SmoothL1(beta=0.2) between game outcome and NN value prediction
  - Total loss = value_loss + 0.1 * policy_loss
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chess_engine import ChessNet, fen_to_tensor, move_to_policy_index
from replay_buffer import ReplayBuffer


def normalize_visit_distribution(visit_counts: List[int]) -> np.ndarray:
    """
    Normalize MCTS visit counts to a probability distribution.

    Args:
        visit_counts: List of visit counts for each legal move

    Returns:
        Normalized probability distribution (sums to 1.0)
    """
    total_visits = sum(visit_counts)
    if total_visits == 0:
        # Uniform distribution if no visits
        return np.ones(len(visit_counts)) / len(visit_counts)
    return np.array(visit_counts, dtype=np.float32) / total_visits


def create_policy_target(
    board: chess.Board,
    move_list: List[str],
    visit_counts: List[int],
) -> torch.Tensor:
    """
    Create policy target tensor from MCTS visit distribution.

    Args:
        board: Chess board position
        move_list: List of legal moves in UCI format
        visit_counts: Visit count for each move

    Returns:
        Policy target tensor of shape (73, 8, 8) with visit distribution
        (matches model output format)
    """
    policy_target = torch.zeros(73, 8, 8, dtype=torch.float32)

    # Normalize visit counts
    visit_probs = normalize_visit_distribution(visit_counts)

    # Fill in policy target
    for move_uci, prob in zip(move_list, visit_probs):
        policy_idx = move_to_policy_index(move_uci)
        if policy_idx is not None:
            src_row, src_col, plane_idx = policy_idx
            policy_target[plane_idx, src_row, src_col] = float(prob)  # (73, 8, 8) format

    return policy_target


def prepare_batch(
    examples: List[Dict],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of training examples for training.

    Args:
        examples: List of training examples from replay buffer
        device: "cpu" or "cuda"

    Returns:
        Tuple of (board_tensors, policy_targets, value_targets, masks)
        - board_tensors: (batch_size, 18, 8, 8)
        - policy_targets: (batch_size, 73, 8, 8)
        - value_targets: (batch_size, 1)
        - masks: (batch_size, 73, 8, 8) - mask for legal moves
    """
    batch_size = len(examples)

    board_tensors = torch.zeros(batch_size, 18, 8, 8, dtype=torch.float32)
    policy_targets = torch.zeros(batch_size, 73, 8, 8, dtype=torch.float32)
    value_targets = torch.zeros(batch_size, 1, dtype=torch.float32)
    masks = torch.zeros(batch_size, 73, 8, 8, dtype=torch.float32)

    for i, example in enumerate(examples):
        # Parse example
        fen = example['fen']
        policy_moves_str = example['policy_moves']
        policy_visits_str = example['policy_visits']
        value_target = float(example['value_target'])

        # Parse moves and visits
        if policy_moves_str:
            move_list = policy_moves_str.split(',')
        else:
            move_list = []

        if policy_visits_str:
            visit_counts = [int(v) for v in policy_visits_str.split(',')]
        else:
            visit_counts = []

        # Create board from FEN
        board = chess.Board(fen)

        # Convert FEN to tensor
        board_tensor = fen_to_tensor(fen)
        board_tensors[i] = board_tensor

        # Create policy target
        policy_target = create_policy_target(board, move_list, visit_counts)
        policy_targets[i] = policy_target

        # Create mask for legal moves
        for move_uci in move_list:
            policy_idx = move_to_policy_index(move_uci)
            if policy_idx is not None:
                src_row, src_col, plane_idx = policy_idx
                masks[i, plane_idx, src_row, src_col] = 1.0

        # Value target
        value_targets[i, 0] = value_target

    # Move to device
    board_tensors = board_tensors.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device)
    masks = masks.to(device)

    return board_tensors, policy_targets, value_targets, masks


def compute_loss(
    model: ChessNet,
    board_tensors: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute training loss.

    Args:
        model: Neural network model
        board_tensors: Input board tensors
        policy_targets: Target policy distributions
        value_targets: Target value labels
        masks: Masks for legal moves

    Returns:
        Tuple of (total_loss, policy_loss, value_loss)
    """
    # Forward pass
    value_pred, policy_logits = model(board_tensors)  # Model returns (value, policy_logits)

    # Policy loss: Cross-entropy on masked legal moves
    # Apply mask to logits (set illegal moves to large negative value)
    masked_logits = policy_logits + (1.0 - masks) * (-1e9)

    # Flatten for cross-entropy
    # Shape: (batch_size, 8*8*73)
    masked_logits_flat = masked_logits.view(masked_logits.size(0), -1)
    policy_targets_flat = policy_targets.view(policy_targets.size(0), -1)

    # Log softmax + negative log likelihood
    log_probs = F.log_softmax(masked_logits_flat, dim=1)
    policy_loss = -(policy_targets_flat * log_probs).sum(dim=1).mean()

    # Value loss: SmoothL1 (beta=0.2), matching supervised training in chess_engine.py
    value_loss = F.smooth_l1_loss(value_pred, value_targets, beta=0.2)

    # Total loss: value + 0.1 * policy, matching supervised training (policy_weight=0.1)
    total_loss = value_loss + 0.1 * policy_loss

    return total_loss, policy_loss, value_loss


def train_on_batch(
    model: ChessNet,
    optimizer: optim.Optimizer,
    examples: List[Dict],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train on a single batch.

    Args:
        model: Neural network model
        optimizer: Optimizer
        examples: List of training examples
        device: "cpu" or "cuda"

    Returns:
        Dictionary with loss statistics
    """
    model.train()
    optimizer.zero_grad()

    # Prepare batch
    board_tensors, policy_targets, value_targets, masks = prepare_batch(examples, device)

    # Compute loss
    total_loss, policy_loss, value_loss = compute_loss(
        model, board_tensors, policy_targets, value_targets, masks
    )

    # Backward pass
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
    }


def train(
    model_path: str,
    replay_buffer_path: str,
    output_dir: str,
    num_batches: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = "cpu",
    save_every: int = 100,
    print_every: int = 10,
    optimizer_state_path: str = None,
):
    """
    Train neural network on self-play data.

    Args:
        model_path: Path to initial model checkpoint
        replay_buffer_path: Path to replay buffer CSV
        output_dir: Directory to save model checkpoints
        num_batches: Number of batches to train
        batch_size: Batch size
        learning_rate: Learning rate
        device: "cpu" or "cuda"
        save_every: Save checkpoint every N batches
        print_every: Print stats every N batches
        optimizer_state_path: If set, load Adam state from this path at start
            and save it back at the end. Lets pipeline iterations preserve
            Adam's running gradient statistics across restarts.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    model = ChessNet()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Warm-start Adam from a previous iteration's running statistics. If the
    # file doesn't exist (first iteration), Adam starts cold — that's fine.
    if optimizer_state_path is not None and os.path.exists(optimizer_state_path):
        try:
            opt_state = torch.load(optimizer_state_path, map_location=device)
            optimizer.load_state_dict(opt_state)
            print(f"Loaded optimizer state from {optimizer_state_path}")
        except Exception as e:
            print(f"WARN: could not load optimizer state from {optimizer_state_path}: {e}")

    replay_buffer = ReplayBuffer(replay_buffer_path, max_positions=500000)

    if replay_buffer.is_empty():
        print("ERROR: Replay buffer is empty! Generate self-play data first.")
        return

    print(f"Starting training for {num_batches} batches "
          f"(buffer={replay_buffer.size()}, batch={batch_size}, lr={learning_rate}, device={device})")

    start_time = time.time()
    running_losses = {'total': 0.0, 'policy': 0.0, 'value': 0.0}

    for batch_idx in range(num_batches):
        # Sample batch from replay buffer
        examples = replay_buffer.sample_batch(batch_size)

        # Train on batch
        losses = train_on_batch(model, optimizer, examples, device)

        # Update running losses
        running_losses['total'] += losses['total_loss']
        running_losses['policy'] += losses['policy_loss']
        running_losses['value'] += losses['value_loss']

        # Print progress
        if (batch_idx + 1) % print_every == 0:
            avg_total = running_losses['total'] / print_every
            avg_policy = running_losses['policy'] / print_every
            avg_value = running_losses['value'] / print_every
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed

            print(f"Batch {batch_idx + 1}/{num_batches}: "
                  f"loss={avg_total:.4f} (policy={avg_policy:.4f}, value={avg_value:.4f}), "
                  f"{batches_per_sec:.1f} batches/s")

            running_losses = {'total': 0.0, 'policy': 0.0, 'value': 0.0}

        # Save checkpoint
        if (batch_idx + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_batch_{batch_idx + 1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'batch': batch_idx + 1,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(output_dir, "model_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch': num_batches,
    }, final_path)

    # Persist optimizer state to a stable path so the next iteration can pick
    # up Adam's running gradient stats even when the new model wasn't promoted.
    if optimizer_state_path is not None:
        torch.save(optimizer.state_dict(), optimizer_state_path)

    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Final model saved to: {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network on self-play data"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to initial model checkpoint"
    )
    parser.add_argument(
        "--replay-buffer",
        required=True,
        help="Path to replay buffer CSV file"
    )
    parser.add_argument(
        "--output-dir",
        default="training_output",
        help="Directory to save checkpoints (default: training_output)"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=1000,
        help="Number of training batches (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)"
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
        help="Device for training (default: cpu)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N batches (default: 100)"
    )

    args = parser.parse_args()

    train(
        model_path=args.model,
        replay_buffer_path=args.replay_buffer,
        output_dir=args.output_dir,
        num_batches=args.batches,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
