"""
Supervised Training from Stockfish Labels

Trains the neural network on positions labeled by Stockfish.
Uses standard cross-entropy loss for policy and MSE for value.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import csv
import chess
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import os

from chess_engine import ChessNet, fen_to_tensor, get_legal_move_indices


class StockfishDataset(Dataset):
    """Dataset for Stockfish-labeled positions from CSV"""

    def __init__(self, csv_path: str):
        """
        Load training data from CSV file.

        CSV format: fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5

        Args:
            csv_path: Path to .csv file with training examples
        """
        self.examples = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.examples.append(row)

        print(f"Loaded {len(self.examples)} training examples from {csv_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        row = self.examples[idx]

        # Get FEN
        fen = row['fen']
        board = chess.Board(fen)

        # Convert FEN to tensor
        board_tensor = fen_to_tensor(fen)

        # Convert eval (in pawns) to value in [-1, 1]
        eval_pawns = float(row['eval'])
        value_target = np.tanh(eval_pawns / 4.0)  # 4 pawns ≈ 76% win prob

        # Convert moves to policy target
        policy_target = self._create_policy_target(board, row)

        return board_tensor, torch.tensor(policy_target, dtype=torch.float32), torch.tensor(value_target, dtype=torch.float32)

    def _create_policy_target(self, board: chess.Board, row: Dict[str, str]) -> np.ndarray:
        """
        Create policy target from CSV row with moves and scores.

        Args:
            board: Chess board
            row: CSV row with move1-5 and score1-5

        Returns:
            Policy target array (4672 floats)
        """
        policy_target = np.zeros(4672, dtype=np.float32)

        # Collect moves with scores
        moves_with_scores = []
        for i in range(1, 6):
            move_key = f'move{i}'
            score_key = f'score{i}'

            if move_key in row and row[move_key]:
                try:
                    move = chess.Move.from_uci(row[move_key])
                    score = int(row[score_key]) if row[score_key] else 0

                    if move in board.legal_moves:
                        moves_with_scores.append((move, score))
                except:
                    continue

        if not moves_with_scores:
            # No valid moves - return uniform over legal moves
            legal_moves = list(board.legal_moves)
            legal_indices = get_legal_move_indices(board, legal_moves)
            for idx in legal_indices:
                policy_target[idx] = 1.0 / len(legal_indices)
            return policy_target

        # Get all legal moves
        legal_moves = list(board.legal_moves)
        legal_indices = get_legal_move_indices(board, legal_moves)

        # Assign scores: labeled moves get their scores, others get worst - 100
        scores = []
        move_to_score = {move: score for move, score in moves_with_scores}
        worst_score = min(score for _, score in moves_with_scores)

        for move in legal_moves:
            if move in move_to_score:
                scores.append(move_to_score[move])
            else:
                scores.append(worst_score - 100)

        # Convert to probabilities with softmax
        scores = np.array(scores, dtype=np.float32)
        scores = scores / 100.0  # Scale to reasonable range
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Fill policy target
        for move, idx, prob in zip(legal_moves, legal_indices, probs):
            policy_target[idx] = prob

        return policy_target


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    policy_weight: float = 1.0,
    value_weight: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss

    Returns:
        Dictionary with loss statistics
    """
    model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    for board_batch, policy_target_batch, value_target_batch in tqdm(dataloader, desc="Training"):
        board_batch = board_batch.to(device)
        policy_target_batch = policy_target_batch.to(device)
        value_target_batch = value_target_batch.to(device)

        # Forward pass
        policy_logits, value_pred = model(board_batch)

        # Policy loss (cross-entropy with soft labels)
        # We need to use KL divergence for soft targets
        policy_log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_target_batch * policy_log_probs).sum(dim=1).mean()

        # Value loss (MSE)
        value_loss = value_criterion(value_pred.squeeze(), value_target_batch)

        # Combined loss
        loss = policy_weight * policy_loss + value_weight * value_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track statistics
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    policy_weight: float = 1.0,
    value_weight: float = 1.0
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Args:
        model: Neural network model
        dataloader: DataLoader for validation data
        device: Device to evaluate on
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss

    Returns:
        Dictionary with loss statistics
    """
    model.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    with torch.no_grad():
        for board_batch, policy_target_batch, value_target_batch in dataloader:
            board_batch = board_batch.to(device)
            policy_target_batch = policy_target_batch.to(device)
            value_target_batch = value_target_batch.to(device)

            # Forward pass
            policy_logits, value_pred = model(board_batch)

            # Policy loss
            policy_log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_target_batch * policy_log_probs).sum(dim=1).mean()

            # Value loss
            value_loss = value_criterion(value_pred.squeeze(), value_target_batch)

            # Combined loss
            loss = policy_weight * policy_loss + value_weight * value_loss

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }


def train_supervised(
    train_data_path: str,
    val_data_path: str,
    model_path: str,
    output_path: str,
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    device: str = "mps",
    save_every: int = 1
):
    """
    Train model with supervised learning from Stockfish labels.

    Args:
        train_data_path: Path to training data (.csv)
        val_data_path: Path to validation data (.csv)
        model_path: Path to initial model checkpoint
        output_path: Directory to save trained models
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss
        device: Device to train on (cpu/cuda/mps)
        save_every: Save checkpoint every N epochs
    """
    print("=" * 80)
    print("SUPERVISED TRAINING FROM STOCKFISH LABELS")
    print("=" * 80)
    print(f"Train data: {train_data_path}")
    print(f"Val data: {val_data_path}")
    print(f"Initial model: {model_path}")
    print(f"Output dir: {output_path}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Policy weight: {policy_weight}")
    print(f"Value weight: {value_weight}")
    print(f"Device: {device}")
    print()

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    train_dataset = StockfishDataset(train_data_path)
    val_dataset = StockfishDataset(val_data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print()

    # Load model
    device_torch = torch.device(device)
    model = ChessNet()

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device_torch, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Starting from scratch (no checkpoint found)")

    model.to(device_torch)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'=' * 80}")

        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, device_torch,
            policy_weight, value_weight
        )

        print(f"\nTrain Loss: {train_stats['loss']:.4f}")
        print(f"  Policy Loss: {train_stats['policy_loss']:.4f}")
        print(f"  Value Loss:  {train_stats['value_loss']:.4f}")

        # Validate
        val_stats = evaluate(
            model, val_loader, device_torch,
            policy_weight, value_weight
        )

        print(f"\nVal Loss: {val_stats['loss']:.4f}")
        print(f"  Policy Loss: {val_stats['policy_loss']:.4f}")
        print(f"  Value Loss:  {val_stats['value_loss']:.4f}")

        # Save checkpoint
        if epoch % save_every == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(output_path, f"model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss']
            }, checkpoint_path)
            print(f"\n✓ Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            best_path = os.path.join(output_path, "model_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_stats['loss'],
                'val_loss': val_stats['loss']
            }, best_path)
            print(f"✓ Saved best model: {best_path} (val_loss: {best_val_loss:.4f})")

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model with supervised learning from Stockfish")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data (.csv)")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation data (.csv)")
    parser.add_argument("--model", type=str, required=True, help="Path to initial model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--policy-weight", type=float, default=1.0, help="Policy loss weight")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Value loss weight")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu/cuda/mps)")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    train_supervised(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_path=args.model,
        output_path=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        device=args.device,
        save_every=args.save_every
    )
