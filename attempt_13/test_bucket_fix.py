"""
Test script to verify bucket distribution fix.
Loads 400K training positions from S3 and prints bucket statistics.
STANDALONE - does not import from chess_engine.py
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import io
import boto3
import csv

# S3 Configuration
S3_BUCKET = "adhoc-query-data"
S3_PREFIX = "vivek.pandey/engine"

# Constants
NUM_PLANES = 18
MAX_POLICY_MOVES = 5
MAX_LEGAL_MOVES = 128


def normalize_eval(eval_pawns: float, max_pawns: float = 10.0) -> float:
    """
    Normalize pawn evaluation to [-1, 1] range using tanh-like scaling.
    """
    eval_pawns = np.clip(eval_pawns, -80.0, 80.0)
    return np.tanh(eval_pawns / max_pawns)


def denormalize_eval(eval_norm: float, max_pawns: float = 10.0) -> float:
    """
    Convert normalized evaluation back to pawn units.
    """
    eval_norm = np.clip(eval_norm, -0.9999999, 0.9999999)
    return np.arctanh(eval_norm) * max_pawns


def load_data_from_s3(filename: str) -> List[Tuple[str, float, List[str], List[float]]]:
    """
    Load training data from S3.
    """
    s3_client = boto3.client('s3')
    s3_key = f"{S3_PREFIX}/{filename}"

    print(f"Downloading s3://{S3_BUCKET}/{s3_key}...")
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
    content = response['Body'].read().decode('utf-8')

    data = []
    reader = csv.reader(io.StringIO(content))
    header = next(reader)  # Skip header row

    # Detect format based on header length
    has_policy = len(header) >= 12

    for row in reader:
        if len(row) >= 2:
            fen = row[0].strip()
            try:
                eval_pawns = float(row[1].strip())

                # Parse moves and scores if available
                moves = []
                scores = []
                if has_policy and len(row) >= 12:
                    for i in range(5):
                        move_idx = 2 + i * 2
                        score_idx = 3 + i * 2
                        if move_idx < len(row) and score_idx < len(row):
                            move = row[move_idx].strip()
                            score_str = row[score_idx].strip()
                            if move and score_str:
                                try:
                                    score = float(score_str)
                                    moves.append(move)
                                    scores.append(score)
                                except ValueError:
                                    pass

                data.append((fen, eval_pawns, moves, scores))
            except ValueError:
                continue

    return data


def fen_to_tensor_simple(fen: str) -> torch.Tensor:
    """
    Simplified FEN to tensor conversion (just return zeros for testing).
    We only care about eval values for bucket testing.
    """
    return torch.zeros(NUM_PLANES, 8, 8, dtype=torch.float32)


class SimpleChessDataset(Dataset):
    """
    Simplified dataset that only stores eval tensors for bucket testing.
    """

    def __init__(self, data: List[Tuple[str, float, List[str], List[float]]], max_pawns: float = 10.0):
        self.max_pawns = max_pawns

        print(f"Precomputing {len(data):,} eval tensors...")
        self.eval_tensors = []

        for i, item in enumerate(data):
            fen, eval_pawns, moves, scores = item

            # Normalize evaluation
            eval_norm = normalize_eval(eval_pawns, self.max_pawns)
            self.eval_tensors.append(torch.tensor([eval_norm], dtype=torch.float32))

            # Progress report every 100K positions
            if (i + 1) % 100_000 == 0:
                print(f"  Processed {i + 1:,} / {len(data):,} positions...")

        # Stack into single tensor
        self.eval_tensors = torch.stack(self.eval_tensors)
        print(f"Done. Eval tensors shape: {self.eval_tensors.shape}")

    def __len__(self) -> int:
        return len(self.eval_tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.eval_tensors[idx]


def count_eval_buckets(dataloader: DataLoader) -> Tuple[int, int, int, int]:
    """
    Count samples in each evaluation magnitude bucket.
    """
    b1, b2, b3, b4 = 0, 0, 0, 0
    for batch in dataloader:
        evals = batch
        true_pawns = denormalize_eval(evals.numpy())
        abs_true = np.abs(true_pawns)
        b1 += np.sum(abs_true < 1.0)
        b2 += np.sum((abs_true >= 1.0) & (abs_true < 5.0))
        b3 += np.sum((abs_true >= 5.0) & (abs_true < 79.70))
        b4 += np.sum(abs_true >= 79.70)
    return (b1, b2, b3, b4)


def print_bucket_stats(name: str, b1: int, b2: int, b3: int, b4: int):
    """Print bucket statistics."""
    total = b1 + b2 + b3 + b4
    print(f"{name} set buckets (total: {total:,}):")
    print(f"  |eval| < 1:      {b1:,} ({100*b1/total:.1f}%)")
    print(f"  1 <= |eval| < 5: {b2:,} ({100*b2/total:.1f}%)")
    print(f"  5 <= |eval| < 80: {b3:,} ({100*b3/total:.1f}%)")
    print(f"  |eval| >= 80:   {b4:,} ({100*b4/total:.1f}%)")


def main():
    print("="*80)
    print("TESTING BUCKET DISTRIBUTION WITH 400K POSITIONS")
    print("="*80)

    # Configuration
    TRAIN_DATA = 'combined_training_data.csv'
    BATCH_SIZE = 2048

    print(f"\nS3 Location: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"Loading data from: {TRAIN_DATA}")
    print(f"Limiting to: 400,000 positions")

    # Load training data from S3 (limited to 400K)
    print("\nLoading data from S3...")
    train_data = load_data_from_s3(TRAIN_DATA)
    print(f"Total positions available: {len(train_data):,}")

    # Limit to 400K
    train_data = train_data[:400_000]
    print(f"Using: {len(train_data):,} positions for testing")

    # Create dataset
    print("\nCreating dataset...")
    train_dataset = SimpleChessDataset(train_data, max_pawns=10.0)

    # Create dataloader
    print("\nCreating dataloader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Count buckets
    print("\nCounting evaluation buckets...")
    train_b1, train_b2, train_b3, train_b4 = count_eval_buckets(train_loader)

    print("\n" + "="*80)
    print("BUCKET DISTRIBUTION RESULTS")
    print("="*80)
    print_bucket_stats("Training", train_b1, train_b2, train_b3, train_b4)

    print("\n" + "="*80)
    print("EXPECTED vs ACTUAL")
    print("="*80)
    print(f"Expected bucket 4 (based on ~9.6% of data): ~{int(400_000 * 0.096):,}")
    print(f"Actual bucket 4:                             {train_b4:,}")

    if train_b4 == 0:
        print("\n⚠️  WARNING: Bucket 4 is empty! Float32 precision issue confirmed.")
        print("    Recommended fix: Change bucket thresholds to account for float32 precision.")
    else:
        print(f"\n✓  Bucket 4 has {train_b4:,} positions ({100*train_b4/400_000:.2f}%)")

    print("="*80)


if __name__ == '__main__':
    main()
