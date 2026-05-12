"""
Mix Foundation Dataset with New SF-Labeled Data

Combines the original supervised training data (3.6M examples) with
newly generated SF-labeled data to prevent catastrophic forgetting.
"""

import random
from typing import List
import csv


def read_csv_dataset(csv_path: str) -> List[str]:
    """
    Read CSV dataset and return list of rows (as strings).

    Args:
        csv_path: Path to CSV file

    Returns:
        List of CSV rows (excluding header)
    """
    rows = []
    with open(csv_path, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            rows.append(line.strip())
    return rows


def write_csv_dataset(rows: List[str], output_path: str):
    """
    Write CSV dataset from list of rows.

    Args:
        rows: List of CSV rows (strings)
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write('fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5\n')
        for row in rows:
            f.write(row + '\n')


def mix_datasets(
    foundation_path: str,
    new_data_path: str,
    output_train_path: str,
    output_val_path: str,
    foundation_ratio: float = 0.3,
    val_split: float = 0.1,
    seed: int = 42
):
    """
    Mix foundation dataset with new data and split into train/val.

    Args:
        foundation_path: Path to foundation dataset (3.6M examples)
        new_data_path: Path to new SF-labeled data
        output_train_path: Output path for mixed training set
        output_val_path: Output path for validation set
        foundation_ratio: Ratio of foundation data in final mix (0.0-1.0)
        val_split: Fraction of data for validation
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print(f"Mixing datasets...")
    print(f"  Foundation: {foundation_path}")
    print(f"  New data: {new_data_path}")
    print(f"  Foundation ratio: {foundation_ratio:.1%}")
    print(f"  Val split: {val_split:.1%}")
    print()

    # Read datasets
    print("Loading datasets...")
    foundation_rows = read_csv_dataset(foundation_path)
    new_data_rows = read_csv_dataset(new_data_path)

    print(f"  Foundation: {len(foundation_rows)} examples")
    print(f"  New data: {len(new_data_rows)} examples")

    # Calculate how many foundation examples to sample
    # If foundation_ratio = 0.3, then:
    # foundation_count / (foundation_count + new_count) = 0.3
    # foundation_count = 0.3 * (foundation_count + new_count)
    # foundation_count = 0.3 * new_count / (1 - 0.3)
    new_count = len(new_data_rows)
    foundation_count = int(new_count * foundation_ratio / (1 - foundation_ratio))

    # Sample from foundation (with replacement if needed)
    if foundation_count <= len(foundation_rows):
        sampled_foundation = random.sample(foundation_rows, foundation_count)
    else:
        # If we need more than available, sample with replacement
        sampled_foundation = random.choices(foundation_rows, k=foundation_count)

    print(f"  Sampled foundation: {len(sampled_foundation)} examples")

    # Combine datasets
    mixed_data = sampled_foundation + new_data_rows
    random.shuffle(mixed_data)

    print(f"  Mixed total: {len(mixed_data)} examples")

    # Split into train/val
    split_idx = int(len(mixed_data) * (1 - val_split))
    train_rows = mixed_data[:split_idx]
    val_rows = mixed_data[split_idx:]

    print(f"  Train: {len(train_rows)} examples")
    print(f"  Val: {len(val_rows)} examples")

    # Write outputs
    print(f"\nSaving...")
    write_csv_dataset(train_rows, output_train_path)
    write_csv_dataset(val_rows, output_val_path)

    print(f"✓ Saved train to: {output_train_path}")
    print(f"✓ Saved val to: {output_val_path}")

    # Calculate actual ratios
    foundation_in_train = sum(1 for row in train_rows if row in sampled_foundation)
    new_in_train = len(train_rows) - foundation_in_train

    print(f"\nFinal statistics:")
    print(f"  Train - Foundation: {foundation_in_train} ({foundation_in_train/len(train_rows):.1%})")
    print(f"  Train - New: {new_in_train} ({new_in_train/len(train_rows):.1%})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mix foundation dataset with new data")
    parser.add_argument("--foundation", type=str, required=True, help="Path to foundation dataset")
    parser.add_argument("--new-data", type=str, required=True, help="Path to new SF-labeled data")
    parser.add_argument("--output-train", type=str, required=True, help="Output path for training set")
    parser.add_argument("--output-val", type=str, required=True, help="Output path for validation set")
    parser.add_argument("--foundation-ratio", type=float, default=0.3, help="Ratio of foundation data (default: 0.3)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    mix_datasets(
        foundation_path=args.foundation,
        new_data_path=args.new_data,
        output_train_path=args.output_train,
        output_val_path=args.output_val,
        foundation_ratio=args.foundation_ratio,
        val_split=args.val_split,
        seed=args.seed
    )
