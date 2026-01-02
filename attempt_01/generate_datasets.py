#!/usr/bin/env python3
"""
Generate training_data.csv and test_data.csv from training_data_base.csv
- training_data.csv: 400,000 samples from each of 9 eval ranges (3.6M total)
- test_data.csv: 40,000 samples from each of 9 eval ranges (360K total)
"""

import csv
import random
from collections import defaultdict

TRAIN_SAMPLES_PER_RANGE = 400000
TEST_SAMPLES_PER_RANGE = 40000
TOTAL_SAMPLES_PER_RANGE = TRAIN_SAMPLES_PER_RANGE + TEST_SAMPLES_PER_RANGE

INPUT_FILE = "training_data_base.csv"
TRAIN_OUTPUT = "training_data.csv"
TEST_OUTPUT = "test_data.csv"

def get_eval_range(eval_score):
    """Categorize eval score into one of 9 ranges."""
    e = float(eval_score)
    if e < -10:
        return 0  # < -10
    elif e >= -10 and e < -5:
        return 1  # -10 to -5
    elif e >= -5 and e < -3:
        return 2  # -5 to -3
    elif e >= -3 and e < -1:
        return 3  # -3 to -1
    elif e >= -1 and e <= 1:
        return 4  # -1 to +1
    elif e > 1 and e <= 3:
        return 5  # +1 to +3
    elif e > 3 and e <= 5:
        return 6  # +3 to +5
    elif e > 5 and e <= 10:
        return 7  # +5 to +10
    else:
        return 8  # > +10

RANGE_NAMES = [
    "< -10",
    "-10 to -5",
    "-5 to -3",
    "-3 to -1",
    "-1 to +1",
    "+1 to +3",
    "+3 to +5",
    "+5 to +10",
    "> +10"
]

def main():
    print("Reading and sampling from training_data_base.csv...")

    # Use reservoir sampling to efficiently sample from each range
    # We need TOTAL_SAMPLES_PER_RANGE from each range
    reservoirs = defaultdict(list)
    counts = defaultdict(int)

    with open(INPUT_FILE, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for i, row in enumerate(reader):
            if i % 5000000 == 0:
                print(f"  Processed {i:,} rows...")

            fen, eval_score = row[0], row[1]
            range_idx = get_eval_range(eval_score)
            counts[range_idx] += 1
            n = counts[range_idx]

            # Reservoir sampling
            if n <= TOTAL_SAMPLES_PER_RANGE:
                reservoirs[range_idx].append((fen, eval_score))
            else:
                # Replace with probability TOTAL_SAMPLES_PER_RANGE / n
                j = random.randint(1, n)
                if j <= TOTAL_SAMPLES_PER_RANGE:
                    reservoirs[range_idx][j - 1] = (fen, eval_score)

    print(f"\nTotal rows processed: {sum(counts.values()):,}")
    print("\nSamples collected per range:")
    for idx in range(9):
        print(f"  {RANGE_NAMES[idx]}: {len(reservoirs[idx]):,} (from {counts[idx]:,} total)")

    # Shuffle each reservoir and split into train/test
    print("\nShuffling and splitting into train/test sets...")
    train_data = []
    test_data = []

    for idx in range(9):
        samples = reservoirs[idx]
        random.shuffle(samples)

        train_samples = samples[:TRAIN_SAMPLES_PER_RANGE]
        test_samples = samples[TRAIN_SAMPLES_PER_RANGE:TRAIN_SAMPLES_PER_RANGE + TEST_SAMPLES_PER_RANGE]

        train_data.extend(train_samples)
        test_data.extend(test_samples)

        print(f"  {RANGE_NAMES[idx]}: {len(train_samples):,} train, {len(test_samples):,} test")

    # Shuffle final datasets
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Write training data
    print(f"\nWriting {TRAIN_OUTPUT}...")
    with open(TRAIN_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fen', 'eval'])
        for fen, eval_score in train_data:
            writer.writerow([fen, eval_score])
    print(f"  Written {len(train_data):,} rows")

    # Write test data
    print(f"\nWriting {TEST_OUTPUT}...")
    with open(TEST_OUTPUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fen', 'eval'])
        for fen, eval_score in test_data:
            writer.writerow([fen, eval_score])
    print(f"  Written {len(test_data):,} rows")

    print("\nDone!")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
