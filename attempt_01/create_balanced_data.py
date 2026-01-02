"""
Create balanced training and test datasets by sampling equally from each eval range.
Uses reservoir sampling for memory efficiency.
"""

import csv
import random

# Configuration
TRAIN_PER_BUCKET = 400_000
TEST_PER_BUCKET = 40_000
TOTAL_PER_BUCKET = TRAIN_PER_BUCKET + TEST_PER_BUCKET

INPUT_FILE = 'training_data_base.csv'
TRAIN_FILE = 'training_data.csv'
TEST_FILE = 'test_data.csv'

SEED = 42
random.seed(SEED)


def get_bucket(eval_val: float) -> str:
    """Categorize evaluation into a bucket."""
    if eval_val < -10:
        return '< -10'
    elif eval_val < -5:
        return '-10 to -5'
    elif eval_val < -3:
        return '-5 to -3'
    elif eval_val < -1:
        return '-3 to -1'
    elif eval_val <= 1:
        return '-1 to +1'
    elif eval_val <= 3:
        return '+1 to +3'
    elif eval_val <= 5:
        return '+3 to +5'
    elif eval_val <= 10:
        return '+5 to +10'
    else:
        return '> +10'


def main():
    # Initialize reservoirs for each bucket
    buckets = [
        '< -10', '-10 to -5', '-5 to -3', '-3 to -1',
        '-1 to +1', '+1 to +3', '+3 to +5', '+5 to +10', '> +10'
    ]

    reservoirs = {b: [] for b in buckets}
    counts = {b: 0 for b in buckets}

    print(f"Reading {INPUT_FILE} and sampling {TOTAL_PER_BUCKET:,} positions per bucket...")
    print(f"(Training: {TRAIN_PER_BUCKET:,}, Test: {TEST_PER_BUCKET:,})")
    print()

    # Single pass with reservoir sampling
    with open(INPUT_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row_num, row in enumerate(reader, 1):
            fen, eval_str = row[0], row[1]
            eval_val = float(eval_str)
            bucket = get_bucket(eval_val)

            counts[bucket] += 1
            n = counts[bucket]

            # Reservoir sampling
            if n <= TOTAL_PER_BUCKET:
                reservoirs[bucket].append(row)
            else:
                # Replace with probability TOTAL_PER_BUCKET / n
                j = random.randint(1, n)
                if j <= TOTAL_PER_BUCKET:
                    reservoirs[bucket][j - 1] = row

            # Progress report
            if row_num % 5_000_000 == 0:
                print(f"  Processed {row_num:,} rows...")

    print(f"\nTotal rows processed: {row_num:,}")
    print()

    # Report sampling results
    print("Samples collected per bucket:")
    print(f"{'Bucket':<15} {'Available':>12} {'Sampled':>10}")
    print("-" * 40)
    for bucket in buckets:
        available = counts[bucket]
        sampled = len(reservoirs[bucket])
        print(f"{bucket:<15} {available:>12,} {sampled:>10,}")
    print()

    # Shuffle and split each reservoir into train/test
    train_data = []
    test_data = []

    for bucket in buckets:
        samples = reservoirs[bucket]
        random.shuffle(samples)

        # Take up to TRAIN_PER_BUCKET for training, up to TEST_PER_BUCKET for test
        train_samples = samples[:TRAIN_PER_BUCKET]
        test_samples = samples[TRAIN_PER_BUCKET:TRAIN_PER_BUCKET + TEST_PER_BUCKET]

        train_data.extend(train_samples)
        test_data.extend(test_samples)

        print(f"{bucket}: train={len(train_samples):,}, test={len(test_samples):,}")

    # Shuffle final datasets
    random.shuffle(train_data)
    random.shuffle(test_data)

    # Write training data
    print(f"\nWriting {len(train_data):,} rows to {TRAIN_FILE}...")
    with open(TRAIN_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_data)

    # Write test data
    print(f"Writing {len(test_data):,} rows to {TEST_FILE}...")
    with open(TEST_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test_data)

    print("\nDone!")
    print(f"  {TRAIN_FILE}: {len(train_data):,} positions")
    print(f"  {TEST_FILE}: {len(test_data):,} positions")


if __name__ == '__main__':
    main()
