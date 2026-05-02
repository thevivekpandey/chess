#!/usr/bin/env python3
"""
Replay buffer for managing self-play training data.

Maintains a circular buffer of the most recent N positions from self-play games.
Supports:
  - Adding new data from CSV files
  - Sampling random batches for training
  - Automatic removal of old data when buffer is full
"""

import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


class ReplayBuffer:
    """
    CSV-based replay buffer with fixed capacity.

    Stores training examples in a CSV file and maintains a circular buffer
    by removing oldest entries when capacity is exceeded.
    """

    def __init__(
        self,
        csv_path: str,
        max_positions: int = 500000,
    ):
        """
        Initialize replay buffer.

        Args:
            csv_path: Path to CSV file storing the buffer
            max_positions: Maximum number of positions to keep (default: 500K)
        """
        self.csv_path = csv_path
        self.max_positions = max_positions
        self.positions = []  # List of dicts: [{fen, policy_moves, policy_visits, value_target, game_id}, ...]

        # Load existing data if file exists
        if os.path.exists(csv_path):
            self._load_from_csv()
            print(f"Loaded {len(self.positions)} positions from {csv_path}")

    def _load_from_csv(self):
        """Load all positions from CSV file."""
        self.positions = []
        with open(self.csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.positions.append(row)

    def _save_to_csv(self):
        """Save all positions to CSV file."""
        with open(self.csv_path, 'w', newline='') as csvfile:
            fieldnames = ['fen', 'policy_moves', 'policy_visits', 'value_target', 'game_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.positions)

    def add_from_csv(self, new_data_csv: str):
        """
        Add new training data from a CSV file.

        Args:
            new_data_csv: Path to CSV file with new self-play data
        """
        new_positions = []
        with open(new_data_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                new_positions.append(row)

        print(f"Adding {len(new_positions)} new positions from {new_data_csv}")

        # Add new positions to buffer
        self.positions.extend(new_positions)

        # Trim to max capacity (keep most recent)
        if len(self.positions) > self.max_positions:
            num_to_remove = len(self.positions) - self.max_positions
            print(f"Buffer full. Removing {num_to_remove} oldest positions")
            self.positions = self.positions[num_to_remove:]

        # Save updated buffer
        self._save_to_csv()
        print(f"Replay buffer now contains {len(self.positions)} positions")

    def sample_batch(self, batch_size: int) -> List[Dict]:
        """
        Sample a random batch of training examples.

        Args:
            batch_size: Number of examples to sample

        Returns:
            List of training examples (dictionaries)
        """
        if len(self.positions) == 0:
            return []

        # Sample with replacement if batch_size > buffer size
        return random.choices(self.positions, k=batch_size)

    def size(self) -> int:
        """Return current number of positions in buffer."""
        return len(self.positions)

    def is_empty(self) -> bool:
        """Return True if buffer is empty."""
        return len(self.positions) == 0

    def parse_example(self, example: Dict) -> Tuple[str, List[str], List[int], float]:
        """
        Parse a training example from CSV format.

        Args:
            example: Dictionary from CSV row

        Returns:
            (fen, move_list, visit_counts, value_target)
        """
        fen = example['fen']

        # Parse comma-separated moves and visits
        policy_moves_str = example['policy_moves']
        policy_visits_str = example['policy_visits']

        if policy_moves_str:
            move_list = policy_moves_str.split(',')
        else:
            move_list = []

        if policy_visits_str:
            visit_counts = [int(v) for v in policy_visits_str.split(',')]
        else:
            visit_counts = []

        value_target = float(example['value_target'])

        return fen, move_list, visit_counts, value_target

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the replay buffer.

        Returns:
            Dictionary with buffer statistics
        """
        if len(self.positions) == 0:
            return {
                'size': 0,
                'capacity': self.max_positions,
                'fill_percentage': 0.0,
                'unique_games': 0,
            }

        # Count unique games
        unique_games = set(pos['game_id'] for pos in self.positions)

        return {
            'size': len(self.positions),
            'capacity': self.max_positions,
            'fill_percentage': 100.0 * len(self.positions) / self.max_positions,
            'unique_games': len(unique_games),
        }

    def print_statistics(self):
        """Print buffer statistics."""
        stats = self.get_statistics()
        print("\n=== Replay Buffer Statistics ===")
        print(f"  Size: {stats['size']:,} / {stats['capacity']:,} positions")
        print(f"  Fill: {stats['fill_percentage']:.1f}%")
        print(f"  Unique games: {stats['unique_games']}")
        print("=" * 33)


def test_replay_buffer():
    """Test the replay buffer functionality."""
    import tempfile

    # Create temporary CSV files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        test_buffer_path = tmp.name

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        test_data_path = tmp.name

    try:
        # Create some test data
        with open(test_data_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['fen', 'policy_moves', 'policy_visits', 'value_target', 'game_id'])
            for i in range(100):
                writer.writerow([
                    f'fen_{i}',
                    'e2e4,d2d4',
                    '100,50',
                    1.0 if i % 2 == 0 else -1.0,
                    f'game_{i // 20}',
                ])

        # Test buffer creation and loading
        print("Creating replay buffer with max 50 positions...")
        buffer = ReplayBuffer(test_buffer_path, max_positions=50)

        print(f"Adding test data...")
        buffer.add_from_csv(test_data_path)

        buffer.print_statistics()

        # Test sampling
        print(f"\nSampling batch of 10...")
        batch = buffer.sample_batch(10)
        print(f"Sampled {len(batch)} examples")

        # Test parsing
        print(f"\nParsing first example...")
        fen, moves, visits, value = buffer.parse_example(batch[0])
        print(f"  FEN: {fen}")
        print(f"  Moves: {moves}")
        print(f"  Visits: {visits}")
        print(f"  Value: {value}")

        print("\nTest passed!")

    finally:
        # Cleanup
        if os.path.exists(test_buffer_path):
            os.remove(test_buffer_path)
        if os.path.exists(test_data_path):
            os.remove(test_data_path)


if __name__ == "__main__":
    test_replay_buffer()
