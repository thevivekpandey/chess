"""
Evaluate positions from test_data.csv using the trained neural network.
Outputs: fen, stockfish_eval, nn_eval
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# =============================================================================
# NEURAL NETWORK (copied from chess_engine.py)
# =============================================================================

NUM_PLANES = 18

PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert a FEN string to an 18-plane tensor representation."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    parts = fen.split()
    board_str = parts[0]
    side_to_move = parts[1] if len(parts) > 1 else 'w'
    castling = parts[2] if len(parts) > 2 else '-'
    en_passant = parts[3] if len(parts) > 3 else '-'

    row = 0
    col = 0
    for char in board_str:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            if char in PIECE_TO_PLANE:
                plane_idx = PIECE_TO_PLANE[char]
                planes[plane_idx, row, col] = 1.0
            col += 1

    if 'K' in castling:
        planes[12, :, :] = 1.0
    if 'Q' in castling:
        planes[13, :, :] = 1.0
    if 'k' in castling:
        planes[14, :, :] = 1.0
    if 'q' in castling:
        planes[15, :, :] = 1.0

    if side_to_move == 'w':
        planes[16, :, :] = 1.0

    if en_passant != '-':
        ep_col = ord(en_passant[0]) - ord('a')
        ep_row = 8 - int(en_passant[1])
        planes[17, ep_row, ep_col] = 1.0

    return torch.from_numpy(planes)


def denormalize_eval(eval_norm: float, max_pawns: float = 10.0) -> float:
    """Convert normalized evaluation back to pawn units."""
    eval_norm = np.clip(eval_norm, -0.99999, 0.99999)
    return np.arctanh(eval_norm) * max_pawns


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    def __init__(self, initial_channels: int = 512, res_channels: int = 256, num_res_blocks: int = 8):
        super().__init__()
        self.initial_conv = nn.Conv2d(NUM_PLANES, initial_channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(initial_channels)
        self.transition_conv = nn.Conv2d(initial_channels, res_channels, kernel_size=1, bias=False)
        self.transition_bn = nn.BatchNorm2d(res_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(res_channels) for _ in range(num_res_blocks)
        ])
        self.value_conv = nn.Conv2d(res_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = F.relu(self.transition_bn(self.transition_conv(out)))
        for block in self.res_blocks:
            out = block(out)
        out = F.relu(self.value_bn(self.value_conv(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out


# =============================================================================
# EVALUATOR
# =============================================================================

class ChessEvaluator:
    """Evaluator for chess positions using the trained neural network."""

    def __init__(self, model_path: str, device: str = 'auto'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=8)

        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def evaluate(self, fen: str) -> float:
        """Evaluate a single position. Returns evaluation in pawn units."""
        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)
        with torch.no_grad():
            eval_norm = self.model(board_tensor).item()
        return denormalize_eval(eval_norm)

    def evaluate_batch(self, fens: List[str], batch_size: int = 256) -> List[float]:
        """Evaluate multiple positions efficiently in batches."""
        results = []
        for i in range(0, len(fens), batch_size):
            batch_fens = fens[i:i + batch_size]
            tensors = torch.stack([fen_to_tensor(fen) for fen in batch_fens]).to(self.device)
            with torch.no_grad():
                evals_norm = self.model(tensors).cpu().numpy().flatten()
            results.extend([denormalize_eval(e) for e in evals_norm])
        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys

    # Configuration
    MODEL_PATH = "chess_model_epoch063.pt"
    INPUT_FILE = "test_data.csv"
    OUTPUT_FILE = "test_data_with_nn_eval.csv"
    BATCH_SIZE = 256

    # Command line arguments
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        INPUT_FILE = sys.argv[2]
    if len(sys.argv) > 3:
        OUTPUT_FILE = sys.argv[3]

    print("=" * 70)
    print("EVALUATE POSITIONS WITH NEURAL NETWORK")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)

    # Load model
    evaluator = ChessEvaluator(MODEL_PATH)

    # Read input data
    print(f"\nReading positions from {INPUT_FILE}...")
    positions = []
    with open(INPUT_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header if present

        # Check if first row is header or data
        if header and len(header) >= 2:
            try:
                float(header[1])
                # First row is data, not header
                positions.append((header[0], header[1]))
            except ValueError:
                # First row is header, skip it
                pass

        for row in reader:
            if len(row) >= 2:
                positions.append((row[0].strip(), row[1].strip()))

    print(f"Loaded {len(positions):,} positions")

    # Extract FENs for batch evaluation
    fens = [fen for fen, _ in positions]

    # Evaluate all positions
    print(f"\nEvaluating positions (batch size: {BATCH_SIZE})...")
    nn_evals = evaluator.evaluate_batch(fens, batch_size=BATCH_SIZE)

    # Write output
    print(f"\nWriting results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fen', 'stockfish_eval', 'nn_eval'])

        for i, ((fen, sf_eval), nn_eval) in enumerate(zip(positions, nn_evals)):
            writer.writerow([fen, sf_eval, f"{nn_eval:.4f}"])

            # Progress update
            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1:,} / {len(positions):,} positions...")

    print(f"\nDone! Results saved to {OUTPUT_FILE}")

    # Print some statistics
    sf_evals = []
    for _, sf_eval_str in positions:
        try:
            if sf_eval_str.startswith('#'):
                # Mate score
                mate_val = sf_eval_str[1:]
                sf_evals.append(100.0 if float(mate_val) > 0 else -100.0)
            else:
                sf_evals.append(float(sf_eval_str))
        except:
            continue

    if sf_evals:
        sf_evals = np.array(sf_evals)
        nn_evals_arr = np.array(nn_evals)

        # Compute error statistics
        errors = nn_evals_arr - sf_evals
        abs_errors = np.abs(errors)

        print("\n" + "=" * 70)
        print("EVALUATION STATISTICS")
        print("=" * 70)
        print(f"Mean Absolute Error: {np.mean(abs_errors):.4f} pawns")
        print(f"Median Absolute Error: {np.median(abs_errors):.4f} pawns")
        print(f"Std of Errors: {np.std(errors):.4f} pawns")
        print(f"Within 0.5 pawns: {np.mean(abs_errors <= 0.5) * 100:.1f}%")
        print(f"Within 1.0 pawns: {np.mean(abs_errors <= 1.0) * 100:.1f}%")
        print(f"Within 2.0 pawns: {np.mean(abs_errors <= 2.0) * 100:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
