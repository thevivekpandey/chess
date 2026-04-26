"""
Neural Network Chess Engine with Value and Policy Heads
Trained on Lichess evaluations from Stockfish

Board Encoding: 18 planes (8x8 each)
- Planes 0-5: White pieces (P, N, B, R, Q, K)
- Planes 6-11: Black pieces (p, n, b, r, q, k)
- Plane 12: White kingside castling right
- Plane 13: White queenside castling right
- Plane 14: Black kingside castling right
- Plane 15: Black queenside castling right
- Plane 16: Side to move (1 = white, 0 = black)
- Plane 17: En passant square

Neural Network Architecture:
- Input: 18 planes of 8x8
- Initial conv (18 -> 512 channels) + BatchNorm + ReLU
- Transition conv (512 -> 256 channels) + BatchNorm + ReLU
- 16 Residual blocks (256 channels)
- Value head: Conv 256->32, FC 2048->256->1, tanh output
- Policy head: Conv 256->128 + BatchNorm + ReLU, Conv 128->73

Outputs:
- Value: Evaluation between -1 (black winning) and +1 (white winning)
- Policy: 73-plane move representation (8x8 per plane) covering all possible move types

Training Data Format (CSV):
fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5
- eval: position evaluation in pawns
- moveN: UCI format moves (e.g., "e2e4")
- scoreN: score change in centipawns after the move
"""

import logging
# Suppress debug messages from Databricks thread monitor
logging.getLogger('ThreadMonitor').setLevel(logging.WARNING)

import chess
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
import time


# =============================================================================
# BOARD ENCODING
# =============================================================================

# Piece to plane index mapping
PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
}

NUM_PLANES = 18


def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Convert a FEN string to an 18-plane tensor representation.

    Args:
        fen: FEN string representing board position

    Returns:
        Tensor of shape (18, 8, 8)
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    parts = fen.split()
    board_str = parts[0]
    side_to_move = parts[1] if len(parts) > 1 else 'w'
    castling = parts[2] if len(parts) > 2 else '-'
    en_passant = parts[3] if len(parts) > 3 else '-'

    # Parse board position (planes 0-11)
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

    # Parse castling rights (planes 12-15)
    if 'K' in castling:
        planes[12, :, :] = 1.0  # White kingside
    if 'Q' in castling:
        planes[13, :, :] = 1.0  # White queenside
    if 'k' in castling:
        planes[14, :, :] = 1.0  # Black kingside
    if 'q' in castling:
        planes[15, :, :] = 1.0  # Black queenside

    # Side to move (plane 16)
    if side_to_move == 'w':
        planes[16, :, :] = 1.0

    # En passant square (plane 17)
    if en_passant != '-':
        ep_col = ord(en_passant[0]) - ord('a')
        ep_row = 8 - int(en_passant[1])
        planes[17, ep_row, ep_col] = 1.0

    return torch.from_numpy(planes)


def normalize_eval(eval_pawns: float, max_pawns: float = 10.0) -> float:
    """
    Normalize pawn evaluation to [-1, 1] range using tanh-like scaling.

    Args:
        eval_pawns: Evaluation in pawn units (clipped to ±80 pawns)
        max_pawns: Scaling factor (evaluation at this value maps to ~0.76)

    Returns:
        Normalized evaluation in [-1, 1]
    """
    # Clip evaluations to ±80 pawns (already completely winning/losing)
    # This avoids numerical issues with extreme values near ±1.0
    eval_pawns = np.clip(eval_pawns, -80.0, 80.0)
    return np.tanh(eval_pawns / max_pawns)


def denormalize_eval(eval_norm: float, max_pawns: float = 10.0) -> float:
    """
    Convert normalized evaluation back to pawn units.

    Args:
        eval_norm: Normalized evaluation in [-1, 1]
        max_pawns: Scaling factor used in normalization

    Returns:
        Evaluation in pawn units (clipped to ±80 range)
    """
    # Clamp to safe range for arctanh (avoid domain errors at exact ±1.0)
    # With input evals clipped to ±80, tanh(8) ≈ 0.99999977, so 0.9999999 is safe
    eval_norm = np.clip(eval_norm, -0.9999999, 0.9999999)
    return np.arctanh(eval_norm) * max_pawns


# =============================================================================
# POLICY ENCODING (73-plane move representation)
# =============================================================================
#
# Move types (73 total):
# - Planes 0-55: Queen-like moves (8 directions × 7 distances)
#   - N1-N7:  planes 0-6   (North)
#   - NE1-NE7: planes 7-13  (NorthEast)
#   - E1-E7:  planes 14-20 (East)
#   - SE1-SE7: planes 21-27 (SouthEast)
#   - S1-S7:  planes 28-34 (South)
#   - SW1-SW7: planes 35-41 (SouthWest)
#   - W1-W7:  planes 42-48 (West)
#   - NW1-NW7: planes 49-55 (NorthWest)
# - Planes 56-63: Knight moves (8 types)
# - Planes 64-72: Underpromotions (3 pieces × 3 directions)
#   - Knight: planes 64-66 (left, straight, right)
#   - Bishop: planes 67-69 (left, straight, right)
#   - Rook:   planes 70-72 (left, straight, right)
#
# Queen promotions use the normal queen-like move planes.
# =============================================================================

NUM_POLICY_PLANES = 73

# Direction vectors: (row_delta, col_delta)
# Row increases going down (rank 8 = row 0, rank 1 = row 7)
DIRECTIONS = [
    (-1, 0),   # N  (North - towards rank 8)
    (-1, 1),   # NE
    (0, 1),    # E  (East - towards h-file)
    (1, 1),    # SE
    (1, 0),    # S  (South - towards rank 1)
    (1, -1),   # SW
    (0, -1),   # W  (West - towards a-file)
    (-1, -1),  # NW
]

# Knight move deltas
KNIGHT_MOVES = [
    (-2, -1), (-2, 1),  # NNW, NNE
    (-1, -2), (-1, 2),  # WNW, ENE
    (1, -2),  (1, 2),   # WSW, ESE
    (2, -1),  (2, 1),   # SSW, SSE
]

# Underpromotion pieces
UNDERPROMO_PIECES = ['n', 'b', 'r']  # Knight, Bishop, Rook

# Underpromotion directions (col delta): -1 = left capture, 0 = straight, 1 = right capture
UNDERPROMO_DIRS = [-1, 0, 1]


def parse_move(move_str: str) -> Optional[Tuple[int, int, int, int, Optional[str]]]:
    """
    Parse a UCI move string into coordinates and optional promotion piece.

    Args:
        move_str: UCI format move (e.g., "e2e4", "g1f3", "e7e8q")

    Returns:
        Tuple of (src_row, src_col, dst_row, dst_col, promo_piece) or None if invalid
        promo_piece is 'q', 'r', 'b', 'n' or None
    """
    if not move_str or len(move_str) < 4:
        return None

    try:
        src_col = ord(move_str[0]) - ord('a')
        src_row = 8 - int(move_str[1])
        dst_col = ord(move_str[2]) - ord('a')
        dst_row = 8 - int(move_str[3])

        # Validate coordinates
        if not (0 <= src_col < 8 and 0 <= src_row < 8 and
                0 <= dst_col < 8 and 0 <= dst_row < 8):
            return None

        # Check for promotion piece
        promo_piece = None
        if len(move_str) >= 5:
            promo_piece = move_str[4].lower()

        return (src_row, src_col, dst_row, dst_col, promo_piece)
    except (ValueError, IndexError):
        return None


def move_to_policy_index(move_str: str) -> Optional[Tuple[int, int, int]]:
    """
    Convert a UCI move string to policy tensor indices.

    Args:
        move_str: UCI format move (e.g., "e2e4", "g1f3", "e7e8q")

    Returns:
        Tuple of (src_row, src_col, plane_index) or None if invalid
    """
    parsed = parse_move(move_str)
    if parsed is None:
        return None

    src_row, src_col, dst_row, dst_col, promo_piece = parsed
    row_delta = dst_row - src_row
    col_delta = dst_col - src_col

    # Check for underpromotion (non-queen promotion)
    if promo_piece in UNDERPROMO_PIECES:
        promo_idx = UNDERPROMO_PIECES.index(promo_piece)
        # Determine direction: left capture (-1), straight (0), right capture (1)
        if col_delta in UNDERPROMO_DIRS:
            dir_idx = UNDERPROMO_DIRS.index(col_delta)
            plane = 64 + promo_idx * 3 + dir_idx
            return (src_row, src_col, plane)
        return None

    # Check for knight move
    if (row_delta, col_delta) in KNIGHT_MOVES:
        knight_idx = KNIGHT_MOVES.index((row_delta, col_delta))
        plane = 56 + knight_idx
        return (src_row, src_col, plane)

    # Check for queen-like move (includes queen promotion)
    if row_delta == 0 and col_delta == 0:
        return None  # No movement

    # Determine direction and distance
    # Normalize to unit direction
    if row_delta != 0:
        row_sign = row_delta // abs(row_delta)
    else:
        row_sign = 0

    if col_delta != 0:
        col_sign = col_delta // abs(col_delta)
    else:
        col_sign = 0

    # Check if it's a valid queen-like move (straight line)
    if row_delta != 0 and col_delta != 0 and abs(row_delta) != abs(col_delta):
        return None  # Not a diagonal

    distance = max(abs(row_delta), abs(col_delta))
    if distance < 1 or distance > 7:
        return None

    # Find direction index
    direction = (row_sign, col_sign)
    if direction not in DIRECTIONS:
        return None

    dir_idx = DIRECTIONS.index(direction)
    plane = dir_idx * 7 + (distance - 1)  # distance 1 -> offset 0, etc.

    return (src_row, src_col, plane)


def policy_index_to_move(src_row: int, src_col: int, plane: int) -> Optional[str]:
    """
    Convert policy tensor indices back to a UCI move string.

    Args:
        src_row: Source row (0-7, where 0 is rank 8)
        src_col: Source column (0-7, where 0 is a-file)
        plane: Plane index (0-72)

    Returns:
        UCI move string or None if invalid
    """
    if not (0 <= src_row < 8 and 0 <= src_col < 8 and 0 <= plane < 73):
        return None

    src_square = chr(ord('a') + src_col) + str(8 - src_row)

    # Underpromotion (planes 64-72)
    if plane >= 64:
        promo_idx = (plane - 64) // 3
        dir_idx = (plane - 64) % 3
        promo_piece = UNDERPROMO_PIECES[promo_idx]
        col_delta = UNDERPROMO_DIRS[dir_idx]

        # Determine row direction based on source row (pawns move towards promotion rank)
        # White pawns on row 1 (rank 7) promote to row 0 (rank 8): row_delta = -1
        # Black pawns on row 6 (rank 2) promote to row 7 (rank 1): row_delta = +1
        if src_row == 1:  # White pawn about to promote
            row_delta = -1
        elif src_row == 6:  # Black pawn about to promote
            row_delta = 1
        else:
            return None  # Invalid promotion position

        dst_row = src_row + row_delta
        dst_col = src_col + col_delta

        if not (0 <= dst_row < 8 and 0 <= dst_col < 8):
            return None

        dst_square = chr(ord('a') + dst_col) + str(8 - dst_row)
        return src_square + dst_square + promo_piece

    # Knight move (planes 56-63)
    if plane >= 56:
        knight_idx = plane - 56
        row_delta, col_delta = KNIGHT_MOVES[knight_idx]
        dst_row = src_row + row_delta
        dst_col = src_col + col_delta

        if not (0 <= dst_row < 8 and 0 <= dst_col < 8):
            return None

        dst_square = chr(ord('a') + dst_col) + str(8 - dst_row)
        return src_square + dst_square

    # Queen-like move (planes 0-55)
    dir_idx = plane // 7
    distance = (plane % 7) + 1

    row_delta, col_delta = DIRECTIONS[dir_idx]
    dst_row = src_row + row_delta * distance
    dst_col = src_col + col_delta * distance

    if not (0 <= dst_row < 8 and 0 <= dst_col < 8):
        return None

    dst_square = chr(ord('a') + dst_col) + str(8 - dst_row)

    # Check if this is a pawn promotion (pawn reaching last rank)
    # We add 'q' for queen promotion
    if (src_row == 1 and dst_row == 0) or (src_row == 6 and dst_row == 7):
        # This could be a pawn promotion - add queen
        return src_square + dst_square + 'q'

    return src_square + dst_square


MAX_POLICY_MOVES = 5  # Maximum number of moves to store per position


def moves_to_policy_sparse(
    moves: List[str],
    scores: List[float],
    temperature: float = 1.0
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert moves and scores to sparse policy representation.

    Args:
        moves: List of UCI move strings
        scores: List of scores in centipawns
        temperature: Softmax temperature for score conversion

    Returns:
        Tuple of (indices, weights) arrays, each of length MAX_POLICY_MOVES,
        or None if no valid moves. Unused slots have index -1 and weight 0.
    """
    # Filter valid moves (non-empty, not -9999 blunders)
    valid_flat_indices = []
    valid_scores = []

    for move, score in zip(moves, scores):
        if move and score > -9000:  # Filter out -9999 blunders
            idx = move_to_policy_index(move)
            if idx is not None:
                src_row, src_col, plane = idx
                # Flatten to single index: plane * 64 + row * 8 + col
                flat_idx = plane * 64 + src_row * 8 + src_col
                valid_flat_indices.append(flat_idx)
                valid_scores.append(score)

    if not valid_flat_indices:
        return None

    # Convert scores to probabilities using softmax
    scores_array = np.array(valid_scores, dtype=np.float32)
    # Scale by temperature (scores are in centipawns, divide by 100 to get pawns first)
    scores_scaled = (scores_array / 100.0) / temperature
    # Softmax
    exp_scores = np.exp(scores_scaled - np.max(scores_scaled))
    probs = exp_scores / np.sum(exp_scores)

    # Create fixed-size arrays with padding
    indices = np.full(MAX_POLICY_MOVES, -1, dtype=np.int32)
    weights = np.zeros(MAX_POLICY_MOVES, dtype=np.float32)

    n_moves = min(len(valid_flat_indices), MAX_POLICY_MOVES)
    indices[:n_moves] = valid_flat_indices[:n_moves]
    weights[:n_moves] = probs[:n_moves]

    # Renormalize weights if we truncated
    if n_moves < len(valid_flat_indices):
        weights[:n_moves] /= weights[:n_moves].sum()

    return indices, weights


def get_top_moves_from_policy(
    policy_logits: np.ndarray,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Extract top-k moves from policy logits.

    Args:
        policy_logits: Array of shape (73, 8, 8)
        top_k: Number of top moves to return

    Returns:
        List of (move_uci, probability) tuples, sorted by probability descending
    """
    # Apply softmax to get probabilities
    flat_logits = policy_logits.flatten()
    exp_logits = np.exp(flat_logits - np.max(flat_logits))
    probs = exp_logits / np.sum(exp_logits)

    # Get top-k indices
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in top_indices:
        plane = idx // 64
        spatial_idx = idx % 64
        src_row = spatial_idx // 8
        src_col = spatial_idx % 8

        move_uci = policy_index_to_move(src_row, src_col, plane)
        if move_uci:
            results.append((move_uci, float(probs[idx])))

    return results


MAX_LEGAL_MOVES = 128  # Max legal moves to store (theoretical max is 218, but 128 covers 99.9% of positions)


def get_legal_move_indices(fen: str) -> np.ndarray:
    """
    Get indices of legal moves for a position (sparse representation).

    Args:
        fen: FEN string of the position

    Returns:
        Array of shape (MAX_LEGAL_MOVES,) containing legal move indices,
        padded with -1 for unused slots. Uses int16 to save memory.
    """
    indices = np.full(MAX_LEGAL_MOVES, -1, dtype=np.int16)

    try:
        board = chess.Board(fen)
    except ValueError:
        return indices

    i = 0
    for move in board.legal_moves:
        if i >= MAX_LEGAL_MOVES:
            break
        move_uci = move.uci()
        idx = move_to_policy_index(move_uci)
        if idx is not None:
            src_row, src_col, plane = idx
            flat_idx = plane * 64 + src_row * 8 + src_col
            indices[i] = flat_idx
            i += 1

    return indices


def legal_indices_to_mask(legal_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert sparse legal move indices to dense mask (done on GPU during training).

    Args:
        legal_indices: Tensor of shape (batch, MAX_LEGAL_MOVES) with legal move indices, -1 for padding
        device: Device to create mask on

    Returns:
        Dense mask of shape (batch, 4672) where 1=legal, 0=illegal
    """
    batch_size = legal_indices.size(0)
    mask = torch.zeros(batch_size, NUM_POLICY_PLANES * 8 * 8, device=device)

    # Create batch indices for scatter
    valid_mask = legal_indices >= 0  # (batch, MAX_LEGAL_MOVES)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(legal_indices)

    # Only scatter where indices are valid
    valid_batch_idx = batch_idx[valid_mask]
    valid_move_idx = legal_indices[valid_mask].long()

    mask[valid_batch_idx, valid_move_idx] = 1.0

    return mask


# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.
    """

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
    """
    Neural network for chess position evaluation and policy prediction.

    Architecture:
    - Input: 18 planes of 8x8
    - Initial conv (18 -> 512 channels) + BatchNorm + ReLU
    - Transition conv (512 -> 256 channels) + BatchNorm + ReLU
    - 16 Residual blocks (256 channels)
    - Value head: Conv 256->32, FC 2048->256->1, Tanh output
    - Policy head: Conv 256->73, outputs 73 planes for move encoding
    """

    def __init__(
        self,
        initial_channels: int = 512,
        res_channels: int = 256,
        num_res_blocks: int = 16,
        policy_channels: int = 128
    ):
        super().__init__()

        # Initial convolution (18 -> 512)
        self.initial_conv = nn.Conv2d(NUM_PLANES, initial_channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(initial_channels)

        # Transition convolution (512 -> 256)
        self.transition_conv = nn.Conv2d(initial_channels, res_channels, kernel_size=1, bias=False)
        self.transition_bn = nn.BatchNorm2d(res_channels)

        # Residual tower (256 channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(res_channels) for _ in range(num_res_blocks)
        ])

        # Value head
        self.value_conv = nn.Conv2d(res_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

        # Policy head - outputs 73 planes (one per move type)
        self.policy_conv1 = nn.Conv2d(res_channels, policy_channels, kernel_size=3, padding=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(policy_channels)
        self.policy_conv2 = nn.Conv2d(policy_channels, NUM_POLICY_PLANES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 18, 8, 8)

        Returns:
            Tuple of (value, policy_logits):
            - value: Evaluation tensor of shape (batch, 1) in range [-1, 1]
            - policy_logits: Policy logits of shape (batch, 73, 8, 8)
        """
        # Initial convolution (18 -> 512)
        out = F.relu(self.initial_bn(self.initial_conv(x)))

        # Transition (512 -> 256)
        out = F.relu(self.transition_bn(self.transition_conv(out)))

        # Residual tower (256 channels)
        for block in self.res_blocks:
            out = block(out)

        # Value head
        value_out = F.relu(self.value_bn(self.value_conv(out)))
        value_out = value_out.view(value_out.size(0), -1)  # Flatten
        value_out = F.relu(self.fc1(value_out))
        value = torch.tanh(self.fc2(value_out))

        # Policy head - outputs (batch, 73, 8, 8)
        policy_out = F.relu(self.policy_bn1(self.policy_conv1(out)))
        policy_logits = self.policy_conv2(policy_out)  # (batch, 73, 8, 8)

        return value, policy_logits


# =============================================================================
# DATASET
# =============================================================================

class ChessDataset(Dataset):
    """
    Dataset for chess position evaluations and policy targets.
    Precomputes all board tensors for faster training.
    Uses sparse policy representation to save memory.

    Expected data format: List of tuples with:
    - fen: FEN string
    - eval_pawns: evaluation in pawn units
    - moves: list of UCI move strings (up to 5)
    - scores: list of scores in centipawns (up to 5)
    """

    def __init__(
        self,
        data: List[Tuple[str, float, List[str], List[float]]],
        max_pawns: float = 10.0,
        policy_temperature: float = 1.0
    ):
        """
        Args:
            data: List of (fen, eval_pawns, moves, scores) tuples
            max_pawns: Scaling factor for evaluation normalization
            policy_temperature: Temperature for softmax over move scores
        """
        self.max_pawns = max_pawns

        # Precompute all board tensors and normalized evaluations
        print(f"Precomputing {len(data):,} board tensors...")
        print("  (Also computing legal move indices for policy masking)")
        self.board_tensors = []
        self.eval_tensors = []
        # Sparse policy storage: indices and weights
        self.policy_indices = []  # Shape (N, MAX_POLICY_MOVES), -1 for invalid
        self.policy_weights = []  # Shape (N, MAX_POLICY_MOVES)
        self.has_policy = []  # Track which samples have valid policy targets
        # Legal move indices for policy training (sparse representation)
        self.legal_move_indices = []  # Shape (N, MAX_LEGAL_MOVES), int16

        for i, item in enumerate(data):
            # Handle both old format (fen, eval) and new format (fen, eval, moves, scores)
            if len(item) == 2:
                fen, eval_pawns = item
                moves, scores = [], []
            else:
                fen, eval_pawns, moves, scores = item

            # Convert FEN to tensor
            board_tensor = fen_to_tensor(fen)
            self.board_tensors.append(board_tensor)

            # Normalize evaluation
            eval_norm = normalize_eval(eval_pawns, self.max_pawns)
            self.eval_tensors.append(torch.tensor([eval_norm], dtype=torch.float32))

            # Compute sparse policy targets
            if moves:
                sparse_result = moves_to_policy_sparse(moves, scores, policy_temperature)
                if sparse_result is not None:
                    indices, weights = sparse_result
                    self.policy_indices.append(indices)
                    self.policy_weights.append(weights)
                    self.has_policy.append(True)
                else:
                    # No valid moves - use placeholder
                    self.policy_indices.append(np.full(MAX_POLICY_MOVES, -1, dtype=np.int32))
                    self.policy_weights.append(np.zeros(MAX_POLICY_MOVES, dtype=np.float32))
                    self.has_policy.append(False)
            else:
                # No moves provided - use placeholder
                self.policy_indices.append(np.full(MAX_POLICY_MOVES, -1, dtype=np.int32))
                self.policy_weights.append(np.zeros(MAX_POLICY_MOVES, dtype=np.float32))
                self.has_policy.append(False)

            # Always compute legal move indices (used for policy masking)
            legal_indices = get_legal_move_indices(fen)
            self.legal_move_indices.append(legal_indices)

            # Progress report every 500K positions
            if (i + 1) % 500_000 == 0:
                print(f"  Processed {i + 1:,} / {len(data):,} positions...")

        # Stack into single tensors for efficiency
        self.board_tensors = torch.stack(self.board_tensors)
        self.eval_tensors = torch.stack(self.eval_tensors)
        self.policy_indices = torch.from_numpy(np.stack(self.policy_indices))  # (N, MAX_POLICY_MOVES) int32
        self.policy_weights = torch.from_numpy(np.stack(self.policy_weights))  # (N, MAX_POLICY_MOVES) float32
        self.has_policy = torch.tensor(self.has_policy, dtype=torch.bool)

        self.legal_move_indices = torch.from_numpy(np.stack(self.legal_move_indices))  # (N, MAX_LEGAL_MOVES) int16

        policy_count = self.has_policy.sum().item()
        print(f"Done. Board tensors shape: {self.board_tensors.shape}")
        print(f"Policy indices shape: {self.policy_indices.shape} (sparse, ~{self.policy_indices.nbytes / 1e6:.1f} MB)")
        print(f"Legal move indices shape: {self.legal_move_indices.shape} (sparse, ~{self.legal_move_indices.nbytes / 1e6:.1f} MB)")
        print(f"Positions with policy targets: {policy_count:,} / {len(data):,} ({100*policy_count/len(data):.1f}%)")

    def __len__(self) -> int:
        return len(self.board_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.board_tensors[idx],
            self.eval_tensors[idx],
            self.policy_indices[idx],
            self.policy_weights[idx],
            self.has_policy[idx],
            self.legal_move_indices[idx]
        )


def load_data_from_csv(filename: str) -> List[Tuple[str, float, List[str], List[float]]]:
    """
    Load training data from a local CSV file.

    Expected format: CSV with header
    "fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5"
    - eval is in pawn units
    - scores are in centipawns
    Mate scores are represented as +100 or -100.

    Args:
        filename: Path to the CSV file

    Returns:
        List of (fen, eval_pawns, moves, scores) tuples
    """
    import csv

    data = []
    print(f"Loading {filename}...")
    with open(filename, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_stratified_accuracy(
    pred_pawns: np.ndarray,
    true_pawns: np.ndarray
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Compute stratified accuracy metrics based on evaluation magnitude.

    Args:
        pred_pawns: Predicted evaluations in pawn units
        true_pawns: True evaluations in pawn units

    Returns:
        Tuple of (w05_count, b1_samples, w1_count, b2_samples,
                  w3_count, b3_samples, w90_count, b4_samples)
    """
    abs_true = np.abs(true_pawns)
    errors = np.abs(pred_pawns - true_pawns)

    # Bucket 1: |eval| < 1, tolerance 0.5
    mask_b1 = abs_true < 1.0
    w05_count = np.sum(errors[mask_b1] <= 0.5) if np.any(mask_b1) else 0
    b1_samples = np.sum(mask_b1)

    # Bucket 2: 1 <= |eval| < 5, tolerance 1
    mask_b2 = (abs_true >= 1.0) & (abs_true < 5.0)
    w1_count = np.sum(errors[mask_b2] <= 1.0) if np.any(mask_b2) else 0
    b2_samples = np.sum(mask_b2)

    # Bucket 3: 5 <= |eval| < 79.70, tolerance 3
    mask_b3 = (abs_true >= 5.0) & (abs_true < 79.70)
    w3_count = np.sum(errors[mask_b3] <= 3.0) if np.any(mask_b3) else 0
    b3_samples = np.sum(mask_b3)

    # Bucket 4: |eval| >= 79.70, tolerance 100 (clipped values at ±80)
    # Note: Using 79.70 to account for float32 precision loss in PyTorch tensors
    mask_b4 = abs_true >= 79.70
    w90_count = np.sum(errors[mask_b4] <= 100.0) if np.any(mask_b4) else 0
    b4_samples = np.sum(mask_b4)

    return (w05_count, b1_samples, w1_count, b2_samples,
            w3_count, b3_samples, w90_count, b4_samples)


def count_eval_buckets(dataloader: DataLoader) -> Tuple[int, int, int, int]:
    """
    Count samples in each evaluation magnitude bucket.

    Args:
        dataloader: DataLoader to count buckets from

    Returns:
        Tuple of (bucket1_count, bucket2_count, bucket3_count, bucket4_count)
    """
    b1, b2, b3, b4 = 0, 0, 0, 0
    for batch in dataloader:
        _, evals, _, _, _, _ = batch
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


def ensure_parent_dir(path: str):
    """Create the parent directory for a file path if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def compute_topk_accuracy(
    policy_logits: torch.Tensor,
    policy_indices: torch.Tensor,
    policy_weights: torch.Tensor,
    has_policy: torch.Tensor,
    legal_move_masks: torch.Tensor,
    device: torch.device
) -> Tuple[int, int, int, int]:
    """
    Compute top-k policy accuracy for k=3, 5, 7.

    Args:
        policy_logits: Policy predictions (batch, 73, 8, 8)
        policy_indices: Sparse target indices (batch, MAX_POLICY_MOVES)
        policy_weights: Target weights (batch, MAX_POLICY_MOVES)
        has_policy: Boolean mask for samples with policy (batch,)
        legal_move_masks: Legal move masks (batch, 4672)
        device: Torch device

    Returns:
        Tuple of (top3_count, top5_count, top7_count, total_samples)
    """
    if not has_policy.any():
        return (0, 0, 0, 0)

    batch_size = policy_logits.size(0)

    # Get best target move from sparse representation (highest weight)
    masked_weights = policy_weights.clone().float()
    masked_weights[policy_indices < 0] = float('-inf')
    best_slot = masked_weights.argmax(dim=1)  # (batch,)
    batch_arange = torch.arange(batch_size, device=device)
    target_best_flat = policy_indices[batch_arange, best_slot].long()  # (batch,)

    # Use legal-move-masked logits for top-k predictions
    logits_flat = policy_logits.view(batch_size, -1)
    masked_logits = logits_flat.masked_fill(legal_move_masks == 0, float('-inf'))
    _, top3_pred = masked_logits.topk(k=3, dim=1)
    _, top5_pred = masked_logits.topk(k=5, dim=1)
    _, top7_pred = masked_logits.topk(k=7, dim=1)

    # Only count samples that have valid policy targets
    valid_for_topk = has_policy & (target_best_flat >= 0)
    target_best_flat_exp = target_best_flat.unsqueeze(1)

    top3_count = ((top3_pred == target_best_flat_exp).any(dim=1) & valid_for_topk).sum().item()
    top5_count = ((top5_pred == target_best_flat_exp).any(dim=1) & valid_for_topk).sum().item()
    top7_count = ((top7_pred == target_best_flat_exp).any(dim=1) & valid_for_topk).sum().item()
    total_samples = valid_for_topk.sum().item()

    return (top3_count, top5_count, top7_count, total_samples)


# =============================================================================
# TRAINING
# =============================================================================

class Trainer:
    """
    Trainer for the chess neural network with value and policy heads.
    """

    def __init__(
        self,
        model: ChessNet,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        policy_weight: float = 0.1
    ):
        """
        Args:
            model: ChessNet model to train
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            policy_weight: Weight for policy loss relative to value loss
        """
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.policy_weight = policy_weight
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.value_criterion = nn.SmoothL1Loss(beta=0.2)

    def _compute_policy_loss(
        self,
        policy_logits: torch.Tensor,
        policy_indices: torch.Tensor,
        policy_weights: torch.Tensor,
        has_policy: torch.Tensor,
        legal_move_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for policy head using sparse targets.
        Uses legal move masking to ensure probability is only distributed among legal moves.

        Args:
            policy_logits: Predicted policy logits (batch, 73, 8, 8)
            policy_indices: Sparse target indices (batch, MAX_POLICY_MOVES), -1 for invalid
            policy_weights: Target weights (batch, MAX_POLICY_MOVES)
            has_policy: Boolean mask for samples with valid policy (batch,)
            legal_move_masks: Binary masks for legal moves (batch, 4672), 1=legal, 0=illegal

        Returns:
            Policy loss (scalar)
        """
        batch_size = policy_logits.size(0)

        # Check which positions have at least one legal move in the mask
        has_legal_moves = legal_move_masks.sum(dim=1) > 0  # (batch,)

        # Combine with has_policy: need both policy targets AND legal moves
        valid_mask = has_policy & has_legal_moves

        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device)

        # Flatten logits to (batch, 73*8*8) = (batch, 4672)
        policy_logits_flat = policy_logits.view(batch_size, -1)

        # Apply legal move masking: set illegal move logits to -inf before softmax
        # This ensures probability mass is only distributed among legal moves
        masked_logits = policy_logits_flat.masked_fill(legal_move_masks == 0, float('-inf'))

        # Apply log_softmax to masked predictions
        log_probs = F.log_softmax(masked_logits, dim=1)

        valid_log_probs = log_probs[valid_mask]  # (n_valid, 4672)
        valid_indices = policy_indices[valid_mask].long()  # (n_valid, MAX_POLICY_MOVES)
        valid_weights = policy_weights[valid_mask].float()  # (n_valid, MAX_POLICY_MOVES)
        valid_legal_masks = legal_move_masks[valid_mask]  # (n_valid, 4672)

        # Compute weighted cross-entropy for each valid sample
        # For each sample, sum over the sparse targets: -sum(weight_i * log_prob[index_i])
        n_valid = valid_log_probs.size(0)
        losses = torch.zeros(n_valid, device=self.device)
        valid_sample_mask = torch.zeros(n_valid, dtype=torch.bool, device=self.device)

        for i in range(MAX_POLICY_MOVES):
            idx = valid_indices[:, i]  # (n_valid,)
            wgt = valid_weights[:, i]  # (n_valid,)

            # Mask for valid indices (not -1)
            valid_idx_mask = idx >= 0

            if valid_idx_mask.any():
                # Gather log probs at the specified indices
                # Use clamp to avoid -1 index issues (masked out anyway)
                safe_idx = idx.clamp(min=0)
                gathered_log_probs = valid_log_probs[torch.arange(n_valid, device=self.device), safe_idx]

                # Also check that the target move is legal (in the mask)
                # This prevents -inf log_probs from causing NaN
                is_legal = valid_legal_masks[torch.arange(n_valid, device=self.device), safe_idx] > 0

                # Combine masks: index must be valid AND move must be legal
                combined_mask = valid_idx_mask & is_legal

                # Add weighted contribution (only for valid AND legal targets)
                # Use torch.where to avoid NaN from 0 * (-inf)
                weighted_log_prob = wgt * gathered_log_probs
                contribution = torch.where(combined_mask, weighted_log_prob, torch.zeros_like(weighted_log_prob))
                losses -= contribution

                # Track which samples have at least one valid target
                valid_sample_mask |= combined_mask

        # Only return mean over samples that have at least one valid target
        if valid_sample_mask.any():
            final_loss = losses[valid_sample_mask].mean()
            return final_loss
        else:
            return torch.tensor(0.0, device=self.device)

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        num_batches = 0
        # Stratified accuracy metrics
        total_w05 = 0  # |eval| < 1, within 0.5
        total_w1 = 0   # 1 <= |eval| < 5, within 1
        total_w3 = 0   # 5 <= |eval| < 80, within 3
        total_w90 = 0  # |eval| >= 80, within 100
        total_b1_samples = 0  # Count for bucket 1
        total_b2_samples = 0  # Count for bucket 2
        total_b3_samples = 0  # Count for bucket 3
        total_b4_samples = 0  # Count for bucket 4
        total_policy_samples = 0
        total_top3 = 0
        total_top5 = 0
        total_top7 = 0
        total_topk_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            boards, evals, policy_indices, policy_weights, has_policy, legal_move_indices = batch
            boards = boards.to(self.device)
            evals = evals.to(self.device)
            policy_indices = policy_indices.to(self.device)
            policy_weights = policy_weights.to(self.device)
            has_policy = has_policy.to(self.device)
            legal_move_indices = legal_move_indices.to(self.device)

            # Convert sparse legal move indices to dense mask on GPU
            legal_move_masks = legal_indices_to_mask(legal_move_indices, self.device)

            self.optimizer.zero_grad()

            # Forward pass
            value_pred, policy_logits = self.model(boards)

            # Value loss
            value_loss = self.value_criterion(value_pred, evals)

            # Policy loss
            policy_loss = self._compute_policy_loss(
                policy_logits, policy_indices, policy_weights, has_policy, legal_move_masks
            )
            loss = value_loss + self.policy_weight * policy_loss
            total_policy_loss += policy_loss.item()
            total_policy_samples += has_policy.sum().item()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

            # Calculate stratified accuracy metrics and top-k policy accuracy
            with torch.no_grad():
                pred_pawns = denormalize_eval(value_pred.cpu().numpy())
                true_pawns = denormalize_eval(evals.cpu().numpy())

                w05, b1_s, w1, b2_s, w3, b3_s, w90, b4_s = compute_stratified_accuracy(pred_pawns, true_pawns)
                total_w05 += w05
                total_b1_samples += b1_s
                total_w1 += w1
                total_b2_samples += b2_s
                total_w3 += w3
                total_b3_samples += b3_s
                total_w90 += w90
                total_b4_samples += b4_s

                # Top-k policy accuracy
                top3, top5, top7, topk_samples = compute_topk_accuracy(
                    policy_logits, policy_indices, policy_weights, has_policy, legal_move_masks, self.device
                )
                total_top3 += top3
                total_top5 += top5
                total_top7 += top7
                total_topk_samples += topk_samples

        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'value_loss': total_value_loss / num_batches if num_batches > 0 else 0.0,
            'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0.0,
            'w05_acc': total_w05 / total_b1_samples if total_b1_samples > 0 else 0.0,
            'w1_acc': total_w1 / total_b2_samples if total_b2_samples > 0 else 0.0,
            'w3_acc': total_w3 / total_b3_samples if total_b3_samples > 0 else 0.0,
            'w90_acc': total_w90 / total_b4_samples if total_b4_samples > 0 else 0.0,
            'top3_acc': total_top3 / total_topk_samples if total_topk_samples > 0 else 0.0,
            'top5_acc': total_top5 / total_topk_samples if total_topk_samples > 0 else 0.0,
            'top7_acc': total_top7 / total_topk_samples if total_topk_samples > 0 else 0.0,
        }
        return metrics

    def validate(self, dataloader: DataLoader) -> dict:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_value_loss = 0.0
        total_policy_loss = 0.0
        total_abs_error = 0.0
        # Stratified accuracy metrics
        total_w05 = 0  # |eval| < 1, within 0.5
        total_w1 = 0   # 1 <= |eval| < 5, within 1
        total_w3 = 0   # 5 <= |eval| < 80, within 3
        total_w90 = 0  # |eval| >= 80, within 100
        total_b1_samples = 0  # Count for bucket 1
        total_b2_samples = 0  # Count for bucket 2
        total_b3_samples = 0  # Count for bucket 3
        total_b4_samples = 0  # Count for bucket 4
        num_samples = 0
        num_batches = 0
        total_policy_samples = 0
        total_top3 = 0
        total_top5 = 0
        total_top7 = 0
        total_topk_samples = 0
        total_entropy = 0.0
        total_max_prob = 0.0
        total_entropy_samples = 0
        entropy_values = []

        with torch.no_grad():
            for batch in dataloader:
                boards, evals, policy_indices, policy_weights, has_policy, legal_move_indices = batch
                boards = boards.to(self.device)
                evals = evals.to(self.device)
                policy_indices = policy_indices.to(self.device)
                policy_weights = policy_weights.to(self.device)
                has_policy = has_policy.to(self.device)
                legal_move_indices = legal_move_indices.to(self.device)

                # Convert sparse legal move indices to dense mask on GPU
                legal_move_masks = legal_indices_to_mask(legal_move_indices, self.device)

                # Forward pass
                value_pred, policy_logits = self.model(boards)

                # Value loss
                value_loss = self.value_criterion(value_pred, evals)

                # Policy loss
                policy_loss = self._compute_policy_loss(
                    policy_logits, policy_indices, policy_weights, has_policy, legal_move_masks
                )
                loss = value_loss + self.policy_weight * policy_loss
                total_policy_loss += policy_loss.item()
                total_policy_samples += has_policy.sum().item()

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

                # Calculate absolute error in pawn units and stratified accuracy
                pred_pawns = denormalize_eval(value_pred.cpu().numpy())
                true_pawns = denormalize_eval(evals.cpu().numpy())
                abs_errors = np.abs(pred_pawns - true_pawns)
                total_abs_error += np.sum(abs_errors)

                # Stratified accuracy by eval magnitude
                w05, b1_s, w1, b2_s, w3, b3_s, w90, b4_s = compute_stratified_accuracy(pred_pawns, true_pawns)
                total_w05 += w05
                total_b1_samples += b1_s
                total_w1 += w1
                total_b2_samples += b2_s
                total_w3 += w3
                total_b3_samples += b3_s
                total_w90 += w90
                total_b4_samples += b4_s

                # Top-k policy accuracy
                top3, top5, top7, topk_samples = compute_topk_accuracy(
                    policy_logits, policy_indices, policy_weights, has_policy, legal_move_masks, self.device
                )
                total_top3 += top3
                total_top5 += top5
                total_top7 += top7
                total_topk_samples += topk_samples

                # Entropy and max_prob of policy distribution over legal moves
                batch_size = boards.size(0)
                logits_flat = policy_logits.view(batch_size, -1)
                mask_flat = legal_move_masks.view(batch_size, -1)

                for i in range(batch_size):
                    legal_mask_i = mask_flat[i] > 0
                    n_legal = legal_mask_i.sum().item()
                    if n_legal > 0:
                        legal_logits_i = logits_flat[i][legal_mask_i]
                        probs_i = F.softmax(legal_logits_i, dim=0)
                        log_probs_i = torch.log2(torch.clamp(probs_i, min=1e-10))
                        entropy_i = -(probs_i * log_probs_i).sum().item()
                        max_prob_i = probs_i.max().item()
                        total_entropy += entropy_i
                        total_max_prob += max_prob_i
                        total_entropy_samples += 1
                        entropy_values.append(entropy_i)

                num_samples += boards.size(0)

        # Compute entropy percentiles
        if entropy_values:
            entropy_arr = np.array(entropy_values)
            median_entropy = float(np.median(entropy_arr))
            p25_entropy = float(np.percentile(entropy_arr, 25))
            p75_entropy = float(np.percentile(entropy_arr, 75))
        else:
            median_entropy = 0.0
            p25_entropy = 0.0
            p75_entropy = 0.0

        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'value_loss': total_value_loss / num_batches if num_batches > 0 else 0.0,
            'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0.0,
            'mae_pawns': total_abs_error / num_samples if num_samples > 0 else 0.0,
            'w05_acc': total_w05 / total_b1_samples if total_b1_samples > 0 else 0.0,
            'w1_acc': total_w1 / total_b2_samples if total_b2_samples > 0 else 0.0,
            'w3_acc': total_w3 / total_b3_samples if total_b3_samples > 0 else 0.0,
            'w90_acc': total_w90 / total_b4_samples if total_b4_samples > 0 else 0.0,
            'top3_acc': total_top3 / total_topk_samples if total_topk_samples > 0 else 0.0,
            'top5_acc': total_top5 / total_topk_samples if total_topk_samples > 0 else 0.0,
            'top7_acc': total_top7 / total_topk_samples if total_topk_samples > 0 else 0.0,
            'avg_entropy': total_entropy / total_entropy_samples if total_entropy_samples > 0 else 0.0,
            'avg_max_prob': total_max_prob / total_entropy_samples if total_entropy_samples > 0 else 0.0,
            'median_entropy': median_entropy,
            'p25_entropy': p25_entropy,
            'p75_entropy': p75_entropy,
        }
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        save_path: str = 'chess_model.pt',
        save_best: bool = True
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_path: Path to save the model
            save_best: Whether to save the best model based on validation loss
        """
        best_combined_metric = float('inf')

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Model selection: combined metric = val_loss + 0.1 * avg_entropy")
        print("-" * 80)

        # Count samples in each stratified bucket
        print("Counting samples per evaluation bucket...")
        train_b1, train_b2, train_b3, train_b4 = count_eval_buckets(train_loader)
        print_bucket_stats("Training", train_b1, train_b2, train_b3, train_b4)
        val_b1, val_b2, val_b3, val_b4 = count_eval_buckets(val_loader)
        print_bucket_stats("Validation", val_b1, val_b2, val_b3, val_b4)
        print("-" * 80)

        # Evaluate before any training (Epoch 0)
        val_m0 = self.validate(val_loader)
        lr0 = self.optimizer.param_groups[0]['lr']
        parts0 = [
            f"Ep 00:",
            f"v_loss:{val_m0['loss']:.4f}",
            f"(v:{val_m0['value_loss']:.4f} p:{val_m0['policy_loss']:.4f})",
            f"mae:{val_m0['mae_pawns']:.2f}",
            f"w.5:{val_m0['w05_acc']:.1%} w1:{val_m0['w1_acc']:.1%} w3:{val_m0['w3_acc']:.1%} w90:{val_m0['w90_acc']:.1%}",
            f"t3/5/7:{val_m0['top3_acc']:.1%}/{val_m0['top5_acc']:.1%}/{val_m0['top7_acc']:.1%}",
            f"ent:{val_m0['avg_entropy']:.2f}(med:{val_m0['median_entropy']:.2f})",
            f"max_p:{val_m0['avg_max_prob']:.3f}",
            f"lr:{str(lr0).replace('0.', '.')}"
        ]
        print(" ".join(parts0))
        print("-" * 80)

        for epoch in range(epochs):
            epoch_start = time.time()
            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            epoch_mins = (time.time() - epoch_start) / 60.0
            current_lr = self.optimizer.param_groups[0]['lr']

            # Combined metric for model selection: low loss + low entropy (confident policy)
            combined_metric = val_metrics['loss'] + 0.1 * val_metrics['avg_entropy']

            # Build log string
            log_parts = [f"Ep {epoch+1:02d} ({epoch_mins:.1f}m):"]
            log_parts.append(f"loss:{train_metrics['loss']:.4f}")
            log_parts.append(f"(v:{train_metrics['value_loss']:.4f} p:{train_metrics['policy_loss']:.4f})")
            log_parts.append(f"t_w.5:{train_metrics['w05_acc']:.1%} w1:{train_metrics['w1_acc']:.1%} w3:{train_metrics['w3_acc']:.1%} w90:{train_metrics['w90_acc']:.1%}")
            log_parts.append(f"t3/5/7:{train_metrics['top3_acc']:.1%}/{train_metrics['top5_acc']:.1%}/{train_metrics['top7_acc']:.1%}")
            log_parts.append(f"| v_loss:{val_metrics['loss']:.4f}")
            log_parts.append(f"(v:{val_metrics['value_loss']:.4f} p:{val_metrics['policy_loss']:.4f})")
            log_parts.append(f"mae:{val_metrics['mae_pawns']:.2f}")
            log_parts.append(f"v_w.5:{val_metrics['w05_acc']:.1%} w1:{val_metrics['w1_acc']:.1%} w3:{val_metrics['w3_acc']:.1%} w90:{val_metrics['w90_acc']:.1%}")
            log_parts.append(f"v3/5/7:{val_metrics['top3_acc']:.1%}/{val_metrics['top5_acc']:.1%}/{val_metrics['top7_acc']:.1%}")
            log_parts.append(f"ent:{val_metrics['avg_entropy']:.2f}(med:{val_metrics['median_entropy']:.2f})")
            log_parts.append(f"max_p:{val_metrics['avg_max_prob']:.3f}")
            log_parts.append(f"lr:{str(current_lr).replace('0.', '.')}")

            # Save best model using combined metric
            if save_best and combined_metric < best_combined_metric:
                best_combined_metric = combined_metric
                epoch_path = save_path.replace('.pt', f'_epoch{epoch+1:03d}.pt')
                self.save_model(epoch_path)
                log_parts.append("[SAVED]")

            print(" ".join(log_parts))

        # Final save
        if not save_best:
            epoch_path = save_path.replace('.pt', f'_epoch{epochs:03d}.pt')
            self.save_model(epoch_path)

    def save_model(self, path: str):
        """Save model checkpoint to a local file."""
        ensure_parent_dir(path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint_for_finetuning(self, path: str):
        """
        Load model weights from a local checkpoint for fine-tuning.
        Only loads model weights, not optimizer state (uses fresh optimizer with new LR).
        """
        print(f"Loading checkpoint from {path} for fine-tuning...")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights. Using fresh optimizer with LR={self.optimizer.param_groups[0]['lr']}")


# =============================================================================
# INFERENCE
# =============================================================================

class ChessEvaluator:
    """
    Evaluator for chess positions using the trained neural network.
    Supports both value evaluation and policy (move) prediction.
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to the saved model (filename in S3)
            device: Device to use for inference
        """
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model = ChessNet(
            initial_channels=512,
            res_channels=256,
            num_res_blocks=16
        )

        # Load model from local checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, fen: str) -> float:
        """
        Evaluate a chess position.

        Args:
            fen: FEN string of the position

        Returns:
            Evaluation in pawn units (positive = white advantage)
        """
        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value, _ = self.model(board_tensor)
            eval_norm = value.item()

        return denormalize_eval(eval_norm)

    def evaluate_batch(self, fens: List[str]) -> List[float]:
        """
        Evaluate multiple positions.

        Args:
            fens: List of FEN strings

        Returns:
            List of evaluations in pawn units
        """
        tensors = torch.stack([fen_to_tensor(fen) for fen in fens]).to(self.device)

        with torch.no_grad():
            value, _ = self.model(tensors)
            evals_norm = value.cpu().numpy().flatten()

        return [denormalize_eval(e) for e in evals_norm]

    def get_policy(self, fen: str) -> np.ndarray:
        """
        Get policy (move probability) predictions for a position.

        Args:
            fen: FEN string of the position

        Returns:
            Policy logits as a (73, 8, 8) numpy array
        """
        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, policy_logits = self.model(board_tensor)

        return policy_logits.squeeze(0).cpu().numpy()

    def get_top_moves(
        self,
        fen: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top predicted moves using the 73-plane policy output.

        Args:
            fen: FEN string of the position
            top_k: Number of top moves to return

        Returns:
            List of (move_uci, probability) tuples
        """
        policy_logits = self.get_policy(fen)
        # policy_logits shape: (73, 8, 8)
        policy_tensor = torch.tensor(policy_logits).unsqueeze(0)  # (1, 73, 8, 8)
        return get_top_moves_from_policy(policy_tensor, top_k)[0]

    def evaluate_with_policy(self, fen: str, top_k: int = 5) -> dict:
        """
        Get both evaluation and policy predictions.

        Args:
            fen: FEN string of the position
            top_k: Number of top moves to return

        Returns:
            Dictionary with 'eval_pawns' and 'top_moves'
        """
        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value, policy_logits = self.model(board_tensor)

            result = {
                'eval_pawns': denormalize_eval(value.item()),
            }

            top_moves = get_top_moves_from_policy(policy_logits, top_k)[0]
            result['top_moves'] = top_moves

        return result

    def evaluate_batch_with_policy(
        self,
        fens: List[str],
        top_k: int = 5
    ) -> Tuple[List[float], List[List[Tuple[str, float]]]]:
        """
        Get both evaluations and top moves for multiple positions.

        Args:
            fens: List of FEN strings
            top_k: Number of top moves to return per position

        Returns:
            Tuple of (scores, top_moves_list) where:
                - scores: List of evaluations in pawn units
                - top_moves_list: List of lists, each containing (move_uci, prob) tuples
        """
        tensors = torch.stack([fen_to_tensor(fen) for fen in fens]).to(self.device)

        with torch.no_grad():
            values, policy_logits = self.model(tensors)
            evals_norm = values.cpu().numpy().flatten()
            scores = [denormalize_eval(e) for e in evals_norm]

            top_moves_list = get_top_moves_from_policy(policy_logits, top_k)

        return scores, top_moves_list


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main function for training the chess evaluation and policy neural network.
    Loads data from local CSV files and saves models locally.
    """
    # ==========================================================================
    # CONFIGURATION - Modify these variables as needed
    # ==========================================================================
    TRAIN_DATA = 'combined_training_data.csv'  # Local training data CSV path
    VAL_DATA = 'combined_test.csv'             # Local validation data CSV path
    EPOCHS = 100                          # Number of training epochs
    BATCH_SIZE = 2048                     # Batch size (optimized for A10G 24GB GPU)
    LEARNING_RATE = 0.001                 # Learning rate
    SAVE_PATH = 'attempt_13.pt'           # Local model checkpoint path
    DEVICE = 'auto'                       # Device: 'cpu', 'cuda', 'mps', or 'auto'
    POLICY_WEIGHT = 0.1                   # Weight for policy loss
    POLICY_TEMPERATURE = 1.0              # Softmax temperature for policy targets

    # Fine-tuning options
    CHECKPOINT_PATH = None                # Local checkpoint path (None = train from scratch)
    # CHECKPOINT_PATH = 'chess_model_epoch057.pt'  # Example: uncomment to fine-tune
    # ==========================================================================

    if CHECKPOINT_PATH:
        print(f"Mode: FINE-TUNING from {CHECKPOINT_PATH}")
        print(f"Learning rate: {LEARNING_RATE} (recommend 0.0001 for fine-tuning)")
    else:
        print("Mode: Training from scratch")
    print("=" * 80)

    # Load training data from local CSV
    train_data = load_data_from_csv(TRAIN_DATA)
    print(f"Loaded {len(train_data):,} training positions")

    # Load validation data from local CSV
    val_data = load_data_from_csv(VAL_DATA)
    print(f"Loaded {len(val_data):,} validation positions")

    # Create datasets and dataloaders
    train_dataset = ChessDataset(
        train_data,
        policy_temperature=POLICY_TEMPERATURE
    )
    val_dataset = ChessDataset(
        val_data,
        policy_temperature=POLICY_TEMPERATURE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create model and trainer
    model = ChessNet(
        initial_channels=512,
        res_channels=256,
        num_res_blocks=16
    )
    trainer = Trainer(
        model,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        policy_weight=POLICY_WEIGHT
    )

    # Load checkpoint for fine-tuning if specified
    if CHECKPOINT_PATH:
        trainer.load_checkpoint_for_finetuning(CHECKPOINT_PATH)

    # Train
    trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        save_path=SAVE_PATH
    )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
