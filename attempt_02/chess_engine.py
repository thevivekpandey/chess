"""
Neural Network Chess Engine
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

Neural Network: Conv + BatchNorm -> 8 ResNet blocks -> 2 FC layers -> tanh
Output: Evaluation between -1 (black winning) and +1 (white winning)
"""

import logging
# Suppress debug messages from Databricks thread monitor
logging.getLogger('ThreadMonitor').setLevel(logging.WARNING)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
import os
import io
import boto3

# S3 Configuration
S3_BUCKET = "adhoc-query-data"
S3_PREFIX = "vivek.pandey/engine"


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
        eval_pawns: Evaluation in pawn units
        max_pawns: Scaling factor (evaluation at this value maps to ~0.76)

    Returns:
        Normalized evaluation in [-1, 1]
    """
    return np.tanh(eval_pawns / max_pawns)


def denormalize_eval(eval_norm: float, max_pawns: float = 10.0) -> float:
    """
    Convert normalized evaluation back to pawn units.

    Args:
        eval_norm: Normalized evaluation in [-1, 1]
        max_pawns: Scaling factor used in normalization

    Returns:
        Evaluation in pawn units
    """
    # Clamp to avoid atanh domain issues
    eval_norm = np.clip(eval_norm, -0.99999, 0.99999)
    return np.arctanh(eval_norm) * max_pawns


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
    Neural network for chess position evaluation.

    Architecture:
    - Input: 18 planes of 8x8
    - Initial conv (18 -> 512 channels) + BatchNorm + ReLU
    - Transition conv (512 -> 256 channels) + BatchNorm + ReLU
    - 8 Residual blocks (256 channels)
    - Value head: Conv 256->32, FC 2048->256->1
    - Tanh output
    """

    def __init__(self, initial_channels: int = 512, res_channels: int = 256, num_res_blocks: int = 8):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 18, 8, 8)

        Returns:
            Evaluation tensor of shape (batch, 1) in range [-1, 1]
        """
        # Initial convolution (18 -> 512)
        out = F.relu(self.initial_bn(self.initial_conv(x)))

        # Transition (512 -> 256)
        out = F.relu(self.transition_bn(self.transition_conv(out)))

        # Residual tower (256 channels)
        for block in self.res_blocks:
            out = block(out)

        # Value head
        out = F.relu(self.value_bn(self.value_conv(out)))
        out = out.view(out.size(0), -1)  # Flatten
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))

        return out


# =============================================================================
# DATASET
# =============================================================================

class ChessDataset(Dataset):
    """
    Dataset for chess position evaluations.
    Precomputes all board tensors for faster training.

    Expected data format: List of (fen, eval_pawns) tuples
    where eval_pawns is the evaluation in pawn units.
    """

    def __init__(self, data: List[Tuple[str, float]], max_pawns: float = 10.0):
        """
        Args:
            data: List of (fen, eval_pawns) tuples
            max_pawns: Scaling factor for evaluation normalization
        """
        self.max_pawns = max_pawns

        # Precompute all board tensors and normalized evaluations
        print(f"Precomputing {len(data):,} board tensors...")
        self.board_tensors = []
        self.eval_tensors = []

        for i, (fen, eval_pawns) in enumerate(data):
            # Convert FEN to tensor
            board_tensor = fen_to_tensor(fen)
            self.board_tensors.append(board_tensor)

            # Normalize evaluation
            eval_norm = normalize_eval(eval_pawns, self.max_pawns)
            self.eval_tensors.append(torch.tensor([eval_norm], dtype=torch.float32))

            # Progress report every 500K positions
            if (i + 1) % 500_000 == 0:
                print(f"  Processed {i + 1:,} / {len(data):,} positions...")

        # Stack into single tensors for efficiency
        self.board_tensors = torch.stack(self.board_tensors)
        self.eval_tensors = torch.stack(self.eval_tensors)
        print(f"Done. Board tensors shape: {self.board_tensors.shape}")

    def __len__(self) -> int:
        return len(self.board_tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.board_tensors[idx], self.eval_tensors[idx]


def load_data_from_s3(filename: str) -> List[Tuple[str, float]]:
    """
    Load training data from S3.

    Expected format: CSV with header "fen,eval" where eval is in pawn units.
    Mate scores are represented as +100 or -100.

    Args:
        filename: Name of the CSV file in S3 (e.g., 'training_data.csv')

    Returns:
        List of (fen, eval_pawns) tuples
    """
    import csv

    s3_client = boto3.client('s3')
    s3_key = f"{S3_PREFIX}/{filename}"

    print(f"Downloading s3://{S3_BUCKET}/{s3_key}...")
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
    content = response['Body'].read().decode('utf-8')

    data = []
    reader = csv.reader(io.StringIO(content))
    header = next(reader)  # Skip header row

    for row in reader:
        if len(row) >= 2:
            fen = row[0].strip()
            try:
                eval_pawns = float(row[1].strip())
                data.append((fen, eval_pawns))
            except ValueError:
                continue

    return data


def load_data_from_file(filepath: str) -> List[Tuple[str, float]]:
    """
    Load training data from a local CSV file.

    Expected format: CSV with header "fen,eval" where eval is in pawn units.
    Mate scores are represented as +100 or -100.

    Args:
        filepath: Path to the CSV data file

    Returns:
        List of (fen, eval_pawns) tuples
    """
    import csv

    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row

        for row in reader:
            if len(row) >= 2:
                fen = row[0].strip()
                try:
                    eval_pawns = float(row[1].strip())
                    data.append((fen, eval_pawns))
                except ValueError:
                    continue

    return data


# =============================================================================
# TRAINING
# =============================================================================

class Trainer:
    """
    Trainer for the chess neural network.
    """

    def __init__(
        self,
        model: ChessNet,
        device: str = 'auto',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: ChessNet model to train
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
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
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.SmoothL1Loss(beta=0.2)

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Tuple of (average loss, within-1-pawn accuracy)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_within_1 = 0
        total_samples = 0

        for batch_idx, (boards, evals) in enumerate(dataloader):
            boards = boards.to(self.device)
            evals = evals.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(boards)
            loss = self.criterion(predictions, evals)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Calculate within-1-pawn accuracy
            with torch.no_grad():
                pred_pawns = denormalize_eval(predictions.cpu().numpy())
                true_pawns = denormalize_eval(evals.cpu().numpy())
                within_1 = np.abs(pred_pawns - true_pawns) <= 1.0
                total_within_1 += np.sum(within_1)
                total_samples += boards.size(0)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        within_1_acc = total_within_1 / total_samples if total_samples > 0 else 0.0

        return avg_loss, within_1_acc

    def validate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader

        Returns:
            Tuple of (average loss, average absolute error in pawn units, within-1-pawn accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_abs_error = 0.0
        total_within_1 = 0
        num_samples = 0

        with torch.no_grad():
            for boards, evals in dataloader:
                boards = boards.to(self.device)
                evals = evals.to(self.device)

                predictions = self.model(boards)
                loss = self.criterion(predictions, evals)

                total_loss += loss.item() * boards.size(0)

                # Calculate absolute error in pawn units
                pred_pawns = denormalize_eval(predictions.cpu().numpy())
                true_pawns = denormalize_eval(evals.cpu().numpy())
                abs_errors = np.abs(pred_pawns - true_pawns)
                total_abs_error += np.sum(abs_errors)

                # Calculate within-1-pawn accuracy
                within_1 = abs_errors <= 1.0
                total_within_1 += np.sum(within_1)

                num_samples += boards.size(0)

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_abs_error = total_abs_error / num_samples if num_samples > 0 else 0.0
        within_1_acc = total_within_1 / num_samples if num_samples > 0 else 0.0

        return avg_loss, avg_abs_error, within_1_acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_path: str = 'chess_model.pt',
        save_best: bool = True
    ) -> dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            save_path: Path to save the model
            save_best: Whether to save the best model based on validation loss

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_within1_acc': [],
            'val_loss': [],
            'val_mae_pawns': [],
            'val_within1_acc': [],
            'best_model_path': None
        }
        best_val_loss = float('inf')

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 80)

        for epoch in range(epochs):
            # Training
            train_loss, train_within1_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_within1_acc'].append(train_within1_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_mae_pawns, val_within1_acc = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_mae_pawns'].append(val_mae_pawns)
                history['val_within1_acc'].append(val_within1_acc)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epoch_path = save_path.replace('.pt', f'_epoch{epoch+1:03d}.pt')
                    self.save_model(epoch_path)
                    history['best_model_path'] = epoch_path
                    print(f"Epoch {epoch+1:3d}: loss={train_loss:.4f}, "
                          f"train_w1={train_within1_acc:.1%}, "
                          f"val_loss={val_loss:.4f}, val_mae={val_mae_pawns:.2f}, "
                          f"val_w1={val_within1_acc:.1%} [SAVED]")
                else:
                    print(f"Epoch {epoch+1:3d}: loss={train_loss:.4f}, "
                          f"train_w1={train_within1_acc:.1%}, "
                          f"val_loss={val_loss:.4f}, val_mae={val_mae_pawns:.2f}, "
                          f"val_w1={val_within1_acc:.1%}")
            else:
                print(f"Epoch {epoch+1:3d}: loss={train_loss:.4f}, train_w1={train_within1_acc:.1%}")
                if (epoch + 1) % 10 == 0:
                    epoch_path = save_path.replace('.pt', f'_epoch{epoch+1:03d}.pt')
                    self.save_model(epoch_path)
                    history['best_model_path'] = epoch_path

        # Final save
        if not save_best or val_loader is None:
            epoch_path = save_path.replace('.pt', f'_epoch{epochs:03d}.pt')
            self.save_model(epoch_path)
            history['best_model_path'] = epoch_path

        return history

    def save_model(self, path: str):
        """Save model checkpoint to S3."""
        # Save to buffer first
        buffer = io.BytesIO()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, buffer)
        buffer.seek(0)

        # Upload to S3
        s3_client = boto3.client('s3')
        s3_key = f"{S3_PREFIX}/{path}"
        s3_client.upload_fileobj(buffer, S3_BUCKET, s3_key)

    def load_model(self, path: str):
        """Load model checkpoint from S3."""
        s3_client = boto3.client('s3')
        s3_key = f"{S3_PREFIX}/{path}"

        buffer = io.BytesIO()
        s3_client.download_fileobj(S3_BUCKET, s3_key, buffer)
        buffer.seek(0)

        checkpoint = torch.load(buffer, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# =============================================================================
# INFERENCE
# =============================================================================

class ChessEvaluator:
    """
    Evaluator for chess positions using the trained neural network.
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

        self.model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=8)

        # Load model from S3
        s3_client = boto3.client('s3')
        s3_key = f"{S3_PREFIX}/{model_path}"
        print(f"Loading model from s3://{S3_BUCKET}/{s3_key}...")

        buffer = io.BytesIO()
        s3_client.download_fileobj(S3_BUCKET, s3_key, buffer)
        buffer.seek(0)

        checkpoint = torch.load(buffer, map_location=self.device)
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
            eval_norm = self.model(board_tensor).item()

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
            evals_norm = self.model(tensors).cpu().numpy().flatten()

        return [denormalize_eval(e) for e in evals_norm]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sample_data() -> List[Tuple[str, float]]:
    """
    Create sample training data for testing purposes.
    Returns a list of (fen, eval_cp) tuples.
    """
    return [
        # Starting position (roughly equal)
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 20),
        # After 1.e4 (slight white advantage)
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 30),
        # Italian Game
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", 25),
        # Sicilian Defense
        ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", 45),
        # Queen's Gambit
        ("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2", 35),
        # Ruy Lopez
        ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", 20),
        # White up a pawn
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", 100),
        # White up a piece
        ("rnbqkb1r/pppppppp/8/8/4n3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", -250),
        # Endgame position
        ("8/5k2/8/8/8/8/4K3/4R3 w - - 0 1", 500),
        # King and pawn vs king
        ("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1", 200),
    ]


def split_data(
    data: List[Tuple[str, float]],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List, List, List]:
    """
    Split data into train, validation, and test sets.

    Args:
        data: Full dataset
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(data))

    n_test = int(len(data) * test_ratio)
    n_val = int(len(data) * val_ratio)

    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main function for training the chess evaluation neural network.
    Loads data from S3 and saves models to S3.

    S3 Location: s3://adhoc-query-data/vivek.pandey/engine/
    """
    # ==========================================================================
    # CONFIGURATION - Modify these variables as needed
    # ==========================================================================
    TRAIN_DATA = 'training_data.csv'      # Training data CSV filename in S3
    VAL_DATA = 'test_data.csv'            # Validation data CSV filename in S3
    EPOCHS = 100                          # Number of training epochs
    BATCH_SIZE = 2048                     # Batch size (optimized for A10G 24GB GPU)
    LEARNING_RATE = 0.001                 # Learning rate
    SAVE_PATH = 'chess_model.pt'          # Model save filename in S3
    DEVICE = 'auto'                       # Device: 'cpu', 'cuda', 'mps', or 'auto'
    # ==========================================================================

    print(f"S3 Location: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print("=" * 80)

    # Load training data from S3
    train_data = load_data_from_s3(TRAIN_DATA)
    print(f"Loaded {len(train_data):,} training positions")

    # Load validation data from S3
    val_data = load_data_from_s3(VAL_DATA)
    print(f"Loaded {len(val_data):,} validation positions")

    # Create datasets and dataloaders
    train_dataset = ChessDataset(train_data)
    val_dataset = ChessDataset(val_data)

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
    model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=8)
    trainer = Trainer(model, device=DEVICE, learning_rate=LEARNING_RATE)

    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        save_path=SAVE_PATH
    )

    print("\nTraining complete!")
    best_model_path = history.get('best_model_path', SAVE_PATH)
    print(f"Model saved to s3://{S3_BUCKET}/{S3_PREFIX}/{best_model_path}")

    # Sample predictions on validation data
    print("\nSample predictions on validation data:")
    evaluator = ChessEvaluator(best_model_path, device=DEVICE)
    for fen, true_pawns in val_data[:5]:
        pred_pawns = evaluator.evaluate(fen)
        print(f"  True: {true_pawns:+.2f}, Pred: {pred_pawns:+.2f} pawns")


if __name__ == '__main__':
    main()
