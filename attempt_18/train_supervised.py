"""
Supervised training from Stockfish-labeled CSV data.

Thin wrapper around chess_engine.Trainer / ChessDataset / load_data_from_csv so
that the data format, the value normalization (tanh(pawns / 10)), the
legal-move-masked policy cross-entropy, the AMP/optimizer setup and the network
architecture all stay exactly consistent with how the base model was trained.

Produces, in `output_path`:
  - model_epoch_<N>.pth for every saved epoch (always at least the final epoch)
  - model_best.pth      (lowest validation loss seen)

The pipeline (supervised_pipeline.py) promotes `model_best.pth`.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from chess_engine import ChessDataset, ChessNet, Trainer, load_data_from_csv


def _load_dataset(csv_path: str) -> ChessDataset:
    data = load_data_from_csv(csv_path)
    if not data:
        raise ValueError(f"No usable rows in {csv_path}")
    return ChessDataset(data)


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
    device: str = "cuda",
    save_every: int = 1,
):
    """Fine-tune `model_path` on the Stockfish-labeled CSVs and save checkpoints."""
    if value_weight != 1.0:
        print(
            f"[warn] value_weight={value_weight} is ignored: chess_engine.Trainer fixes the "
            f"value-loss weight at 1.0 and scales the policy term by policy_weight ({policy_weight})."
        )

    os.makedirs(output_path, exist_ok=True)

    print("=" * 80)
    print("SUPERVISED TRAINING FROM STOCKFISH LABELS")
    print("=" * 80)
    print(f"Train data:   {train_data_path}")
    print(f"Val data:     {val_data_path}")
    print(f"Initial model:{model_path}")
    print(f"Output dir:   {output_path}")
    print(f"Epochs={num_epochs}  batch={batch_size}  lr={learning_rate}  "
          f"policy_weight={policy_weight}  device={device}")
    print()

    pin_memory = device == "cuda"
    # ChessDataset precomputes every tensor up front, so __getitem__ is a pure
    # index op -> a single-process DataLoader is the fast path here.
    train_loader = DataLoader(
        _load_dataset(train_data_path), batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        _load_dataset(val_data_path), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=pin_memory,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = ChessNet()
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No checkpoint found - training from random initialization")

    trainer = Trainer(
        model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        policy_weight=policy_weight,
    )

    best_val_loss = float("inf")
    final_epoch_path = os.path.join(output_path, f"model_epoch_{num_epochs}.pth")

    for epoch in range(1, num_epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        trainer.scheduler.step(val_metrics["loss"])

        print(
            f"Epoch {epoch:02d}/{num_epochs}  "
            f"train {train_metrics['loss']:.4f} (v {train_metrics['value_loss']:.4f} "
            f"p {train_metrics['policy_loss']:.4f})  |  "
            f"val {val_metrics['loss']:.4f} (v {val_metrics['value_loss']:.4f} "
            f"p {val_metrics['policy_loss']:.4f})  "
            f"mae {val_metrics['mae_pawns']:.2f}  "
            f"top3/5 {val_metrics['top3_acc']:.1%}/{val_metrics['top5_acc']:.1%}"
        )

        if epoch % max(1, save_every) == 0 or epoch == num_epochs:
            trainer.save_model(os.path.join(output_path, f"model_epoch_{epoch}.pth"))
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            trainer.save_model(os.path.join(output_path, "model_best.pth"))
            print(f"  -> new best (val loss {best_val_loss:.4f})")

    # The pipeline promotes model_epoch_<num_epochs>.pth; make sure it exists
    # even if save_every was set so it never lined up with the last epoch.
    if not os.path.exists(final_epoch_path):
        trainer.save_model(final_epoch_path)

    print(f"\nTraining complete. Best val loss {best_val_loss:.4f}. Checkpoints in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with supervised learning from Stockfish")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data (.csv)")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation data (.csv)")
    parser.add_argument("--model", type=str, required=True, help="Path to initial model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--policy-weight", type=float, default=1.0, help="Policy loss weight (relative to value loss)")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Ignored (kept for CLI compatibility)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda/mps)")
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
        save_every=args.save_every,
    )
