"""
H1 sanity probe: for N random buffer rows, compare MCTS visit distribution
against M0's policy and the trunk's policy.

Usage:
    ~/myvenv/bin/python probe_h1.py [--n 5] [--seed 0]
"""
import argparse
import csv
import math
import random
from typing import List, Tuple

import chess
import numpy as np
import torch

from chess_engine import ChessNet, board_to_tensor, move_to_policy_index


BUFFER = "selfplay_run02/replay_buffer.csv"
M0_PATH = "attempt_14b_epoch032.pt"
TRUNK_PATH = "selfplay_run02/models/train_trunk.pt"


def load_model(path: str, device: torch.device) -> ChessNet:
    model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=16)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def sample_rows(path: str, n: int, rng: random.Random) -> List[dict]:
    """Reservoir sample n rows from the CSV (excluding header)."""
    reservoir: List[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < n:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = row
    return reservoir


def decode_target(row: dict) -> Tuple[List[str], np.ndarray]:
    moves = row["policy_moves"].split(",") if row["policy_moves"] else []
    visits = (
        np.array([int(v) for v in row["policy_visits"].split(",")], dtype=np.float64)
        if row["policy_visits"]
        else np.zeros(0)
    )
    total = visits.sum()
    if total > 0:
        target = visits / total
    else:
        target = np.ones(len(moves)) / max(1, len(moves))
    return moves, target


def model_legal_probs(model: ChessNet, fen: str, moves: List[str], device: torch.device) -> np.ndarray:
    board = chess.Board(fen)
    bt = board_to_tensor(board)  # (18, 8, 8)
    with torch.no_grad():
        x = bt.unsqueeze(0).to(device)
        _, policy_logits = model(x)  # (1, 73, 8, 8)
    flat = policy_logits.view(-1).cpu().numpy().astype(np.float64)

    indices = []
    for uci in moves:
        idx = move_to_policy_index(uci)
        if idx is None:
            indices.append(-1)
        else:
            r, c, p = idx
            indices.append(p * 64 + r * 8 + c)
    indices_arr = np.array(indices, dtype=np.int64)
    valid_mask = indices_arr >= 0
    legal_logits = np.full(len(moves), -1e18)
    legal_logits[valid_mask] = flat[indices_arr[valid_mask]]
    legal_logits -= legal_logits.max()
    p = np.exp(legal_logits)
    p /= p.sum()
    return p


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def cross_entropy(target: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(pred, 1e-12, 1.0)
    return float(-(target * np.log(pred)).sum())


def topk(moves: List[str], probs: np.ndarray, k: int = 5):
    order = np.argsort(probs)[::-1][:k]
    return [(moves[i], float(probs[i])) for i in order]


def fmt_top(items):
    return "  ".join(f"{m}:{p:.3f}" for m, p in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"Sampling {args.n} rows from {BUFFER} ...")
    rows = sample_rows(BUFFER, args.n, rng)

    print("Loading M0 ...")
    m0 = load_model(M0_PATH, device)
    print("Loading trunk ...")
    trunk = load_model(TRUNK_PATH, device)

    for k, row in enumerate(rows):
        fen = row["fen"]
        moves, target = decode_target(row)
        side = "W" if " w " in fen else "B"

        m0_probs = model_legal_probs(m0, fen, moves, device)
        tr_probs = model_legal_probs(trunk, fen, moves, device)

        print(f"\n========== Sample {k+1}/{args.n} ==========")
        print(f"FEN: {fen}")
        print(f"side-to-move: {side}   legal_moves: {len(moves)}   "
              f"value_target: {row['value_target']}   "
              f"total_visits: {int(np.array([int(v) for v in row['policy_visits'].split(',')]).sum())}")
        print(f"MCTS    entropy={entropy(target):.3f}    top: {fmt_top(topk(moves, target))}")
        print(f"M0      entropy={entropy(m0_probs):.3f}    top: {fmt_top(topk(moves, m0_probs))}")
        print(f"trunk   entropy={entropy(tr_probs):.3f}    top: {fmt_top(topk(moves, tr_probs))}")
        print(f"CE(MCTS, M0)    = {cross_entropy(target, m0_probs):.3f}")
        print(f"CE(MCTS, trunk) = {cross_entropy(target, tr_probs):.3f}")

        m_top = moves[int(np.argmax(target))]
        m0_top = moves[int(np.argmax(m0_probs))]
        tr_top = moves[int(np.argmax(tr_probs))]
        print(f"top-1: MCTS={m_top}   M0={m0_top}{'  ✓' if m0_top == m_top else ''}"
              f"   trunk={tr_top}{'  ✓' if tr_top == m_top else ''}")


if __name__ == "__main__":
    main()
