"""
Interactive REPL: paste a FEN, get the NN's value and policy for that position.

Usage:
    ~/myvenv/bin/python probe_fen.py <model.pth> [--device cuda|cpu] [--top-k 10]

Then type or paste a FEN per line; Ctrl-D (or 'q') to quit.

Value is shown both as raw white-POV in [-1, 1] (network output) and as pawns
(via denormalize_eval), plus a side-to-move convenience number.  Policy is
softmaxed over LEGAL moves only and the top-K are listed.
"""

import argparse
import sys

import chess
import numpy as np
import torch

from chess_engine import (
    ChessNet,
    board_to_tensor,
    denormalize_eval,
    move_to_policy_index,
)


def _legal_move_logits(flat_policy: np.ndarray, board: chess.Board):
    legal = list(board.legal_moves)
    logits = np.full(len(legal), -1e9, dtype=np.float64)
    for i, mv in enumerate(legal):
        idx = move_to_policy_index(mv.uci())
        if idx is None:
            continue
        src_row, src_col, plane = idx
        logits[i] = float(flat_policy[plane * 64 + src_row * 8 + src_col])
    return legal, logits


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    p = np.exp(x)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))


@torch.no_grad()
def probe(model, board: chess.Board, device: torch.device, top_k: int):
    x = board_to_tensor(board).unsqueeze(0).to(device)
    value, policy_logits = model(x)
    v_white = float(value.reshape(-1).float().cpu().numpy()[0])
    flat = policy_logits.reshape(-1).float().cpu().numpy()

    legal, logits = _legal_move_logits(flat, board)
    if not legal:
        return v_white, []
    probs = _softmax(logits)
    order = np.argsort(-probs)[:top_k]
    return v_white, [(legal[i].uci(), float(probs[i]), float(logits[i])) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to .pth checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = ChessNet()
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded {args.model} on {device}.  Enter FEN (blank/'q' to quit).")

    while True:
        try:
            line = input("FEN> ").strip()
        except EOFError:
            print()
            break
        if not line or line.lower() in ("q", "quit", "exit"):
            break
        try:
            board = chess.Board(line)
        except ValueError as e:
            print(f"  invalid FEN: {e}")
            continue

        v_white, top = probe(model, board, device, args.top_k)
        v_pawns = float(denormalize_eval(v_white))
        stm_sign = 1.0 if board.turn == chess.WHITE else -1.0
        print(f"  value (white-POV): {v_white:+.4f}  "
              f"~{v_pawns:+.2f} pawns  "
              f"(stm-POV {stm_sign * v_white:+.4f})")
        if board.is_game_over(claim_draw=True):
            print(f"  position is terminal: {board.result(claim_draw=True)}")
            continue
        print(f"  top {len(top)} legal moves by policy:")
        for uci, p, lg in top:
            print(f"    {uci:6s}  p={p:6.2%}  logit={lg:+.3f}")


if __name__ == "__main__":
    main()
