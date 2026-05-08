"""
Test 5b: verify game_id grouping on a SINGLE prefill CSV (no cross-iter collisions possible).
If value_target is constant within game_id here, the buffer data is fine and only
my game grouping was confused by global-vs-iter-local game IDs.
"""
import csv
import sys
from collections import defaultdict

import chess


def material_balance(board: chess.Board) -> int:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    diff = 0
    for piece_type, val in values.items():
        diff += val * (len(board.pieces(piece_type, chess.WHITE)) -
                       len(board.pieces(piece_type, chess.BLACK)))
    return diff


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "prefill_cache/prefill_0000.csv"
    print(f"Loading {path} ...")

    rows_by_game = defaultdict(list)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_game[row["game_id"]].append(row)

    print(f"Total games: {len(rows_by_game)}")
    sizes = [len(v) for v in rows_by_game.values()]
    print(f"Game sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")

    bad = 0
    weird_sign = 0
    sample_lines = []
    for gid in sorted(rows_by_game.keys()):
        rows = rows_by_game[gid]
        values = sorted({float(r["value_target"]) for r in rows})
        if len(values) != 1:
            bad += 1
            if bad <= 3:
                print(f"  [BAD] {gid}: mixed values {values}")
        else:
            v = values[0]
            last_fen = rows[-1]["fen"]
            board = chess.Board(last_fen)
            mat = material_balance(board)
            if (mat >= 5 and v < 0) or (mat <= -5 and v > 0):
                weird_sign += 1
                if weird_sign <= 5:
                    sample_lines.append(f"  [WEIRD-SIGN] {gid}: material={mat:+d} value={v:+.1f} last_fen={last_fen}")

    print(f"\nGames with constant value_target: {len(rows_by_game) - bad}/{len(rows_by_game)}")
    print(f"Games with lopsided material but inconsistent sign: {weird_sign}/{len(rows_by_game)}")
    for line in sample_lines:
        print(line)

    # Show a few good games for sanity
    print("\n-- 5 random good games (value vs final material) --")
    import random
    random.seed(0)
    sample = random.sample(sorted(rows_by_game.keys()), min(10, len(rows_by_game)))
    for gid in sample:
        rows = rows_by_game[gid]
        values = sorted({float(r["value_target"]) for r in rows})
        if len(values) != 1:
            continue
        v = values[0]
        last_fen = rows[-1]["fen"]
        board = chess.Board(last_fen)
        mat = material_balance(board)
        # also count side-to-move distribution
        n = len(rows)
        sides = [chess.Board(r["fen"]).turn for r in rows]
        n_w = sum(1 for s in sides if s == chess.WHITE)
        n_b = n - n_w
        # Last position state
        state = "ongoing"
        if board.is_checkmate():
            state = f"mate ({'B' if board.turn == chess.WHITE else 'W'} wins)"
        elif board.is_stalemate():
            state = "stalemate"
        elif board.is_insufficient_material():
            state = "insufficient mat"
        print(f"  {gid}: n={n} ({n_w}W/{n_b}B) value={v:+.1f} last_state={state} last_material={mat:+d}")


if __name__ == "__main__":
    main()
