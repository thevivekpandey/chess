"""
Test 5c: terminal-position sign check.

For every game in a single prefill CSV, find the actual highest-ply position
(the true game-end position) and verify:
- value_target sign matches the game outcome consistent with the position
- For games ending in checkmate, value sign matches winner
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


def ply_index(board: chess.Board) -> int:
    return (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "prefill_cache/prefill_0000.csv"
    print(f"Loading {path} ...")

    rows_by_game = defaultdict(list)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_game[row["game_id"]].append(row)

    n_games = len(rows_by_game)
    print(f"Total games: {n_games}")

    # For each game, find the row with highest ply (the latest position recorded)
    mismatch_mate = 0
    mismatch_lopsided = 0
    n_terminal = 0
    n_lopsided = 0
    examples = []

    for gid, rows in rows_by_game.items():
        # Pick row with max ply
        last_row = max(rows, key=lambda r: ply_index(chess.Board(r["fen"])))
        fen = last_row["fen"]
        v = float(last_row["value_target"])
        board = chess.Board(fen)
        mat = material_balance(board)
        ply = ply_index(board)

        # Note: in the buffer, terminal positions ARE included (after game ends).
        # If board.is_checkmate(): the side TO MOVE is checkmated -> opposite side won.
        if board.is_checkmate():
            n_terminal += 1
            expected_v = +1.0 if board.turn == chess.BLACK else -1.0
            if abs(v - expected_v) > 1e-6:
                mismatch_mate += 1
                if mismatch_mate <= 5:
                    examples.append(f"  [MATE-MISMATCH] {gid}: ply={ply} side_to_mate={'W' if board.turn==chess.WHITE else 'B'} expected={expected_v:+.1f} got={v:+.1f} fen={fen}")
            continue
        if board.is_stalemate() or board.is_insufficient_material():
            n_terminal += 1
            if v != 0.0:
                mismatch_mate += 1
                if mismatch_mate <= 5:
                    examples.append(f"  [DRAW-MISMATCH] {gid}: ply={ply} expected=0 got={v:+.1f} fen={fen}")
            continue

        # Non-terminal final position. Could be 50-move, 3-fold, or max plies.
        # Use lopsided material as a soft check.
        if abs(mat) >= 7:
            n_lopsided += 1
            if (mat > 0 and v < 0) or (mat < 0 and v > 0):
                mismatch_lopsided += 1
                if mismatch_lopsided <= 5:
                    examples.append(f"  [LOPSIDED-MISMATCH] {gid}: ply={ply} material={mat:+d} value={v:+.1f} fen={fen}")

    print(f"\nGames with terminal final position (mate/stalemate/insufficient): {n_terminal}")
    print(f"  - Mismatched signs at terminal: {mismatch_mate}")
    print(f"\nGames with non-terminal final position and lopsided (>=7) material: {n_lopsided}")
    print(f"  - Mismatched signs (winning material but lost): {mismatch_lopsided}")

    if examples:
        print(f"\nExamples:")
        for ex in examples:
            print(ex)


if __name__ == "__main__":
    main()
