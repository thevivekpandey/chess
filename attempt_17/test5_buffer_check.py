"""
Test 5: Replay buffer spot-check.

Validates that buffer rows are internally consistent:
- FEN parses to a legal position
- All listed policy_moves are legal at that FEN
- All legal moves at that FEN appear in policy_moves (coverage)
- visit_counts has same length as policy_moves, all >= 0, sum > 0
- For sampled games, value_target is constant across all positions in the game
- value_target ∈ {-1.0, 0.0, 1.0}
- Sign sanity: for terminal-ish positions, check value matches obvious material
"""
import argparse
import csv
import random
import sys
from collections import defaultdict

import chess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="selfplay_run05/replay_buffer.csv")
    ap.add_argument("--n-rows", type=int, default=20)
    ap.add_argument("--n-games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # First pass: reservoir-sample rows
    print(f"Reservoir-sampling {args.n_rows} rows from {args.buffer} ...")
    reservoir = []
    total_rows = 0
    with open(args.buffer, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            total_rows += 1
            if i < args.n_rows:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < args.n_rows:
                    reservoir[j] = row
    print(f"Buffer has {total_rows:,} rows.\n")

    # Pick game_ids for game-level checks
    sample_game_ids = set()
    for r in reservoir:
        sample_game_ids.add(r["game_id"])
        if len(sample_game_ids) >= args.n_games:
            break
    print(f"Sampling all rows for {len(sample_game_ids)} games: {sorted(sample_game_ids)}\n")

    # Second pass: collect all rows for sampled games
    game_rows = defaultdict(list)
    with open(args.buffer, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["game_id"] in sample_game_ids:
                game_rows[row["game_id"]].append(row)

    # ---- PER-ROW CHECKS ----
    print("=" * 80)
    print("PER-ROW CHECKS")
    print("=" * 80)
    n_pass = 0
    n_fail = 0
    failures = []

    for k, row in enumerate(reservoir):
        fen = row["fen"]
        policy_moves = row["policy_moves"].split(",") if row["policy_moves"] else []
        try:
            policy_visits = [int(v) for v in row["policy_visits"].split(",")] if row["policy_visits"] else []
        except ValueError as e:
            failures.append(f"row {k}: visit_counts parse error: {e}")
            n_fail += 1
            continue
        try:
            value_target = float(row["value_target"])
        except ValueError as e:
            failures.append(f"row {k}: value_target parse error: {e}")
            n_fail += 1
            continue

        # Parse FEN
        try:
            board = chess.Board(fen)
        except ValueError as e:
            failures.append(f"row {k}: FEN parse failed: {fen!r}: {e}")
            n_fail += 1
            continue

        legal_uci = {m.uci() for m in board.legal_moves}

        problems = []
        # Move legality
        illegal = [m for m in policy_moves if m not in legal_uci]
        if illegal:
            problems.append(f"illegal moves in policy_moves: {illegal}")
        # Move coverage
        missing = [m for m in legal_uci if m not in policy_moves]
        if missing:
            problems.append(f"missing legal moves from policy_moves: {missing}")
        # Visit count length
        if len(policy_visits) != len(policy_moves):
            problems.append(f"len(visits)={len(policy_visits)} != len(moves)={len(policy_moves)}")
        # Visit count values
        if any(v < 0 for v in policy_visits):
            problems.append(f"negative visits: {policy_visits}")
        if sum(policy_visits) <= 0:
            problems.append(f"sum(visits)={sum(policy_visits)} <= 0")
        # Value target range
        if value_target not in (-1.0, 0.0, 1.0):
            problems.append(f"value_target={value_target} not in {{-1, 0, 1}}")

        side = "W" if board.turn == chess.WHITE else "B"
        if problems:
            n_fail += 1
            print(f"\n[FAIL] row {k}: side={side} v={value_target} game={row['game_id']}")
            print(f"  FEN: {fen}")
            for p in problems:
                print(f"  - {p}")
            failures.append(f"row {k}: {'; '.join(problems)}")
        else:
            n_pass += 1
            print(f"[ok]   row {k}: side={side} v={value_target:+.1f} "
                  f"legal={len(legal_uci):2d} sum_visits={sum(policy_visits):4d} game={row['game_id']}")

    print(f"\nPer-row: {n_pass}/{n_pass + n_fail} passed")

    # ---- GAME-LEVEL CHECKS ----
    print("\n" + "=" * 80)
    print("GAME-LEVEL CHECKS (value_target constant across positions in a game)")
    print("=" * 80)
    game_pass = 0
    game_fail = 0
    for gid in sorted(game_rows):
        rows = game_rows[gid]
        values = sorted({float(r["value_target"]) for r in rows})
        n = len(rows)
        if len(values) != 1:
            game_fail += 1
            print(f"[FAIL] {gid}: {n} positions, MIXED value_targets={values}")
            # Show breakdown
            counts = defaultdict(int)
            for r in rows:
                counts[float(r["value_target"])] += 1
            for v, c in sorted(counts.items()):
                print(f"        value={v:+.1f}: {c} positions")
        else:
            v = values[0]
            game_pass += 1
            # Sample side-to-move distribution
            sides = [chess.Board(r["fen"]).turn for r in rows]
            n_w = sum(1 for s in sides if s == chess.WHITE)
            n_b = n - n_w
            # Find last (likely terminal) position
            last_fen = rows[-1]["fen"]
            try:
                last_board = chess.Board(last_fen)
                last_state = "ongoing"
                if last_board.is_checkmate():
                    last_state = "checkmate (loser to move)"
                elif last_board.is_stalemate():
                    last_state = "stalemate"
                elif last_board.is_insufficient_material():
                    last_state = "insufficient material"
                elif last_board.can_claim_threefold_repetition():
                    last_state = "3-fold repetition"
                elif last_board.can_claim_fifty_moves():
                    last_state = "50-move rule"
            except ValueError:
                last_state = "unparseable"
            material_diff = material_balance(chess.Board(last_fen))
            print(f"[ok]   {gid}: {n} positions ({n_w}W / {n_b}B to move) "
                  f"value={v:+.1f} last_state={last_state} last_material={material_diff:+d}")

    print(f"\nGame-level: {game_pass}/{game_pass + game_fail} games have constant value_target")

    # ---- SIGN SANITY ON LAST POSITIONS ----
    print("\n" + "=" * 80)
    print("SIGN SANITY (last position of each game vs value_target)")
    print("=" * 80)
    print("For each sampled game's last position in the buffer:")
    print("  white_material - black_material vs value_target sign")
    print()
    weird = 0
    for gid in sorted(game_rows):
        rows = game_rows[gid]
        last_fen = rows[-1]["fen"]
        v = float(rows[-1]["value_target"])
        last_board = chess.Board(last_fen)
        material_diff = material_balance(last_board)
        # If material is heavily lopsided (>= 5), value sign should match
        flag = ""
        if material_diff >= 5 and v < 0:
            flag = "  <-- WEIRD: white up material but value=negative"
            weird += 1
        elif material_diff <= -5 and v > 0:
            flag = "  <-- WEIRD: black up material but value=positive"
            weird += 1
        print(f"  {gid}: material={material_diff:+d} value={v:+.1f}{flag}")
    print(f"\n{weird} weird cases (lopsided material but inconsistent value sign)")

    print("\n" + "=" * 80)
    if n_fail == 0 and game_fail == 0 and weird == 0:
        print("OVERALL: PASS")
    else:
        print(f"OVERALL: {n_fail} per-row failures, {game_fail} game-level failures, {weird} sign sanity warnings")
    print("=" * 80)


def material_balance(board: chess.Board) -> int:
    """White minus black material in pawn units."""
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    diff = 0
    for piece_type, val in values.items():
        diff += val * (len(board.pieces(piece_type, chess.WHITE)) -
                       len(board.pieces(piece_type, chess.BLACK)))
    return diff


if __name__ == "__main__":
    main()
