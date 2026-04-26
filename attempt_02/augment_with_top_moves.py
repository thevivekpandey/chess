#!/usr/bin/env python3
"""
Augment chess position CSV files with top 5 Stockfish moves and score deltas.

Input format:  fen,eval
Output format: fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5

Moves are in UCI notation. Scores are deltas from the base eval (both from white's
perspective): score_i = move_eval_white - base_eval_white.

Usage:
    python augment_with_top_moves.py <input_csv> <output_csv>
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time

import chess
import chess.engine

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
DEPTH = 16
NUM_MOVES = 5
BATCH_SIZE = 500  # rows per batch sent to workers
MATE_SCORE = 100.0
PROGRESS_INTERVAL = 1000


def score_to_white_pov(score: chess.engine.Score, turn: chess.Color) -> float:
    """Convert a Stockfish score (from side-to-move perspective) to white's perspective.

    Stockfish returns PovScore relative to the side to move. We convert to a float
    from white's point of view, matching the CSV convention (positive = white better).

    Mate scores are clamped to +/- MATE_SCORE.
    """
    # score is from the perspective of the side to move
    cp = score.relative.score(mate_score=10000)
    if cp is None:
        # Should not happen with mate_score fallback, but just in case
        return 0.0
    value = cp / 100.0
    # Clamp mate scores
    if value >= 99.0:
        value = MATE_SCORE
    elif value <= -99.0:
        value = -MATE_SCORE
    # Convert from side-to-move perspective to white's perspective
    if turn == chess.BLACK:
        value = -value
    return value


def analyze_position(fen: str, base_eval: float, engine: chess.engine.SimpleEngine):
    """Analyze a single position and return the top moves with score deltas.

    Returns a list of (uci_move, score_delta) tuples, padded to NUM_MOVES with ('', '').
    """
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    legal_move_count = len(legal_moves)

    if legal_move_count == 0:
        return [("", "")] * NUM_MOVES

    multi_pv = min(NUM_MOVES, legal_move_count)

    try:
        results = engine.analyse(
            board,
            chess.engine.Limit(depth=DEPTH),
            multipv=multi_pv,
        )
    except chess.engine.EngineTerminatedError:
        # Engine crashed; return empty
        return [("", "")] * NUM_MOVES

    moves = []
    for info in results:
        move = info["pv"][0].uci()
        move_eval_white = score_to_white_pov(info["score"], board.turn)
        delta = round(move_eval_white - base_eval, 2)
        moves.append((move, delta))

    # Pad with empty entries if fewer than NUM_MOVES
    while len(moves) < NUM_MOVES:
        moves.append(("", ""))

    return moves


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue):
    """Worker process: owns a Stockfish engine, processes batches from the queue."""
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    # Set hash and threads for each worker
    engine.configure({"Hash": 64, "Threads": 1})

    while True:
        item = task_queue.get()
        if item is None:
            # Poison pill - shut down
            engine.quit()
            break

        batch_index, rows = item
        batch_results = []
        for row_index, fen, base_eval_str in rows:
            base_eval = float(base_eval_str)
            moves = analyze_position(fen, base_eval, engine)
            # Build the output row
            out = [fen, base_eval_str]
            for uci_move, delta in moves:
                out.append(str(uci_move))
                out.append(str(delta) if delta != "" else "")
            batch_results.append((row_index, out))

        result_queue.put((batch_index, batch_results))


def count_lines(filepath: str) -> int:
    """Count lines in a file efficiently."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Augment chess CSV with top Stockfish moves"
    )
    parser.add_argument("input_csv", help="Input CSV file (fen,eval)")
    parser.add_argument("output_csv", help="Output CSV file")
    args = parser.parse_args()

    input_path = args.input_csv
    output_path = args.output_csv

    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    # Count total data rows (excluding header)
    total_rows = count_lines(input_path) - 1
    print(f"Total data rows in input: {total_rows}")

    # Check for resume: count existing output lines
    skip_rows = 0
    if os.path.exists(output_path):
        existing_lines = count_lines(output_path)
        if existing_lines > 1:
            skip_rows = existing_lines - 1  # subtract header
            print(f"Resuming: output has {existing_lines} lines ({skip_rows} data rows), skipping those.")
        elif existing_lines == 1:
            # Only header exists, no skipping needed but file exists
            skip_rows = 0
    remaining_rows = total_rows - skip_rows
    if remaining_rows <= 0:
        print("All rows already processed. Nothing to do.")
        return

    print(f"Rows to process: {remaining_rows}")

    num_workers = mp.cpu_count()
    print(f"Using {num_workers} worker processes")

    # Queues
    task_queue = mp.Queue(maxsize=num_workers * 4)
    result_queue = mp.Queue()

    # Start workers
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=worker_process, args=(task_queue, result_queue))
        p.daemon = True
        p.start()
        workers.append(p)

    # Open output file
    write_mode = "a" if skip_rows > 0 else "w"
    outfile = open(output_path, write_mode, newline="")
    writer = csv.writer(outfile)

    # Write header if starting fresh
    if skip_rows == 0:
        header = ["fen", "eval"]
        for i in range(1, NUM_MOVES + 1):
            header.extend([f"move{i}", f"score{i}"])
        writer.writerow(header)
        outfile.flush()

    # Producer: read input and send batches
    processed = 0
    batches_sent = 0
    batches_received = 0
    pending_results = {}  # batch_index -> results, for ordered writing
    next_write_batch = 0
    start_time = time.time()
    last_progress_time = start_time

    def write_ready_batches():
        """Write batches in order as they become available."""
        nonlocal next_write_batch, processed, last_progress_time
        while next_write_batch in pending_results:
            batch_results = pending_results.pop(next_write_batch)
            # Sort by row_index within batch to maintain order
            batch_results.sort(key=lambda x: x[0])
            for _, out_row in batch_results:
                writer.writerow(out_row)
            processed += len(batch_results)
            outfile.flush()
            next_write_batch += 1

            now = time.time()
            if processed % PROGRESS_INTERVAL < BATCH_SIZE or now - last_progress_time > 10:
                elapsed = now - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (remaining_rows - processed) / rate if rate > 0 else 0
                eta_min = eta_seconds / 60
                print(
                    f"  Processed {processed}/{remaining_rows} "
                    f"({100*processed/remaining_rows:.1f}%) "
                    f"| {rate:.1f} pos/s "
                    f"| ETA: {eta_min:.1f} min"
                )
                last_progress_time = now

    # Read input and dispatch batches
    with open(input_path, "r", newline="") as infile:
        reader = csv.reader(infile)
        next(reader)  # skip header

        # Skip already-processed rows
        for _ in range(skip_rows):
            next(reader)

        batch = []
        global_row_idx = skip_rows

        for row in reader:
            if len(row) < 2:
                continue
            fen, eval_str = row[0], row[1]
            batch.append((global_row_idx, fen, eval_str))
            global_row_idx += 1

            if len(batch) >= BATCH_SIZE:
                # Before sending, drain any available results
                while not result_queue.empty():
                    bi, br = result_queue.get_nowait()
                    pending_results[bi] = br
                    batches_received += 1
                write_ready_batches()

                task_queue.put((batches_sent, batch))
                batches_sent += 1
                batch = []

        # Send final partial batch
        if batch:
            task_queue.put((batches_sent, batch))
            batches_sent += 1

    # Send poison pills to stop workers
    for _ in range(num_workers):
        task_queue.put(None)

    # Collect remaining results
    while batches_received < batches_sent:
        bi, br = result_queue.get()
        pending_results[bi] = br
        batches_received += 1
        write_ready_batches()

    # Final flush
    write_ready_batches()
    outfile.close()

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=10)

    elapsed = time.time() - start_time
    print(f"\nDone! Processed {processed} positions in {elapsed:.1f}s ({processed/elapsed:.1f} pos/s)")


if __name__ == "__main__":
    main()
