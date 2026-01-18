"""
Generate Top 5 Stockfish Moves for Chess Positions

Usage:
    python generate_top_moves.py input.csv output.csv --threads 8 --depth 16

Input CSV format (fen,eval):
    fen,eval
    rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,0.2
    ...

Output CSV format:
    fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5

The eval from the input file is preserved for joint value/policy head training.
"""

import argparse
import csv
import chess
import chess.engine
import multiprocessing as mp
from typing import List, Tuple, Optional
import time
import os
import sys


# Default Stockfish path (adjust as needed)
DEFAULT_STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def worker_init(stockfish_path: str, depth: int, num_moves: int):
    """Initialize worker with persistent Stockfish engine."""
    global WORKER_ENGINE, WORKER_DEPTH, WORKER_NUM_MOVES
    WORKER_ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    # Optional: configure engine for speed
    try:
        WORKER_ENGINE.configure({"Threads": 1, "Hash": 16})
    except chess.engine.EngineError:
        pass  # Some options might not be available
    WORKER_DEPTH = depth
    WORKER_NUM_MOVES = num_moves


def worker_analyze(fen_eval: Tuple[str, str]) -> Tuple[str, str, List[Tuple[str, int]]]:
    """Worker function that reuses persistent Stockfish engine."""
    global WORKER_ENGINE, WORKER_DEPTH, WORKER_NUM_MOVES
    fen, eval_score = fen_eval

    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return (fen, eval_score, [])
    except ValueError:
        return (fen, eval_score, [])

    try:
        # Run multi-PV analysis using persistent engine
        analysis = WORKER_ENGINE.analyse(
            board,
            chess.engine.Limit(depth=WORKER_DEPTH),
            multipv=WORKER_NUM_MOVES
        )

        moves = []
        for info in analysis:
            if "pv" in info and len(info["pv"]) > 0:
                move = info["pv"][0]
                score = info["score"].relative

                # Convert score to centipawns
                if score.is_mate():
                    mate_in = score.mate()
                    cp = 10000 - abs(mate_in) if mate_in > 0 else -10000 + abs(mate_in)
                else:
                    cp = score.score()

                moves.append((move.uci(), cp))

        return (fen, eval_score, moves)

    except Exception as e:
        print(f"Error analyzing {fen}: {e}", file=sys.stderr)
        return (fen, eval_score, [])


def process_positions(
    fen_evals: List[Tuple[str, str]],
    stockfish_path: str,
    num_threads: int,
    depth: int,
    num_moves: int,
    output_file: str,
    report_interval: int = 10
) -> int:
    """
    Process FENs in parallel with progress reporting and incremental saving.

    Args:
        fen_evals: List of (fen, eval) tuples
        stockfish_path: Path to Stockfish executable
        num_threads: Number of parallel workers
        depth: Search depth
        num_moves: Number of top moves to return
        output_file: Path to output CSV file (written incrementally)
        report_interval: How often to print progress and save (every N positions)

    Returns:
        Number of positions processed
    """
    total = len(fen_evals)
    start_time = time.time()
    pending_results = []

    # Write header
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['fen', 'eval']
        for i in range(1, num_moves + 1):
            header.extend([f'move{i}', f'score{i}'])
        writer.writerow(header)

    with mp.Pool(
        processes=num_threads,
        initializer=worker_init,
        initargs=(stockfish_path, depth, num_moves)
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_analyze, fen_evals), 1):
            pending_results.append(result)

            if i % report_interval == 0 or i == total:
                # Write pending results to file
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for fen, eval_score, moves in pending_results:
                        row = [fen, eval_score]
                        for j in range(num_moves):
                            if j < len(moves):
                                move_uci, score = moves[j]
                                row.extend([move_uci, score])
                            else:
                                row.extend(['', ''])
                        writer.writerow(row)
                pending_results = []

                # Progress report
                elapsed = time.time() - start_time
                positions_per_sec = i / elapsed if elapsed > 0 else 0
                eta_seconds = (total - i) / positions_per_sec if positions_per_sec > 0 else 0
                print(f"  Progress: {i:,}/{total:,} "
                      f"({i/total*100:.1f}%) | "
                      f"{positions_per_sec:.1f} pos/sec | "
                      f"ETA: {eta_seconds/60:.1f} min [saved]", flush=True)

    return total


def load_fen_evals(input_file: str) -> List[Tuple[str, str]]:
    """Load FENs and evals from CSV file (format: fen,eval)."""
    fen_evals = []
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Check if first row looks like a header
        if header and len(header) >= 2:
            first_field = header[0].strip().lower()
            if first_field not in ('fen', 'position', 'board'):
                # First row is data, not header
                fen_evals.append((header[0].strip(), header[1].strip()))

        for row in reader:
            if row and len(row) >= 2:
                fen_evals.append((row[0].strip(), row[1].strip()))

    return fen_evals


def write_results(
    output_file: str,
    results: List[Tuple[str, str, List[Tuple[str, int]]]],
    num_moves: int
):
    """Write results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['fen', 'eval']
        for i in range(1, num_moves + 1):
            header.extend([f'move{i}', f'score{i}'])
        writer.writerow(header)

        # Data rows
        for fen, eval_score, moves in results:
            row = [fen, eval_score]
            for i in range(num_moves):
                if i < len(moves):
                    move_uci, score = moves[i]
                    row.extend([move_uci, score])
                else:
                    row.extend(['', ''])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='Generate top N Stockfish moves for chess positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_top_moves.py input.csv output.csv --threads 8
    python generate_top_moves.py data.csv moves.csv --threads 16 --depth 20 --moves 10
    python generate_top_moves.py data.csv out.csv -t 8 -r 100  # Report every 100 positions
        """
    )
    parser.add_argument('input_file', help='Input CSV file with FENs')
    parser.add_argument('output_file', help='Output CSV file for results')
    parser.add_argument('--threads', '-t', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('--depth', '-d', type=int, default=16,
                        help='Stockfish search depth (default: 16)')
    parser.add_argument('--moves', '-m', type=int, default=5,
                        help='Number of top moves to generate (default: 5)')
    parser.add_argument('--stockfish', '-s', type=str, default=DEFAULT_STOCKFISH_PATH,
                        help=f'Path to Stockfish executable (default: {DEFAULT_STOCKFISH_PATH})')
    parser.add_argument('--report-interval', '-r', type=int, default=10,
                        help='Print progress every N positions (default: 10)')

    args = parser.parse_args()

    # Validate Stockfish path
    if not os.path.exists(args.stockfish):
        print(f"Error: Stockfish not found at {args.stockfish}", file=sys.stderr)
        print("Use --stockfish to specify the correct path", file=sys.stderr)
        sys.exit(1)

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Stockfish Multi-PV Analysis")
    print("=" * 70)
    print(f"Input file:  {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Threads:     {args.threads}")
    print(f"Depth:       {args.depth}")
    print(f"Top moves:   {args.moves}")
    print(f"Stockfish:   {args.stockfish}")
    print("=" * 70)

    # Load FENs and evals
    print("Loading positions...", end=' ', flush=True)
    fen_evals = load_fen_evals(args.input_file)
    print(f"loaded {len(fen_evals):,} positions")

    if len(fen_evals) == 0:
        print("Error: No positions found in input file", file=sys.stderr)
        sys.exit(1)

    # Process positions with progress reporting and incremental saving
    print(f"\nProcessing positions (saving every {args.report_interval} positions)...")
    start_time = time.time()

    positions_processed = process_positions(
        fen_evals,
        args.stockfish,
        args.threads,
        args.depth,
        args.moves,
        args.output_file,
        report_interval=args.report_interval
    )

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Positions processed: {positions_processed:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {positions_processed/total_time:.1f} positions/second")
    print(f"Output saved to: {args.output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
