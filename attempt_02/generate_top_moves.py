"""
Generate Top 5 Stockfish Moves for Chess Positions

Usage:
    python generate_top_moves.py input.csv output.csv --threads 8 --depth 16

Input CSV format:
    fen
    rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    ...

Output CSV format:
    fen,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5
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


def analyze_position(args: Tuple[str, str, int, int]) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Analyze a single position with Stockfish multi-PV.

    Args:
        args: Tuple of (fen, stockfish_path, depth, num_moves)

    Returns:
        Tuple of (fen, [(move_uci, score_cp), ...])
    """
    fen, stockfish_path, depth, num_moves = args

    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return (fen, [])
    except ValueError:
        return (fen, [])

    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        # Run multi-PV analysis
        analysis = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=num_moves
        )

        moves = []
        for info in analysis:
            if "pv" in info and len(info["pv"]) > 0:
                move = info["pv"][0]
                score = info["score"].relative

                # Convert score to centipawns
                if score.is_mate():
                    mate_in = score.mate()
                    # Use large values for mate scores
                    cp = 10000 - abs(mate_in) if mate_in > 0 else -10000 + abs(mate_in)
                else:
                    cp = score.score()

                moves.append((move.uci(), cp))

        return (fen, moves)

    except Exception as e:
        print(f"Error analyzing {fen}: {e}", file=sys.stderr)
        return (fen, [])

    finally:
        if engine:
            engine.quit()


def worker_init(stockfish_path: str, depth: int, num_moves: int):
    """Initialize worker with shared parameters."""
    global WORKER_STOCKFISH_PATH, WORKER_DEPTH, WORKER_NUM_MOVES
    WORKER_STOCKFISH_PATH = stockfish_path
    WORKER_DEPTH = depth
    WORKER_NUM_MOVES = num_moves


def worker_analyze(fen: str) -> Tuple[str, List[Tuple[str, int]]]:
    """Worker function that uses global parameters."""
    return analyze_position((fen, WORKER_STOCKFISH_PATH, WORKER_DEPTH, WORKER_NUM_MOVES))


def process_batch(
    fens: List[str],
    stockfish_path: str,
    num_threads: int,
    depth: int,
    num_moves: int
) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """
    Process a batch of FENs in parallel.

    Args:
        fens: List of FEN strings
        stockfish_path: Path to Stockfish executable
        num_threads: Number of parallel workers
        depth: Search depth
        num_moves: Number of top moves to return

    Returns:
        List of (fen, moves) tuples
    """
    with mp.Pool(
        processes=num_threads,
        initializer=worker_init,
        initargs=(stockfish_path, depth, num_moves)
    ) as pool:
        results = pool.map(worker_analyze, fens)

    return results


def load_fens(input_file: str) -> List[str]:
    """Load FENs from CSV file."""
    fens = []
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Check if first row looks like a header
        if header:
            first_field = header[0].strip().lower()
            if first_field not in ('fen', 'position', 'board'):
                # First row is data, not header
                fens.append(header[0].strip())

        for row in reader:
            if row:
                fens.append(row[0].strip())

    return fens


def write_results(
    output_file: str,
    results: List[Tuple[str, List[Tuple[str, int]]]],
    num_moves: int
):
    """Write results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['fen']
        for i in range(1, num_moves + 1):
            header.extend([f'move{i}', f'score{i}'])
        writer.writerow(header)

        # Data rows
        for fen, moves in results:
            row = [fen]
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
    python generate_top_moves.py positions.csv output.csv --threads 8
    python generate_top_moves.py data.csv moves.csv --threads 16 --depth 20 --moves 10
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
    parser.add_argument('--batch-size', '-b', type=int, default=10000,
                        help='Batch size for progress reporting (default: 10000)')

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

    # Load FENs
    print("Loading FENs...", end=' ', flush=True)
    fens = load_fens(args.input_file)
    print(f"loaded {len(fens):,} positions")

    if len(fens) == 0:
        print("Error: No positions found in input file", file=sys.stderr)
        sys.exit(1)

    # Process in batches for progress reporting
    all_results = []
    batch_size = args.batch_size
    start_time = time.time()

    print(f"\nProcessing positions...")

    for batch_start in range(0, len(fens), batch_size):
        batch_end = min(batch_start + batch_size, len(fens))
        batch_fens = fens[batch_start:batch_end]

        batch_results = process_batch(
            batch_fens,
            args.stockfish,
            args.threads,
            args.depth,
            args.moves
        )
        all_results.extend(batch_results)

        # Progress report
        elapsed = time.time() - start_time
        positions_done = len(all_results)
        positions_per_sec = positions_done / elapsed if elapsed > 0 else 0
        eta_seconds = (len(fens) - positions_done) / positions_per_sec if positions_per_sec > 0 else 0

        print(f"  Progress: {positions_done:,}/{len(fens):,} "
              f"({positions_done/len(fens)*100:.1f}%) | "
              f"{positions_per_sec:.1f} pos/sec | "
              f"ETA: {eta_seconds/60:.1f} min", flush=True)

    # Write results
    print(f"\nWriting results to {args.output_file}...", end=' ', flush=True)
    write_results(args.output_file, all_results, args.moves)
    print("done")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Positions processed: {len(all_results):,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {len(all_results)/total_time:.1f} positions/second")
    print(f"Output saved to: {args.output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
