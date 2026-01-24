"""
Generate Training Data from Puzzle FENs using Stockfish

Reads puzzle FENs and generates:
- Position evaluation in pawns
- Top 5 moves with score deltas in centipawns

Usage:
    python generate_puzzle_training_data.py training_puzzle_fens.csv training_puzzle_data.csv --threads 8
    python generate_puzzle_training_data.py test_puzzle_fens.csv test_puzzle_data.csv --threads 8

Input CSV format (fen only):
    fen
    r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP1bgPP/7K w - - 0 1
    ...

Output CSV format (matches training data format):
    fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5

Where:
- eval: position evaluation in pawns (positive = white advantage)
- moveN: UCI format move
- scoreN: score delta in centipawns (relative to best move, so move1 score is always 0)
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


# Default Stockfish path
DEFAULT_STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def worker_init(stockfish_path: str, depth: int, num_moves: int):
    """Initialize worker with persistent Stockfish engine."""
    global WORKER_ENGINE, WORKER_DEPTH, WORKER_NUM_MOVES
    WORKER_ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        WORKER_ENGINE.configure({"Threads": 1, "Hash": 16})
    except chess.engine.EngineError:
        pass
    WORKER_DEPTH = depth
    WORKER_NUM_MOVES = num_moves


def worker_cleanup():
    """Cleanup worker engine."""
    global WORKER_ENGINE
    if WORKER_ENGINE:
        WORKER_ENGINE.quit()


def score_to_centipawns(score: chess.engine.Score, is_white_turn: bool) -> int:
    """
    Convert Stockfish score to centipawns from current player's perspective.

    Args:
        score: Stockfish score object (relative to current player)
        is_white_turn: Whether it's white's turn

    Returns:
        Score in centipawns (positive = good for current player)
    """
    if score.is_mate():
        mate_in = score.mate()
        # Mate in N: use large value, decreasing with distance
        if mate_in > 0:
            return 10000 - abs(mate_in)
        else:
            return -10000 + abs(mate_in)
    else:
        return score.score()


def score_to_pawns_white_perspective(score: chess.engine.Score, is_white_turn: bool) -> float:
    """
    Convert Stockfish score to pawns from white's perspective.

    Args:
        score: Stockfish score object (relative to current player)
        is_white_turn: Whether it's white's turn

    Returns:
        Evaluation in pawns (positive = white advantage)
    """
    cp = score_to_centipawns(score, is_white_turn)

    # Convert to white's perspective
    if not is_white_turn:
        cp = -cp

    # Convert to pawns and clamp
    pawns = cp / 100.0
    return max(-100.0, min(100.0, pawns))


def worker_analyze(fen: str) -> Tuple[str, float, List[Tuple[str, int]]]:
    """
    Analyze a position and return eval + top moves with score deltas.

    Returns:
        Tuple of (fen, eval_pawns, [(move_uci, score_delta_cp), ...])
        Score deltas are relative to the best move (best move has delta 0)
    """
    global WORKER_ENGINE, WORKER_DEPTH, WORKER_NUM_MOVES

    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return (fen, 0.0, [])
    except ValueError:
        return (fen, 0.0, [])

    try:
        # Run multi-PV analysis
        analysis = WORKER_ENGINE.analyse(
            board,
            chess.engine.Limit(depth=WORKER_DEPTH),
            multipv=WORKER_NUM_MOVES
        )

        if not analysis:
            return (fen, 0.0, [])

        is_white_turn = board.turn == chess.WHITE

        # Get position eval from best move (first PV)
        best_score = analysis[0]["score"].relative
        eval_pawns = score_to_pawns_white_perspective(best_score, is_white_turn)
        best_cp = score_to_centipawns(best_score, is_white_turn)

        # Collect moves with score deltas
        moves = []
        for info in analysis:
            if "pv" in info and len(info["pv"]) > 0:
                move = info["pv"][0]
                score = info["score"].relative
                cp = score_to_centipawns(score, is_white_turn)

                # Score delta: how much worse than best move (0 for best, negative for others)
                score_delta = cp - best_cp
                moves.append((move.uci(), score_delta))

        return (fen, eval_pawns, moves)

    except Exception as e:
        print(f"Error analyzing {fen}: {e}", file=sys.stderr)
        return (fen, 0.0, [])


def load_fens(input_file: str) -> List[str]:
    """Load FENs from CSV file (format: fen)."""
    fens = []
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Check if first row is header
        if header and len(header) >= 1:
            first_field = header[0].strip().lower()
            if first_field not in ('fen', 'position', 'board'):
                # First row is data, not header
                fens.append(header[0].strip())

        for row in reader:
            if row and len(row) >= 1:
                fens.append(row[0].strip())

    return fens


def process_positions(
    fens: List[str],
    stockfish_path: str,
    num_threads: int,
    depth: int,
    num_moves: int,
    output_file: str,
    report_interval: int = 1000
) -> int:
    """
    Process FENs in parallel with progress reporting and incremental saving.
    """
    total = len(fens)
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
        for i, result in enumerate(pool.imap_unordered(worker_analyze, fens), 1):
            pending_results.append(result)

            if i % report_interval == 0 or i == total:
                # Write pending results to file
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for fen, eval_pawns, moves in pending_results:
                        row = [fen, f"{eval_pawns:.2f}"]
                        for j in range(num_moves):
                            if j < len(moves):
                                move_uci, score_delta = moves[j]
                                row.extend([move_uci, score_delta])
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
                      f"ETA: {eta_seconds/60:.1f} min", flush=True)

    return total


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data from puzzle FENs using Stockfish',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_puzzle_training_data.py training_puzzle_fens.csv training_puzzle_data.csv -t 8
    python generate_puzzle_training_data.py test_puzzle_fens.csv test_puzzle_data.csv -t 8 -d 20
        """
    )
    parser.add_argument('input_file', help='Input CSV file with puzzle FENs')
    parser.add_argument('output_file', help='Output CSV file for training data')
    parser.add_argument('--threads', '-t', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('--depth', '-d', type=int, default=16,
                        help='Stockfish search depth (default: 16)')
    parser.add_argument('--moves', '-m', type=int, default=5,
                        help='Number of top moves to generate (default: 5)')
    parser.add_argument('--stockfish', '-s', type=str, default=DEFAULT_STOCKFISH_PATH,
                        help=f'Path to Stockfish executable (default: {DEFAULT_STOCKFISH_PATH})')
    parser.add_argument('--report-interval', '-r', type=int, default=1000,
                        help='Print progress every N positions (default: 1000)')

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
    print("Puzzle Training Data Generator")
    print("=" * 70)
    print(f"Input file:  {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Threads:     {args.threads}")
    print(f"Depth:       {args.depth}")
    print(f"Top moves:   {args.moves}")
    print(f"Stockfish:   {args.stockfish}")
    print("=" * 70)

    # Load FENs
    print("Loading puzzle FENs...", end=' ', flush=True)
    fens = load_fens(args.input_file)
    print(f"loaded {len(fens):,} positions")

    if len(fens) == 0:
        print("Error: No positions found in input file", file=sys.stderr)
        sys.exit(1)

    # Process positions
    print(f"\nAnalyzing positions with Stockfish (depth {args.depth})...")
    start_time = time.time()

    positions_processed = process_positions(
        fens,
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
