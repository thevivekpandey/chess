"""
Script to extract FEN positions and Stockfish evaluations from PGN file.
Creates training_data_base.csv with two columns: fen, eval
"""

import chess
import re
import csv
import sys


def parse_eval(eval_str: str) -> float:
    """
    Parse evaluation string to pawn units.
    Handles mate scores like #1, #-2, etc.

    Args:
        eval_str: Evaluation string (e.g., "0.19", "#7", "#-3")

    Returns:
        Evaluation in pawn units (mate = +/-100)
    """
    eval_str = eval_str.strip()

    if eval_str.startswith('#'):
        # Mate score
        mate_num = eval_str[1:]
        if mate_num.startswith('-'):
            return -100.0  # Black is mating
        else:
            return 100.0   # White is mating
    else:
        # Regular evaluation in pawn units (keep as-is)
        try:
            return float(eval_str)
        except ValueError:
            return None


def extract_moves_and_evals(line: str):
    """
    Extract moves and their evaluations from a game line.

    Args:
        line: A line containing a chess game with evaluations

    Yields:
        Tuples of (move_san, eval_cp)
    """
    # Pattern to match evaluations like { [%eval 0.19] }
    eval_pattern = r'\{\s*\[%eval\s+([^\]]+)\]\s*\}'

    # Remove PGN headers like [Black "player"] but NOT eval annotations
    # Headers are at the start of line or after newline, contain quotes
    line = re.sub(r'^\s*\[[A-Za-z]+\s+"[^"]*"\]\s*', '', line)

    # Find all moves with their evaluations
    # Split by evaluation markers
    parts = re.split(eval_pattern, line)

    # parts alternates between: text_with_move, eval, text_with_move, eval, ...
    moves = []
    evals = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # This is text containing moves
            # Extract SAN moves (skip move numbers and annotations like ?!, ??, etc.)
            move_pattern = r'([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O-O|O-O)(?:[?!]*)'
            found_moves = re.findall(move_pattern, part)
            moves.extend(found_moves)
        else:
            # This is an evaluation
            evals.append(part)

    # Pair moves with evals (each move has an eval)
    for move, eval_str in zip(moves, evals):
        eval_cp = parse_eval(eval_str)
        if eval_cp is not None:
            yield move, eval_cp


def process_game(line: str):
    """
    Process a single game line and extract FEN positions with evaluations.

    Args:
        line: A line containing a chess game with evaluations

    Yields:
        Tuples of (fen, eval_cp)
    """
    board = chess.Board()

    for move_san, eval_cp in extract_moves_and_evals(line):
        try:
            # Parse and make the move
            move = board.parse_san(move_san)
            board.push(move)

            # Get FEN of position AFTER the move (this is what the eval refers to)
            fen = board.fen()

            yield fen, eval_cp

        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
            # Skip invalid moves but continue processing
            continue


def main():
    input_file = 'games_with_evals.pgn'
    output_file = 'training_data_base.csv'

    total_positions = 0
    total_games = 0
    errors = 0

    print(f"Processing {input_file}...")

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['fen', 'eval'])  # Header

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line or line.startswith('['):
                # Skip empty lines and header-only lines
                continue

            game_positions = 0
            try:
                for fen, eval_cp in process_game(line):
                    writer.writerow([fen, eval_cp])
                    game_positions += 1
                    total_positions += 1

                if game_positions > 0:
                    total_games += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error on line {line_num}: {e}")

            # Progress report every 10000 games
            if total_games % 10000 == 0 and total_games > 0:
                print(f"Processed {total_games} games, {total_positions} positions...")

    print(f"\nDone!")
    print(f"Total games processed: {total_games}")
    print(f"Total positions extracted: {total_positions}")
    print(f"Errors: {errors}")
    print(f"Output saved to: {output_file}")


if __name__ == '__main__':
    main()
