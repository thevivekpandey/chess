#!/usr/bin/env python3
"""
Analyze weak spots in chess engine games by finding positions where
Stockfish evaluation drops by more than 2 pawns after the engine's move.

For each weak position, extract:
- FEN of the position
- Model evaluation
- Top 7 moves from model policy
- The actual move played by the engine

Output is saved to a CSV file.
"""

import argparse
import csv
import os
from typing import List, Tuple, Optional

import chess
import chess.engine
import chess.pgn

from chess_engine import ChessEvaluator, get_top_moves_from_policy


def cp_to_pawns(cp_score: int) -> float:
    """Convert centipawn score to pawns."""
    return cp_score / 100.0


def get_stockfish_eval(engine: chess.engine.SimpleEngine, board: chess.Board,
                       depth: int = 20) -> Optional[float]:
    """
    Get Stockfish evaluation of a position in pawns.

    Returns:
        Evaluation in pawns from white's perspective, or None if mate score
    """
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()

        if score.is_mate():
            # Return a large value for mate scores
            mate_in = score.mate()
            if mate_in > 0:
                return 100.0  # White is mating
            else:
                return -100.0  # Black is mating
        else:
            return cp_to_pawns(score.score())
    except Exception as e:
        print(f"Warning: Stockfish analysis failed: {e}")
        return None


def analyze_pgn_file(
    pgn_path: str,
    model_path: str,
    stockfish_path: str,
    output_csv: str,
    eval_threshold: float = 2.0,
    stockfish_depth: int = 20,
    device: str = "auto",
    max_games: Optional[int] = None,
):
    """
    Analyze a PGN file to find weak moves.

    Args:
        pgn_path: Path to PGN file
        model_path: Path to neural network model checkpoint
        stockfish_path: Path to Stockfish binary
        output_csv: Output CSV file path
        eval_threshold: Minimum eval drop in pawns to consider a move weak (default: 2.0)
        stockfish_depth: Stockfish analysis depth (default: 20)
        device: Device for neural network ('auto', 'cpu', 'cuda', 'mps')
        max_games: Maximum number of games to analyze (None = all)
    """
    print(f"Loading neural network model from {model_path}...")
    evaluator = ChessEvaluator(model_path, device=device)

    print(f"Starting Stockfish from {stockfish_path}...")
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    print(f"Opening PGN file: {pgn_path}")
    weak_positions: List[Tuple[str, str, float, float, List[Tuple[str, float]]]] = []

    try:
        with open(pgn_path, 'r') as pgn_file:
            game_num = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                game_num += 1
                if max_games is not None and game_num > max_games:
                    break

                print(f"\nAnalyzing game {game_num}...")
                board = game.board()
                node = game
                move_num = 0

                while node.variations:
                    node = node.variation(0)
                    move = node.move
                    move_num += 1

                    # Get position before the move
                    fen_before = board.fen()

                    # Evaluate position before move
                    eval_before = get_stockfish_eval(stockfish, board, stockfish_depth)
                    if eval_before is None:
                        board.push(move)
                        continue

                    # Make the move
                    board.push(move)

                    # Evaluate position after move
                    eval_after = get_stockfish_eval(stockfish, board, stockfish_depth)
                    if eval_after is None:
                        continue

                    # Calculate eval change from the moving side's perspective
                    # If white just moved, we want to see if white's advantage decreased
                    # eval_before is from white's perspective before white's move
                    # eval_after is from white's perspective after white's move
                    moving_color = not board.turn  # board.turn is flipped after push

                    if moving_color == chess.WHITE:
                        # White moved: check if eval dropped (from white's perspective)
                        eval_drop = eval_before - eval_after
                    else:
                        # Black moved: check if eval increased (from white's perspective)
                        # which means black's position got worse
                        eval_drop = eval_after - eval_before

                    # If eval dropped by more than threshold, this is a weak move
                    if eval_drop > eval_threshold:
                        # Create a temporary board to convert moves to SAN
                        temp_board = chess.Board(fen_before)
                        move_san = temp_board.san(move)

                        # Calculate full move number (move_num is ply count starting from 1)
                        full_move_num = (move_num + 1) // 2

                        print(f"  Move {full_move_num} ({move_san}): "
                              f"eval drop = {eval_drop:.2f} pawns "
                              f"({eval_before:+.2f} -> {eval_after:+.2f})")

                        # Get model evaluation and policy for the position BEFORE the move
                        result = evaluator.evaluate_with_policy(fen_before, top_k=7)
                        model_eval = result['eval_pawns']
                        top_moves = result['top_moves']

                        # Convert top moves from UCI to SAN
                        top_moves_san = []
                        for move_uci, prob in top_moves:
                            try:
                                move_obj = chess.Move.from_uci(move_uci)
                                if move_obj in temp_board.legal_moves:
                                    move_san_str = temp_board.san(move_obj)
                                    top_moves_san.append((move_san_str, prob))
                                else:
                                    # Keep UCI if conversion fails
                                    top_moves_san.append((move_uci, prob))
                            except:
                                # Keep UCI if conversion fails
                                top_moves_san.append((move_uci, prob))

                        weak_positions.append((
                            fen_before,
                            full_move_num,
                            move_san,
                            model_eval,
                            eval_drop,
                            top_moves_san
                        ))

        print(f"\nFound {len(weak_positions)} weak positions across {game_num} games")

        # Write to CSV
        print(f"\nWriting results to {output_csv}...")
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Header
            header = ['fen', 'move_number', 'actual_move', 'model_eval', 'eval_drop']
            for i in range(1, 8):
                header.extend([f'top{i}_move', f'top{i}_prob'])
            writer.writerow(header)

            # Data rows
            for fen, move_number, actual_move, model_eval, eval_drop, top_moves in weak_positions:
                row = [fen, move_number, actual_move, f"{model_eval:.4f}", f"{eval_drop:.4f}"]

                # Add top 7 moves (pad with empty if less than 7)
                for i in range(7):
                    if i < len(top_moves):
                        move_san, prob = top_moves[i]
                        row.extend([move_san, f"{prob:.6f}"])
                    else:
                        row.extend(['', ''])

                writer.writerow(row)

        print(f"Successfully wrote {len(weak_positions)} positions to {output_csv}")

    finally:
        stockfish.quit()
        print("\nAnalysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PGN games to find weak engine moves"
    )
    parser.add_argument(
        "pgn_file",
        help="Path to PGN file to analyze"
    )
    parser.add_argument(
        "model",
        help="Path to neural network model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--stockfish",
        default="/usr/games/stockfish",
        help="Path to Stockfish binary (default: /usr/games/stockfish)"
    )
    parser.add_argument(
        "--output",
        default="weak_spots.csv",
        help="Output CSV file (default: weak_spots.csv)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Minimum eval drop in pawns to consider weak (default: 2.0)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Stockfish analysis depth (default: 20)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for neural network (default: auto)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to analyze (default: all)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.pgn_file):
        print(f"Error: PGN file not found: {args.pgn_file}")
        return 1

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1

    if not os.path.exists(args.stockfish):
        print(f"Error: Stockfish binary not found: {args.stockfish}")
        print("Please specify the correct path with --stockfish")
        return 1

    print("=" * 80)
    print("Weak Spot Analyzer")
    print("=" * 80)
    print(f"PGN file: {args.pgn_file}")
    print(f"Model: {args.model}")
    print(f"Stockfish: {args.stockfish}")
    print(f"Output CSV: {args.output}")
    print(f"Eval threshold: {args.threshold} pawns")
    print(f"Stockfish depth: {args.depth}")
    print(f"Device: {args.device}")
    if args.max_games:
        print(f"Max games: {args.max_games}")
    print("=" * 80)

    analyze_pgn_file(
        pgn_path=args.pgn_file,
        model_path=args.model,
        stockfish_path=args.stockfish,
        output_csv=args.output,
        eval_threshold=args.threshold,
        stockfish_depth=args.depth,
        device=args.device,
        max_games=args.max_games,
    )

    return 0


if __name__ == "__main__":
    exit(main())
