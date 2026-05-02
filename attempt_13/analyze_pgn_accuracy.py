#!/usr/bin/env python3
"""
Analyze PGN games using Stockfish as ground truth.

Provides move quality statistics:
- Top-1 accuracy (does engine's move match Stockfish?)
- Average centipawn loss (ACPL)
- Move quality distribution (best/good/mistake/blunder)
- Per-game statistics

Usage:
  python analyze_pgn_accuracy.py neural_vs_stockfish_mcts.pgn \
      --stockfish /opt/homebrew/bin/stockfish \
      --depth 20 \
      --engine-name "NeuralEngine"
"""

import argparse
import chess
import chess.engine
import chess.pgn
import time
import multiprocessing
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

import numpy as np


def cp_to_pawns(cp: int) -> float:
    """Convert centipawns to pawns."""
    return cp / 100.0


def categorize_move(cp_loss: float) -> str:
    """Categorize move quality by centipawn loss."""
    cp_loss_cp = cp_loss * 100  # Convert to centipawns

    if cp_loss_cp < 10:
        return "best"
    elif cp_loss_cp < 25:
        return "excellent"
    elif cp_loss_cp < 50:
        return "good"
    elif cp_loss_cp < 100:
        return "inaccuracy"
    elif cp_loss_cp < 200:
        return "mistake"
    else:
        return "blunder"


def get_stockfish_best_move_and_eval(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int
) -> Tuple[Optional[chess.Move], Optional[float], Optional[bool]]:
    """
    Get Stockfish's best move and evaluation after that move.

    Uses a single analyse() call - the returned score is the evaluation
    after the best move is played (from white's perspective).

    Returns:
        (best_move, eval_after_best_move, is_mate)
        where eval is in pawns from white's perspective, and is_mate is True if it's a mate score
    """
    try:
        # Single analyse call - gives us best move (from PV) and eval after it
        info = engine.analyse(board, chess.engine.Limit(depth=depth))

        # Best move is the first move in the principal variation
        pv = info.get("pv")
        if not pv:
            return None, None, None
        best_move = pv[0]

        # The score from analyse() is the evaluation after the best move
        score = info["score"].white()

        if score.is_mate():
            mate_in = score.mate()
            if mate_in > 0:
                eval_after = 100.0  # White wins
            else:
                eval_after = -100.0  # Black wins
            return best_move, eval_after, True
        else:
            eval_after = cp_to_pawns(score.score())
            return best_move, eval_after, False

    except Exception as e:
        print(f"Warning: Stockfish analysis failed: {e}")
        return None, None, None


def get_eval_after_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move: chess.Move,
    depth: int
) -> Tuple[Optional[float], Optional[bool]]:
    """
    Get evaluation after a specific move.

    Returns:
        (eval, is_mate) where eval is in pawns from white's perspective
    """
    try:
        board_copy = board.copy()
        board_copy.push(move)
        info = engine.analyse(board_copy, chess.engine.Limit(depth=depth))
        score = info["score"].white()

        if score.is_mate():
            mate_in = score.mate()
            if mate_in > 0:
                return 100.0, True
            else:
                return -100.0, True
        else:
            return cp_to_pawns(score.score()), False

    except Exception as e:
        return None, None


def analyze_game(
    game: chess.pgn.Game,
    stockfish: chess.engine.SimpleEngine,
    depth: int,
    engine_name: str,
    verbose: bool = False,
    game_num: int = 0,
    round_num: str = "?",
    opponent: str = "Unknown",
) -> Dict:
    """
    Analyze a single game.

    Returns:
        Dictionary with game statistics, messages, and CSV lines
    """
    # Determine which color the engine played
    white_player = game.headers.get("White", "")
    black_player = game.headers.get("Black", "")

    if engine_name in white_player:
        engine_color = chess.WHITE
        engine_side = "White"
    elif engine_name in black_player:
        engine_color = chess.BLACK
        engine_side = "Black"
    else:
        # Can't determine engine color
        return None

    board = game.board()
    node = game

    game_stats = {
        'engine_color': engine_side,
        'result': game.headers.get("Result", "*"),
        'moves_analyzed': 0,
        'top1_matches': 0,
        'centipawn_losses': [],
        'move_categories': defaultdict(int),
        'weak_moves': [],  # List of (move_num, move_san, cp_loss)
        'console_messages': [],  # Messages to print to console
        'csv_lines': [],  # CSV lines to write to output file
    }

    move_num = 0

    while node.variations:
        node = node.variation(0)
        move = node.move
        move_num += 1

        # Only analyze engine's moves
        if board.turn != engine_color:
            board.push(move)
            continue

        # Skip if game is already over
        if board.is_game_over():
            board.push(move)
            continue

        # Get Stockfish's best move and eval
        sf_best_move, sf_eval_after, sf_is_mate = get_stockfish_best_move_and_eval(
            stockfish, board, depth
        )

        if sf_best_move is None or sf_eval_after is None:
            board.push(move)
            continue

        # Get eval after engine's actual move
        engine_eval_after, engine_is_mate = get_eval_after_move(stockfish, board, move, depth)

        if engine_eval_after is None:
            board.push(move)
            continue

        game_stats['moves_analyzed'] += 1

        # Check if engine's move is checkmate
        board_copy = board.copy()
        board_copy.push(move)
        is_checkmate = board_copy.is_checkmate()

        # Check if engine's move matches Stockfish's best move
        if move == sf_best_move:
            # Perfect match - no centipawn loss
            cp_loss = 0.0
            game_stats['top1_matches'] += 1
        elif is_checkmate:
            # Engine delivered checkmate - can't do better than that!
            cp_loss = 0.0
        elif sf_is_mate and engine_is_mate:
            # Both moves lead to mate - check if they're for the same side
            # If sf_eval is +100 (white mates) and engine_eval is +100 (white mates), same side
            # If sf_eval is -100 (black mates) and engine_eval is -100 (black mates), same side
            same_side_mate = (sf_eval_after > 0) == (engine_eval_after > 0)
            if same_side_mate:
                # Both lead to mate for the same side - just different mate distances
                # This is not a blunder, just a different path to victory
                cp_loss = 0.0
            else:
                # One leads to us mating, other leads to getting mated - this IS a blunder!
                if engine_color == chess.WHITE:
                    cp_loss = sf_eval_after - engine_eval_after
                else:
                    cp_loss = engine_eval_after - sf_eval_after
                cp_loss = max(0, cp_loss)
        else:
            # Calculate centipawn loss from moving side's perspective
            if engine_color == chess.WHITE:
                cp_loss = sf_eval_after - engine_eval_after
            else:
                cp_loss = engine_eval_after - sf_eval_after

            # Ensure non-negative (sometimes engine finds equal or better move)
            cp_loss = max(0, cp_loss)

        game_stats['centipawn_losses'].append(cp_loss)

        # Categorize move
        category = categorize_move(cp_loss)
        game_stats['move_categories'][category] += 1

        # Add CSV line for this move
        full_move_num = (move_num + 1) // 2
        san = board.san(move)
        csv_line = f"{game_num},{round_num},{opponent},{full_move_num},{san},{cp_loss:.4f},{category}\n"
        game_stats['csv_lines'].append(csv_line)

        # Record significant weak moves (>= 1.0 pawns)
        if cp_loss >= 1.0:
            sf_san = board.san(sf_best_move)
            game_stats['weak_moves'].append((full_move_num, san, cp_loss))
            weak_msg = f"    WEAK: Move {full_move_num}. {san} ({category}, -{cp_loss:.2f} pawns) - Stockfish prefers {sf_san}"
            game_stats['console_messages'].append(weak_msg)

        if verbose and cp_loss > 0.5:
            verbose_msg = f"  Move {full_move_num}. {san}: {category}, CP loss = {cp_loss*100:.0f}"
            game_stats['console_messages'].append(verbose_msg)

        board.push(move)

    return game_stats


def analyze_game_worker(args):
    """
    Worker function for parallel game analysis.

    Args:
        Tuple of (game_str, stockfish_path, depth, engine_name, verbose, game_num, round_num, opponent)

    Returns:
        game_stats dictionary with analysis results
    """
    game_str, stockfish_path, depth, engine_name, verbose, game_num, round_num, opponent = args

    # Initialize Stockfish for this worker
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        # Configure Stockfish with fewer threads per worker
        num_threads = 2  # Use 2 threads per worker for better parallelization
        stockfish.configure({"Threads": num_threads, "Hash": 128})
    except Exception:
        pass

    try:
        # Parse the game from string
        from io import StringIO
        game = chess.pgn.read_game(StringIO(game_str))
        if game is None:
            return None

        # Analyze the game
        game_stats = analyze_game(
            game, stockfish, depth, engine_name, verbose,
            game_num, round_num, opponent
        )

        return game_stats
    finally:
        stockfish.quit()


def analyze_pgn_file(
    pgn_path: str,
    stockfish_path: str,
    depth: int,
    engine_name: str,
    max_games: Optional[int] = None,
    verbose: bool = False,
    output_file_path: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> Dict:
    """
    Analyze all games in a PGN file, using parallel processing.

    Returns:
        Aggregated statistics across all games
    """
    # Determine number of worker processes
    if num_workers is None:
        cpu_count = multiprocessing.cpu_count()
        num_workers = max(1, cpu_count // 2)  # Use half of CPUs, 2 threads each

    print(f"Opening PGN file: {pgn_path}")
    print(f"Analyzing moves by: {engine_name}")
    print(f"Stockfish depth: {depth}")
    print(f"Using {num_workers} parallel workers (2 threads each)")
    if output_file_path:
        print(f"Output file: {output_file_path}")
    print()

    # Read all games from PGN file and prepare for parallel processing
    games_to_analyze = []
    with open(pgn_path, 'r') as pgn_file:
        game_num = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            game_num += 1
            if max_games and game_num > max_games:
                break

            # Extract metadata
            white_player = game.headers.get("White", "Unknown")
            black_player = game.headers.get("Black", "Unknown")
            round_num = game.headers.get("Round", "?")

            # Determine opponent
            if engine_name in white_player:
                opponent = black_player
            elif engine_name in black_player:
                opponent = white_player
            else:
                opponent = "Unknown"

            # Convert game to string for pickling
            from io import StringIO
            exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
            game_str = game.accept(exporter)

            games_to_analyze.append((
                game_str, stockfish_path, depth, engine_name, verbose,
                game_num, round_num, opponent
            ))

    total_games = len(games_to_analyze)
    print(f"Loaded {total_games} games from PGN file\n")

    # Open output file if specified
    output_file = None
    if output_file_path:
        output_file = open(output_file_path, 'w')
        output_file.write("Game,Round,Opponent,MoveNum,Move,CPLoss,Category\n")
        output_file.flush()

    all_stats = {
        'games_analyzed': 0,
        'total_moves': 0,
        'top1_matches': 0,
        'all_cp_losses': [],
        'move_categories': defaultdict(int),
        'per_game_stats': [],
        'results': {'wins': 0, 'losses': 0, 'draws': 0},
    }

    start_time = time.time()

    # Analyze games in parallel using multiprocessing
    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Use imap_unordered for better performance (results come as they finish)
            for game_stats in pool.imap_unordered(analyze_game_worker, games_to_analyze):
                if game_stats is None:
                    continue

                if game_stats['moves_analyzed'] == 0:
                    continue

                # Get game metadata from stats
                game_num = games_to_analyze[all_stats['games_analyzed']][5]  # Not accurate for unordered, will fix
                round_num = games_to_analyze[all_stats['games_analyzed']][6]
                opponent = games_to_analyze[all_stats['games_analyzed']][7]

                # Actually, we need to track game_num from the worker result
                # Let me add it to the return value
                # For now, extract from first CSV line
                if game_stats['csv_lines']:
                    first_csv = game_stats['csv_lines'][0]
                    parts = first_csv.split(',')
                    game_num = int(parts[0])
                    round_num = parts[1]
                    opponent = parts[2]

                # Print game header
                print(f"Game {game_num} (Round {round_num} vs {opponent})...")

                # Print console messages (weak moves, verbose output)
                for msg in game_stats['console_messages']:
                    print(msg)

                # Aggregate stats
                all_stats['games_analyzed'] += 1
                all_stats['total_moves'] += game_stats['moves_analyzed']
                all_stats['per_game_stats'].append(game_stats)
                all_stats['top1_matches'] += game_stats['top1_matches']
                all_stats['all_cp_losses'].extend(game_stats['centipawn_losses'])

                for cat, count in game_stats['move_categories'].items():
                    all_stats['move_categories'][cat] += count

                # Track results
                result = game_stats['result']
                engine_color = game_stats['engine_color']

                # Determine result from engine's perspective
                if result == "1-0":
                    if engine_color == "White":
                        all_stats['results']['wins'] += 1
                        result_display = "NN won"
                    else:
                        all_stats['results']['losses'] += 1
                        result_display = "NN lost"
                elif result == "0-1":
                    if engine_color == "Black":
                        all_stats['results']['wins'] += 1
                        result_display = "NN won"
                    else:
                        all_stats['results']['losses'] += 1
                        result_display = "NN lost"
                elif result == "1/2-1/2":
                    all_stats['results']['draws'] += 1
                    result_display = "draw"
                else:
                    result_display = result

                # Show quick stats
                avg_cp = np.mean(game_stats['centipawn_losses']) if game_stats['centipawn_losses'] else 0
                weak_count = len(game_stats['weak_moves'])
                elapsed_placeholder = 0  # Worker doesn't track time, would need to add
                print(f"  Summary: {game_stats['moves_analyzed']} moves, ACPL={avg_cp*100:.0f}, "
                      f"weak={weak_count}, Result={result_display}")

                # Write CSV lines to file
                if output_file:
                    for csv_line in game_stats['csv_lines']:
                        output_file.write(csv_line)
                    output_file.flush()

                print()  # Blank line between games

        elapsed = time.time() - start_time
        all_stats['elapsed_time'] = elapsed

    finally:
        if output_file:
            output_file.close()
            print(f"\nOutput written to: {output_file_path}")

    return all_stats


def print_results(stats: Dict, depth: int):
    """Print analysis results."""
    print("\n" + "="*80)
    print("PGN MOVE QUALITY ANALYSIS")
    print("="*80)

    print(f"\nGames analyzed: {stats['games_analyzed']}")
    print(f"Total engine moves: {stats['total_moves']}")
    print(f"Time taken: {stats['elapsed_time']:.1f}s")
    if stats['total_moves'] > 0:
        print(f"Time per move: {stats['elapsed_time']/stats['total_moves']:.2f}s")

    # Game results
    w = stats['results']['wins']
    l = stats['results']['losses']
    d = stats['results']['draws']
    total = w + l + d
    if total > 0:
        print(f"\n--- Game Results ---")
        print(f"  Wins:   {w:3d} ({100*w/total:.1f}%)")
        print(f"  Losses: {l:3d} ({100*l/total:.1f}%)")
        print(f"  Draws:  {d:3d} ({100*d/total:.1f}%)")

    # Top-1 accuracy
    print("\n--- Move Accuracy ---")
    if stats['total_moves'] > 0:
        top1_acc = 100.0 * stats['top1_matches'] / stats['total_moves']
        print(f"  Top-1 accuracy: {top1_acc:.2f}% "
              f"({stats['top1_matches']}/{stats['total_moves']} moves match Stockfish)")

    # Centipawn loss
    if stats['all_cp_losses']:
        avg_cp_loss = np.mean(stats['all_cp_losses'])
        median_cp_loss = np.median(stats['all_cp_losses'])
        std_cp_loss = np.std(stats['all_cp_losses'])

        print("\n--- Centipawn Loss ---")
        print(f"  Average:  {avg_cp_loss:.3f} pawns ({avg_cp_loss*100:.0f} centipawns)")
        print(f"  Median:   {median_cp_loss:.3f} pawns ({median_cp_loss*100:.0f} centipawns)")
        print(f"  Std dev:  {std_cp_loss:.3f} pawns")

    # Move categories
    print("\n--- Move Quality Distribution ---")
    categories_order = ["best", "excellent", "good", "inaccuracy", "mistake", "blunder"]
    total_cat = sum(stats['move_categories'].values())
    for cat in categories_order:
        count = stats['move_categories'].get(cat, 0)
        pct = 100.0 * count / total_cat if total_cat > 0 else 0
        print(f"  {cat.capitalize():12s}: {count:5d} ({pct:5.1f}%)")

    # Per-game statistics
    if stats['per_game_stats']:
        print("\n--- Per-Game Statistics ---")
        game_acpls = [np.mean(g['centipawn_losses']) * 100 if g['centipawn_losses'] else 0
                      for g in stats['per_game_stats']]
        print(f"  Best game ACPL:  {min(game_acpls):.0f} centipawns")
        print(f"  Worst game ACPL: {max(game_acpls):.0f} centipawns")

        # Show games with most weak moves (>= 1 pawn loss)
        games_with_weak = [(i+1, len(g['weak_moves'])) for i, g in enumerate(stats['per_game_stats'])]
        games_with_weak.sort(key=lambda x: x[1], reverse=True)
        if games_with_weak[0][1] > 0:
            print(f"\n  Games with most weak moves (>= 1 pawn loss):")
            for game_idx, weak_count in games_with_weak[:5]:
                if weak_count > 0:
                    acpl = np.mean(stats['per_game_stats'][game_idx-1]['centipawn_losses']) * 100
                    print(f"    Game {game_idx}: {weak_count} weak moves, ACPL={acpl:.0f}")

    print("\n" + "="*80)

    # Interpretation
    if stats['all_cp_losses']:
        print("\nINTERPRETATION:")
        acpl = avg_cp_loss * 100

        if acpl < 20:
            rating = "2400+ Elo (Super-GM level)"
        elif acpl < 30:
            rating = "2200-2400 Elo (GM level)"
        elif acpl < 50:
            rating = "2000-2200 Elo (IM level)"
        elif acpl < 80:
            rating = "1800-2000 Elo (Expert level)"
        elif acpl < 120:
            rating = "1600-1800 Elo (Advanced)"
        else:
            rating = "<1600 Elo (Intermediate)"

        print(f"  ACPL {acpl:.0f} centipawns suggests: {rating}")
        print(f"  Top-1 accuracy {top1_acc:.1f}% means engine agrees with Stockfish depth-{depth}")
        print(f"  on {top1_acc:.1f}% of moves in actual games")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PGN games for move quality using Stockfish"
    )
    parser.add_argument("pgn_file", help="Path to PGN file")
    parser.add_argument(
        "--stockfish",
        default="/usr/games/stockfish",
        help="Path to Stockfish binary (default: /usr/games/stockfish)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Stockfish analysis depth (default: 20)"
    )
    parser.add_argument(
        "--engine-name",
        default="NeuralEngine",
        help="Name of the engine in PGN headers (default: NeuralEngine)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum number of games to analyze (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-move analysis"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path for move-by-move centipawn loss (optional)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: CPU_count/2)"
    )

    args = parser.parse_args()

    stats = analyze_pgn_file(
        pgn_path=args.pgn_file,
        stockfish_path=args.stockfish,
        depth=args.depth,
        engine_name=args.engine_name,
        max_games=args.max_games,
        verbose=args.verbose,
        output_file_path=args.output,
        num_workers=args.workers,
    )

    print_results(stats, args.depth)


if __name__ == "__main__":
    main()
