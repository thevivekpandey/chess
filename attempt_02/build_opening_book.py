"""
Build opening book by playing 100K short games against Stockfish.
Only plays first 5 moves per side (10 ply total per game).
Stores compact_fen -> best_move mapping.
"""

import chess
import chess.engine
import chess.polyglot
import csv
import os
import time
from play_match import NNPlayer

# Configuration
BOOK_FILE = "opening_book.csv"
NUM_GAMES = 100_000
MAX_MOVES_PER_SIDE = 5  # 5 moves each = 10 ply total
STOCKFISH_TIME_LIMIT = 1.0  # seconds per move for Stockfish


def compact_fen(fen: str) -> str:
    """Remove halfmove clock and fullmove number from FEN."""
    parts = fen.split()
    return ' '.join(parts[:4])


def load_opening_book(filepath: str) -> dict:
    """Load opening book from CSV file."""
    if os.path.exists(filepath):
        print(f"Loading existing opening book from {filepath}...")
        book = {}
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    fen, move = row
                    book[fen] = move
        print(f"Loaded {len(book):,} positions")
        return book
    print("No existing opening book found, starting fresh.")
    return {}


def save_opening_book(book: dict, filepath: str):
    """Save opening book to CSV file."""
    print(f"Saving opening book ({len(book):,} positions) to {filepath}...")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for fen, move in book.items():
            writer.writerow([fen, move])
    print("Saved.")


def append_position(fen: str, move: str, filepath: str):
    """Append a single position to the opening book CSV file."""
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([fen, move])


def play_opening_game(nn_player: NNPlayer, engine: chess.engine.SimpleEngine,
                      nn_is_white: bool, opening_book: dict,
                      max_moves_per_side: int = 5) -> int:
    """
    Play only the first few moves of a game, building opening book.

    Args:
        nn_player: Neural network player
        engine: Stockfish engine
        nn_is_white: True if NN plays white
        opening_book: Dict mapping compact_fen -> move_uci
        max_moves_per_side: Maximum moves per side

    Returns:
        Number of new positions added to opening book
    """
    board = chess.Board()
    new_positions = 0
    total_ply = max_moves_per_side * 2  # Both sides

    for ply in range(total_ply):
        if board.is_game_over():
            break

        is_nn_turn = (board.turn == chess.WHITE) == nn_is_white

        if is_nn_turn:
            fen = compact_fen(board.fen())

            if fen in opening_book:
                # Cache hit - use stored move
                move_uci = opening_book[fen]
                move = chess.Move.from_uci(move_uci)
                # Verify move is legal (in case of hash collision or corruption)
                if move not in board.legal_moves:
                    # Recompute if stored move is illegal
                    move = nn_player.get_best_move(board)
                    opening_book[fen] = move.uci()
                    append_position(fen, move.uci(), BOOK_FILE)
                    new_positions += 1
            else:
                # Cache miss - compute and store
                move = nn_player.get_best_move(board)
                opening_book[fen] = move.uci()
                append_position(fen, move.uci(), BOOK_FILE)
                new_positions += 1
        else:
            # Stockfish's turn
            result = engine.play(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
            move = result.move

        board.push(move)

    return new_positions


def build_opening_book(model_path: str, stockfish_path: str,
                       num_games: int = NUM_GAMES,
                       max_moves_per_side: int = MAX_MOVES_PER_SIDE,
                       search_depth: int = 3):
    """
    Build opening book by playing many short games.

    Args:
        model_path: Path to the trained model
        stockfish_path: Path to Stockfish executable
        num_games: Number of games to play
        max_moves_per_side: Maximum moves per side per game
        search_depth: Search depth for NN player
    """
    print("=" * 70)
    print("BUILDING OPENING BOOK")
    print("=" * 70)
    print(f"Games to play: {num_games:,}")
    print(f"Moves per side: {max_moves_per_side}")
    print(f"NN search depth: {search_depth} ply")
    print(f"Book file: {BOOK_FILE}")
    print("=" * 70)

    # Load existing book
    opening_book = load_opening_book(BOOK_FILE)
    initial_size = len(opening_book)

    # Initialize players
    nn_player = NNPlayer(model_path, search_depth=search_depth)
    # Disable NNPlayer's built-in opening book - we manage our own here
    nn_player.opening_book = {}
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": 1})

    start_time = time.time()
    total_new_positions = 0
    cache_hits = 0
    cache_misses = 0

    try:
        for game_num in range(1, num_games + 1):
            # Alternate colors: odd games = white, even games = black
            nn_is_white = (game_num % 2 == 1)

            # Clear NN's eval cache between games
            nn_player.clear_cache()

            # Play the opening
            positions_before = len(opening_book)
            new_in_game = play_opening_game(
                nn_player, engine, nn_is_white, opening_book, max_moves_per_side
            )

            total_new_positions += new_in_game
            moves_played = max_moves_per_side  # Approximate
            cache_hits += moves_played - new_in_game
            cache_misses += new_in_game

            # Progress update every 100 games
            if game_num % 100 == 0:
                elapsed = time.time() - start_time
                games_per_sec = game_num / elapsed
                eta_seconds = (num_games - game_num) / games_per_sec if games_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60

                hit_rate = 100 * cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0

                print(f"Game {game_num:,}/{num_games:,} | "
                      f"Book size: {len(opening_book):,} | "
                      f"New: {total_new_positions:,} | "
                      f"Hit rate: {hit_rate:.1f}% | "
                      f"Speed: {games_per_sec:.1f} games/s | "
                      f"ETA: {eta_minutes:.1f}m")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        engine.quit()

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("OPENING BOOK BUILD COMPLETE")
        print("=" * 70)
        print(f"Total games played: {game_num:,}")
        print(f"Initial book size: {initial_size:,}")
        print(f"Final book size: {len(opening_book):,}")
        print(f"New positions added: {len(opening_book) - initial_size:,}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Book saved to: {BOOK_FILE}")
        print("=" * 70)


if __name__ == "__main__":
    import sys

    # Configuration
    MODEL_PATH = "chess_model_epoch070.pt"
    STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

    # Check for command line arguments
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        STOCKFISH_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        NUM_GAMES = int(sys.argv[3])

    build_opening_book(MODEL_PATH, STOCKFISH_PATH, NUM_GAMES, MAX_MOVES_PER_SIDE)
