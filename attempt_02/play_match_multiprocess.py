"""
Multiprocess Match: Neural Network vs Stockfish
Runs N completely independent games in parallel processes.
Each process does its own tree exploration and cloud calls.
"""

import chess
import chess.pgn
import chess.engine
import chess.polyglot
import gzip
import msgpack
import csv
import requests
import os
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import time

# Cloud GPU inference endpoint
MODEL_INFERENCE_URL = "https://legislation-champion-gonna-ventures.trycloudflare.com/evaluate"
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Encoding": "gzip",
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip"
}

OPENING_BOOK_FILE = "opening_book.csv"


# =============================================================================
# STANDALONE EVALUATOR (one per process)
# =============================================================================

class GameEvaluator:
    """Evaluator for a single game - runs in its own process."""

    BATCH_SIZE = 1024 * 8

    def __init__(self, search_depth: int = 3):
        self.search_depth = search_depth
        self.eval_cache = {}
        self.opening_book = self._load_opening_book()
        self.cloud_inference_time = 0.0
        self.positions_evaluated = 0

    def _load_opening_book(self) -> dict:
        """Load opening book from CSV file if it exists."""
        if os.path.exists(OPENING_BOOK_FILE):
            book = {}
            with open(OPENING_BOOK_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        fen, move = row
                        book[fen] = move
            return book
        return {}

    def clear_cache(self):
        """Clear the evaluation cache."""
        self.eval_cache = {}

    @staticmethod
    def compact_fen(fen: str) -> str:
        """Remove halfmove clock and fullmove number from FEN."""
        parts = fen.split()
        return ' '.join(parts[:4])

    def get_opening_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Check if position is in opening book."""
        if self.opening_book:
            fen = self.compact_fen(board.fen())
            if fen in self.opening_book:
                move_uci = self.opening_book[fen]
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    return move
        return None

    def collect_leaf_positions(self, board: chess.Board, depth: int) -> List[Tuple[int, str]]:
        """Collect all positions at leaf nodes for batch evaluation."""
        if board.is_game_over():
            return []

        if depth == 0:
            z_hash = chess.polyglot.zobrist_hash(board)
            if z_hash in self.eval_cache:
                return [(z_hash, None)]
            fen = self.compact_fen(board.fen())
            return [(z_hash, fen)]

        positions = []
        for move in board.legal_moves:
            board.push(move)
            positions.extend(self.collect_leaf_positions(board, depth - 1))
            board.pop()
        return positions

    def evaluate_batch_cloud(self, fens: List[str]) -> Dict[str, float]:
        """Send batch of FENs to cloud GPU for evaluation."""
        results = {}
        for i in range(0, len(fens), self.BATCH_SIZE):
            batch = fens[i:i + self.BATCH_SIZE]
            payload = gzip.compress(msgpack.packb({"fens": batch}))

            # Retry loop
            while True:
                try:
                    request_start = time.time()
                    response = requests.post(
                        MODEL_INFERENCE_URL,
                        data=payload,
                        headers=HEADERS,
                        timeout=120
                    )
                    response.raise_for_status()
                    scores = response.json()["scores"]
                    self.cloud_inference_time += time.time() - request_start
                    for fen, score in zip(batch, scores):
                        results[fen] = score
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"  [Cloud error: {e}] Retrying in 5s...", flush=True)
                    time.sleep(5)
        return results

    def minimax(self, board: chess.Board, depth: int, maximizing: bool) -> float:
        """Full minimax search using cached evaluations."""
        if board.is_game_over():
            if board.is_checkmate():
                return (-1.0 - depth * 0.01) if maximizing else (1.0 + depth * 0.01)
            return 0.0

        if depth == 0:
            z_hash = chess.polyglot.zobrist_hash(board)
            return self.eval_cache[z_hash]

        if maximizing:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
            return min_eval

    def get_best_move(self, board: chess.Board) -> Tuple[chess.Move, dict]:
        """Get best move for a single board."""
        # Check opening book first
        book_move = self.get_opening_book_move(board)
        if book_move:
            return book_move, {'from_book': True, 'positions': 0, 'cloud_time': 0}

        # Collect leaf positions
        leaf_positions = self.collect_leaf_positions(board, self.search_depth)

        # Deduplicate by hash
        unique_positions = {}
        for z_hash, fen in leaf_positions:
            if z_hash not in unique_positions:
                unique_positions[z_hash] = fen

        # Filter to positions needing evaluation
        new_positions = {h: f for h, f in unique_positions.items() if f is not None}
        total_positions = len(unique_positions)

        # Batch evaluate new positions
        cloud_time = 0.0
        if new_positions:
            fens_to_eval = list(new_positions.values())
            hashes_to_eval = list(new_positions.keys())

            cloud_start = time.time()
            scores = self.evaluate_batch_cloud(fens_to_eval)
            cloud_time = time.time() - cloud_start

            for z_hash, fen in zip(hashes_to_eval, fens_to_eval):
                self.eval_cache[z_hash] = scores[fen]

            self.positions_evaluated += len(new_positions)

        # Run minimax
        maximizing = board.turn == chess.WHITE
        best_move = None

        if maximizing:
            best_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, self.search_depth - 1, False)
                board.pop()
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
        else:
            best_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, self.search_depth - 1, True)
                board.pop()
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move

        stats = {
            'from_book': False,
            'positions': total_positions,
            'cloud_time': cloud_time
        }
        return best_move, stats


# =============================================================================
# SINGLE GAME RUNNER (runs in separate process)
# =============================================================================

def format_move_number(move_count: int) -> str:
    """Format move number in chess notation."""
    move_num = (move_count // 2) + 1
    if move_count % 2 == 0:
        return f"{move_num}. "
    else:
        return f"{move_num}.."


def play_single_game(args) -> dict:
    """
    Play a single game in its own process.
    Returns game result dict.
    """
    game_num, nn_is_white, stockfish_path, time_limit, search_depth, skill_level = args

    # Initialize evaluator (each process gets its own)
    evaluator = GameEvaluator(search_depth=search_depth)

    # Initialize Stockfish engine (each process gets its own)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": skill_level})

    # Initialize game
    board = chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "NN vs Stockfish Match"
    pgn_game.headers["Site"] = "Local"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_num)
    pgn_game.headers["White"] = "NeuralNet" if nn_is_white else "Stockfish"
    pgn_game.headers["Black"] = "Stockfish" if nn_is_white else "NeuralNet"
    pgn_node = pgn_game

    move_count = 0

    try:
        while not board.is_game_over() and move_count < 300:
            is_nn_turn = (board.turn == chess.WHITE) == nn_is_white
            move_label = format_move_number(move_count)

            if is_nn_turn:
                move, stats = evaluator.get_best_move(board)
                move_san = board.san(move)
                if stats['from_book']:
                    print(f"  Game {game_num}, {move_label} (NN): {move_san} | [BOOK]", flush=True)
                else:
                    print(f"  Game {game_num}, {move_label} (NN): {move_san} | Positions: {stats['positions']:,}", flush=True)
            else:
                result = engine.play(board, chess.engine.Limit(time=time_limit))
                move = result.move
                move_san = board.san(move)
                print(f"  Game {game_num}, {move_label} (SF): {move_san}", flush=True)

            pgn_node = pgn_node.add_variation(move)
            board.push(move)
            move_count += 1

    finally:
        engine.quit()

    # Determine result
    if move_count >= 300:
        result = "1/2-1/2"
        winner = 0
        pgn_game.headers["Result"] = "1/2-1/2"
        pgn_game.headers["Termination"] = "Move limit reached"
    else:
        result = board.result()
        pgn_game.headers["Result"] = result
        if result == "1-0":
            winner = 1
        elif result == "0-1":
            winner = -1
        else:
            winner = 0

    # Determine if NN won
    nn_won = (winner == 1 and nn_is_white) or (winner == -1 and not nn_is_white)
    sf_won = (winner == 1 and not nn_is_white) or (winner == -1 and nn_is_white)

    return {
        'game_num': game_num,
        'nn_is_white': nn_is_white,
        'result': result,
        'winner': winner,
        'nn_won': nn_won,
        'sf_won': sf_won,
        'moves': move_count,
        'pgn': str(pgn_game),
        'positions_evaluated': evaluator.positions_evaluated
    }


# =============================================================================
# MULTIPROCESS MATCH RUNNER
# =============================================================================

def run_multiprocess_match(
    model_path: str,
    stockfish_path: str,
    num_games: int = 100,
    num_workers: int = 4,
    time_limit: float = 1.0,
    search_depth: int = 3,
    skill_level: int = 3,
    games_file: str = "match_games.pgn"
):
    """
    Run a match with multiple games in parallel processes.

    Args:
        model_path: Path to the trained model (unused, for compatibility)
        stockfish_path: Path to Stockfish executable
        num_games: Total number of games to play
        num_workers: Number of parallel worker processes
        time_limit: Time limit per move for Stockfish
        search_depth: Search depth for NN player
        skill_level: Stockfish skill level (0-20)
        games_file: Path to save all completed games (PGN)
    """
    print("=" * 70)
    print("MULTIPROCESS MATCH: Neural Network vs Stockfish")
    print("=" * 70)
    print(f"Total games: {num_games}")
    print(f"Worker processes: {num_workers}")
    print(f"NN search depth: {search_depth} ply")
    print(f"Stockfish skill level: {skill_level}")
    print(f"Stockfish time: {time_limit}s per move")
    print(f"Games saved to: {games_file}")
    print("=" * 70)

    # Prepare game arguments
    game_args = [
        (game_num, game_num % 2 == 1, stockfish_path, time_limit, search_depth, skill_level)
        for game_num in range(1, num_games + 1)
    ]

    # Statistics
    nn_wins = 0
    sf_wins = 0
    draws = 0
    total_positions = 0

    start_time = time.time()

    # Open file for all games
    with open(games_file, 'w') as f:
        f.write(f"; Multiprocess Match: Neural Network vs Stockfish\n")
        f.write(f"; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"; NN Search Depth: {search_depth} ply\n")
        f.write(f"; Stockfish Skill Level: {skill_level}\n")
        f.write(f"; Stockfish Time: {time_limit}s per move\n")
        f.write(f"; Worker Processes: {num_workers}\n\n")

    # Run games in parallel
    try:
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(play_single_game, game_args):
                # Update statistics
                if result['nn_won']:
                    nn_wins += 1
                    outcome = "NN wins"
                elif result['sf_won']:
                    sf_wins += 1
                    outcome = "SF wins"
                else:
                    draws += 1
                    outcome = "Draw"

                total_positions += result['positions_evaluated']

                color = "White" if result['nn_is_white'] else "Black"
                print(f"Game {result['game_num']} complete: {result['result']} "
                      f"(NN={color}) - {outcome} [{result['moves']} moves]", flush=True)
                print()

                # Save PGN
                with open(games_file, 'a') as f:
                    f.write(result['pgn'] + "\n\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    total_time = time.time() - start_time
    games_completed = nn_wins + sf_wins + draws

    # Print results
    print("\n" + "=" * 70)
    print("MATCH RESULTS")
    print("=" * 70)
    print(f"Neural Network: {nn_wins} wins ({nn_wins/max(games_completed,1)*100:.1f}%)")
    print(f"Stockfish:      {sf_wins} wins ({sf_wins/max(games_completed,1)*100:.1f}%)")
    print(f"Draws:          {draws} ({draws/max(games_completed,1)*100:.1f}%)")
    print(f"\nGames completed: {games_completed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Positions evaluated: {total_positions:,}")
    print(f"Games saved to: {games_file}")
    print("=" * 70)

    # Write summary to PGN file
    with open(games_file, 'a') as f:
        f.write("\n")
        f.write("; " + "=" * 50 + "\n")
        f.write("; MATCH SUMMARY\n")
        f.write("; " + "=" * 50 + "\n")
        f.write(f"; Neural Network wins: {nn_wins} ({nn_wins/max(games_completed,1)*100:.1f}%)\n")
        f.write(f"; Stockfish wins:      {sf_wins} ({sf_wins/max(games_completed,1)*100:.1f}%)\n")
        f.write(f"; Draws:               {draws} ({draws/max(games_completed,1)*100:.1f}%)\n")
        f.write(f"; Games completed:     {games_completed}\n")
        f.write(f"; Total time:          {total_time/60:.1f} minutes\n")
        f.write(f"; Positions evaluated: {total_positions:,}\n")
        f.write("; " + "=" * 50 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run a match between Neural Network and Stockfish',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python play_match_multiprocess.py --skill 3 --games 100 --workers 4
    python play_match_multiprocess.py --skill 5 --games 50 --depth 4
        """
    )
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of parallel worker processes (default: 4)')
    parser.add_argument('--games', '-g', type=int, default=100,
                        help='Number of games to play (default: 100)')
    parser.add_argument('--skill', '-s', type=int, default=3,
                        help='Stockfish skill level 0-20 (default: 3)')
    parser.add_argument('--depth', '-d', type=int, default=3,
                        help='NN search depth in ply (default: 3)')
    parser.add_argument('--time', '-t', type=float, default=5.0,
                        help='Stockfish time per move in seconds (default: 5.0)')
    parser.add_argument('--stockfish', type=str, default="/opt/homebrew/bin/stockfish",
                        help='Path to Stockfish executable')
    parser.add_argument('--output', '-o', type=str, default="match_games.pgn",
                        help='Output PGN file (default: match_games.pgn)')

    args = parser.parse_args()

    # Configuration
    MODEL_PATH = "chess_model_epoch070.pt"

    run_multiprocess_match(
        MODEL_PATH,
        args.stockfish,
        args.games,
        args.workers,
        args.time,
        args.depth,
        args.skill,
        args.output
    )
