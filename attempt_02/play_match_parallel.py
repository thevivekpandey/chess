"""
Parallel Match: Neural Network vs Stockfish
Plays N games in parallel with coordinated cloud batching for efficiency.
"""

import chess
import chess.pgn
import chess.engine
import chess.polyglot
import numpy as np
import gzip
import json
import msgpack
import csv
import requests
import os
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

# Cloud GPU inference endpoint
MODEL_INFERENCE_URL = "https://feel-roy-view-archived.trycloudflare.com/evaluate"
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Encoding": "gzip",
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip"
}

# =============================================================================
# BATCH EVALUATOR (shared across all parallel games)
# =============================================================================

class BatchEvaluator:
    """Handles batched cloud evaluation for multiple parallel games."""

    BATCH_SIZE = 1024 * 32
    OPENING_BOOK_FILE = "opening_book.csv"

    def __init__(self, search_depth: int = 3):
        self.search_depth = search_depth
        self.eval_cache = {}  # Shared cache across all games
        self.opening_book = self._load_opening_book()
        self.cloud_inference_time = 0.0
        self.total_positions_evaluated = 0

    def _load_opening_book(self) -> dict:
        """Load opening book from CSV file if it exists."""
        if os.path.exists(self.OPENING_BOOK_FILE):
            book = {}
            with open(self.OPENING_BOOK_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        fen, move = row
                        book[fen] = move
            print(f"Loaded opening book with {len(book):,} positions")
            return book
        print("No opening book found")
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

    def get_best_moves_batch(self, boards: List[chess.Board]) -> List[Tuple[chess.Move, dict]]:
        """
        Get best moves for multiple boards with coordinated cloud batching.
        Returns list of (move, stats_dict) tuples.
        """
        results = []
        boards_needing_search = []
        board_indices = []

        # First pass: check opening book for each board
        for i, board in enumerate(boards):
            book_move = self.get_opening_book_move(board)
            if book_move:
                results.append((i, book_move, {'from_book': True}))
            else:
                boards_needing_search.append(board)
                board_indices.append(i)

        if not boards_needing_search:
            # All moves from opening book
            results.sort(key=lambda x: x[0])
            return [(move, stats) for _, move, stats in results]

        # Collect leaf positions from all boards
        all_positions = {}  # hash -> fen
        board_position_counts = []

        for board in boards_needing_search:
            leaf_positions = self.collect_leaf_positions(board, self.search_depth)
            unique_for_board = {}
            for z_hash, fen in leaf_positions:
                if z_hash not in unique_for_board:
                    unique_for_board[z_hash] = fen
                if z_hash not in all_positions and fen is not None:
                    all_positions[z_hash] = fen
            board_position_counts.append(len(unique_for_board))

        # Filter to positions not in cache
        new_positions = {h: f for h, f in all_positions.items() if f is not None and h not in self.eval_cache}

        total_positions = len(all_positions) + sum(1 for h, f in all_positions.items() if f is None or h in self.eval_cache)
        cache_hits = total_positions - len(new_positions)

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

            self.total_positions_evaluated += len(new_positions)

        # Run minimax for each board
        for idx, board in enumerate(boards_needing_search):
            original_idx = board_indices[idx]
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
                'positions': board_position_counts[idx],
                'new_positions': len(new_positions),
                'cache_hits': cache_hits,
                'cloud_time': cloud_time / len(boards_needing_search),  # Amortized
            }
            results.append((original_idx, best_move, stats))

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [(move, stats) for _, move, stats in results]


# =============================================================================
# PARALLEL GAME STATE
# =============================================================================

class GameState:
    """State for a single game in the parallel match."""

    def __init__(self, game_num: int, nn_is_white: bool):
        self.game_num = game_num
        self.nn_is_white = nn_is_white
        self.board = chess.Board()
        self.pgn_game = chess.pgn.Game()
        self.pgn_game.headers["Event"] = "NN vs Stockfish Match"
        self.pgn_game.headers["Site"] = "Local"
        self.pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        self.pgn_game.headers["Round"] = str(game_num)
        self.pgn_game.headers["White"] = "NeuralNet" if nn_is_white else "Stockfish"
        self.pgn_game.headers["Black"] = "Stockfish" if nn_is_white else "NeuralNet"
        self.pgn_node = self.pgn_game
        self.move_count = 0
        self.finished = False
        self.result = None
        self.winner = None  # 1=white, -1=black, 0=draw

    def is_nn_turn(self) -> bool:
        """Check if it's the neural network's turn."""
        return (self.board.turn == chess.WHITE) == self.nn_is_white

    def format_move_number(self) -> str:
        """Format move number in chess notation (1.  for white, 1.. for black)."""
        move_num = (self.move_count // 2) + 1
        if self.move_count % 2 == 0:
            return f"{move_num}. "  # Extra space for alignment
        else:
            return f"{move_num}.."

    def make_move(self, move: chess.Move):
        """Make a move and update game state."""
        self.move_count += 1
        self.pgn_node = self.pgn_node.add_variation(move)
        self.board.push(move)

        # Check for game end
        if self.board.is_game_over() or self.move_count > 300:
            self.finished = True
            if self.move_count > 300:
                self.result = "1/2-1/2"
                self.winner = 0
                self.pgn_game.headers["Result"] = "1/2-1/2"
                self.pgn_game.headers["Termination"] = "Move limit reached"
            else:
                result = self.board.result()
                self.pgn_game.headers["Result"] = result
                self.result = result
                if result == "1-0":
                    self.winner = 1
                elif result == "0-1":
                    self.winner = -1
                else:
                    self.winner = 0


# =============================================================================
# PARALLEL MATCH RUNNER
# =============================================================================

def run_parallel_match(
    model_path: str,
    stockfish_path: str,
    num_games: int = 100,
    num_parallel: int = 4,
    time_limit: float = 1.0,
    search_depth: int = 3,
    games_file: str = "match_games.pgn"
):
    """
    Run a match with multiple games in parallel.

    Args:
        model_path: Path to the trained model (unused, for compatibility)
        stockfish_path: Path to Stockfish executable
        num_games: Total number of games to play
        num_parallel: Number of games to run in parallel
        time_limit: Time limit per move for Stockfish
        search_depth: Search depth for NN player
        games_file: Path to save all completed games (PGN)
    """
    print("=" * 70)
    print("PARALLEL MATCH: Neural Network vs Stockfish")
    print("=" * 70)
    print(f"Total games: {num_games}")
    print(f"Parallel games: {num_parallel}")
    print(f"NN search depth: {search_depth} ply")
    print(f"Stockfish time: {time_limit}s per move")
    print(f"Games saved to: {games_file}")
    print("=" * 70)

    # Initialize batch evaluator (shared)
    evaluator = BatchEvaluator(search_depth=search_depth)

    # Initialize Stockfish engines (one per parallel game)
    engines = []
    for i in range(num_parallel):
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Skill Level": 1})
        engines.append(engine)
    print(f"Initialized {num_parallel} Stockfish engines")

    # Statistics
    nn_wins = 0
    sf_wins = 0
    draws = 0
    games_completed = 0

    start_time = time.time()

    # Open file for all games
    with open(games_file, 'w') as all_games_f:
        all_games_f.write(f"; Parallel Match: Neural Network vs Stockfish\n")
        all_games_f.write(f"; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        all_games_f.write(f"; NN Search Depth: {search_depth} ply\n")
        all_games_f.write(f"; Stockfish Time: {time_limit}s per move\n")
        all_games_f.write(f"; Parallel Games: {num_parallel}\n\n")

    try:
        game_num = 0
        while games_completed < num_games:
            # Start a batch of parallel games
            batch_size = min(num_parallel, num_games - games_completed)
            games = []

            for i in range(batch_size):
                game_num += 1
                nn_is_white = (game_num % 2 == 1)
                games.append(GameState(game_num, nn_is_white))

            # Clear cache for new batch
            evaluator.clear_cache()

            batch_start = time.time()
            moves_this_batch = 0

            # Play until all games in batch are finished
            while any(not g.finished for g in games):
                active_games = [g for g in games if not g.finished]

                # Separate games by whose turn it is
                nn_turn_games = [g for g in active_games if g.is_nn_turn()]
                sf_turn_games = [g for g in active_games if not g.is_nn_turn()]

                # Process NN moves (batched)
                if nn_turn_games:
                    boards = [g.board for g in nn_turn_games]
                    move_results = evaluator.get_best_moves_batch(boards)

                    for game, (move, stats) in zip(nn_turn_games, move_results):
                        move_san = game.board.san(move)
                        move_label = game.format_move_number()
                        if stats['from_book']:
                            print(f"  Game {game.game_num}, {move_label} (NN): {move_san} | [BOOK]", flush=True)
                        else:
                            print(f"  Game {game.game_num}, {move_label} (NN): {move_san} | "
                                  f"Positions: {stats['positions']:,}", flush=True)
                        game.make_move(move)
                        moves_this_batch += 1

                # Process Stockfish moves (parallel using ThreadPoolExecutor)
                if sf_turn_games:
                    def get_sf_move(game_and_engine):
                        game, engine = game_and_engine
                        result = engine.play(game.board, chess.engine.Limit(time=time_limit))
                        return game, result.move

                    # Pair each game with its engine
                    game_engine_pairs = [
                        (game, engines[games.index(game) % len(engines)])
                        for game in sf_turn_games
                    ]

                    # Execute all SF moves in parallel
                    with ThreadPoolExecutor(max_workers=len(sf_turn_games)) as executor:
                        sf_results = list(executor.map(get_sf_move, game_engine_pairs))

                    # Apply the moves
                    for game, move in sf_results:
                        move_san = game.board.san(move)
                        move_label = game.format_move_number()
                        print(f"  Game {game.game_num}, {move_label} (SF): {move_san}", flush=True)
                        game.make_move(move)
                        moves_this_batch += 1

            batch_time = time.time() - batch_start

            # Process completed games
            for game in games:
                games_completed += 1

                # Update statistics
                nn_won = (game.winner == 1 and game.nn_is_white) or (game.winner == -1 and not game.nn_is_white)
                sf_won = (game.winner == 1 and not game.nn_is_white) or (game.winner == -1 and game.nn_is_white)

                if nn_won:
                    nn_wins += 1
                    outcome = "NN wins"
                elif sf_won:
                    sf_wins += 1
                    outcome = "SF wins"
                else:
                    draws += 1
                    outcome = "Draw"

                color = "White" if game.nn_is_white else "Black"
                print(f"Game {game.game_num} complete: {game.result} (NN={color}) - {outcome}")

                # Save to PGN
                with open(games_file, 'a') as f:
                    f.write(str(game.pgn_game) + "\n\n")

            # Batch summary
            print(f"--- Batch complete: {batch_size} games, {moves_this_batch} moves, {batch_time:.1f}s ---\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Cleanup engines
        for engine in engines:
            engine.quit()

    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("MATCH RESULTS")
    print("=" * 70)
    print(f"Neural Network: {nn_wins} wins ({nn_wins/max(games_completed,1)*100:.1f}%)")
    print(f"Stockfish:      {sf_wins} wins ({sf_wins/max(games_completed,1)*100:.1f}%)")
    print(f"Draws:          {draws} ({draws/max(games_completed,1)*100:.1f}%)")
    print(f"\nGames completed: {games_completed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Positions evaluated: {evaluator.total_positions_evaluated:,}")
    print(f"Games saved to: {games_file}")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # Configuration
    MODEL_PATH = "chess_model_epoch070.pt"
    STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
    NUM_GAMES = 100
    NUM_PARALLEL = 4
    TIME_LIMIT = 5.0
    SEARCH_DEPTH = 3

    # Command line arguments
    if len(sys.argv) > 1:
        NUM_PARALLEL = int(sys.argv[1])
    if len(sys.argv) > 2:
        NUM_GAMES = int(sys.argv[2])

    run_parallel_match(
        MODEL_PATH,
        STOCKFISH_PATH,
        NUM_GAMES,
        NUM_PARALLEL,
        TIME_LIMIT,
        SEARCH_DEPTH
    )
