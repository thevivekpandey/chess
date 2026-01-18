"""
Batched Multiprocess Match: Neural Network vs Stockfish
Combines multiprocess parallelization with cloud batching.

Architecture:
- Multiple game processes run in parallel (CPU parallelization)
- A central eval server batches requests from all games
- Batched requests go to cloud GPU (efficient batching)
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
from multiprocessing import Manager
import threading
import queue
import time
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# Cloud GPU inference endpoint
MODEL_INFERENCE_URL = "https://readings-lauren-leo-net.trycloudflare.com/evaluate"
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Encoding": "gzip",
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip"
}

OPENING_BOOK_FILE = "opening_book.csv"
BATCH_SIZE = 1024 * 8
BATCH_INTERVAL = 1.0  # Seconds between batch submissions


# =============================================================================
# LOCAL EVAL SERVER (runs in dedicated process)
# =============================================================================

def eval_server_process(request_queue: mp.Queue, response_dict: dict,
                        shutdown_event: mp.Event, stats_dict: dict):
    """
    Central evaluation server that batches requests from all game processes.

    Args:
        request_queue: Queue where game processes submit (request_id, fens) tuples
        response_dict: Shared dict where results are stored {request_id: {fen: score}}
        shutdown_event: Event to signal server shutdown
        stats_dict: Shared dict for statistics
    """
    pending_requests = []  # List of (request_id, fens) waiting to be batched
    last_batch_time = time.time()

    stats_dict['batches_sent'] = 0
    stats_dict['total_positions'] = 0
    stats_dict['cloud_time'] = 0.0

    print("[EvalServer] Started", flush=True)

    while not shutdown_event.is_set():
        # Collect requests from queue (non-blocking)
        try:
            while True:
                request = request_queue.get_nowait()
                pending_requests.append(request)
        except queue.Empty:
            pass

        # Check if we should send a batch
        current_time = time.time()
        time_since_last = current_time - last_batch_time

        # Collect all unique FENs from pending requests
        all_fens = set()
        for _, fens in pending_requests:
            all_fens.update(fens)

        should_batch = (
            len(all_fens) >= BATCH_SIZE or
            (len(pending_requests) > 0 and time_since_last >= BATCH_INTERVAL)
        )

        if should_batch and len(all_fens) > 0:
            # Send batch to cloud
            fens_list = list(all_fens)

            try:
                payload = gzip.compress(msgpack.packb({"fens": fens_list}))
                cloud_start = time.time()
                response = requests.post(
                    MODEL_INFERENCE_URL,
                    data=payload,
                    headers=HEADERS,
                    timeout=120
                )
                response.raise_for_status()
                scores = response.json()["scores"]
                cloud_time = time.time() - cloud_start

                # Build fen -> score mapping
                fen_scores = {fen: score for fen, score in zip(fens_list, scores)}

                # Distribute results to each request
                for request_id, request_fens in pending_requests:
                    result = {fen: fen_scores[fen] for fen in request_fens}
                    response_dict[request_id] = result

                stats_dict['batches_sent'] += 1
                stats_dict['total_positions'] += len(fens_list)
                stats_dict['cloud_time'] += cloud_time

                print(f"[EvalServer] Batch sent: {len(fens_list):,} positions, "
                      f"{len(pending_requests)} requests, {cloud_time:.1f}s", flush=True)

            except Exception as e:
                print(f"[EvalServer] Error: {e}", flush=True)
                # Mark all pending requests as failed with empty results
                for request_id, _ in pending_requests:
                    response_dict[request_id] = {}

            pending_requests = []
            last_batch_time = time.time()

        # Small sleep to avoid busy-waiting
        time.sleep(0.01)

    # Process any remaining requests before shutdown
    if pending_requests:
        all_fens = set()
        for _, fens in pending_requests:
            all_fens.update(fens)

        if all_fens:
            try:
                fens_list = list(all_fens)
                payload = gzip.compress(msgpack.packb({"fens": fens_list}))
                response = requests.post(
                    MODEL_INFERENCE_URL,
                    data=payload,
                    headers=HEADERS,
                    timeout=120
                )
                response.raise_for_status()
                scores = response.json()["scores"]
                fen_scores = {fen: score for fen, score in zip(fens_list, scores)}

                for request_id, request_fens in pending_requests:
                    result = {fen: fen_scores[fen] for fen in request_fens}
                    response_dict[request_id] = result

            except Exception as e:
                print(f"[EvalServer] Final batch error: {e}", flush=True)

    print("[EvalServer] Shutdown complete", flush=True)


# =============================================================================
# GAME EVALUATOR (uses shared eval server)
# =============================================================================

class GameEvaluator:
    """Evaluator for a single game - queries shared eval server."""

    def __init__(self, request_queue: mp.Queue, response_dict: dict,
                 game_id: int, search_depth: int = 3):
        self.request_queue = request_queue
        self.response_dict = response_dict
        self.game_id = game_id
        self.search_depth = search_depth
        self.eval_cache = {}
        self.opening_book = self._load_opening_book()
        self.request_counter = 0
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

    def evaluate_batch_via_server(self, fens: List[str]) -> Dict[str, float]:
        """Send batch of FENs to eval server and wait for results."""
        request_id = f"{self.game_id}_{self.request_counter}"
        self.request_counter += 1

        # Submit request
        self.request_queue.put((request_id, fens))

        # Wait for response
        while request_id not in self.response_dict:
            time.sleep(0.01)

        result = self.response_dict[request_id]
        del self.response_dict[request_id]  # Clean up
        return result

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
            return book_move, {'from_book': True, 'positions': 0}

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

        # Batch evaluate new positions via server
        if new_positions:
            fens_to_eval = list(new_positions.values())
            hashes_to_eval = list(new_positions.keys())

            scores = self.evaluate_batch_via_server(fens_to_eval)

            for z_hash, fen in zip(hashes_to_eval, fens_to_eval):
                if fen in scores:
                    self.eval_cache[z_hash] = scores[fen]
                else:
                    # Fallback: neutral evaluation if not found
                    self.eval_cache[z_hash] = 0.0

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
            'positions': total_positions
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
    (game_num, nn_is_white, stockfish_path, time_limit, search_depth,
     request_queue, response_dict) = args

    # Initialize evaluator (queries shared server)
    evaluator = GameEvaluator(
        request_queue, response_dict,
        game_id=game_num, search_depth=search_depth
    )

    # Initialize Stockfish engine (each process gets its own)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": 1})

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
# MULTIPROCESS MATCH RUNNER WITH BATCHING
# =============================================================================

def run_batched_multiprocess_match(
    model_path: str,
    stockfish_path: str,
    num_games: int = 100,
    num_workers: int = 4,
    time_limit: float = 1.0,
    search_depth: int = 3,
    games_file: str = "match_games.pgn"
):
    """
    Run a match with multiple games in parallel processes,
    with centralized cloud batching.

    Args:
        model_path: Path to the trained model (unused, for compatibility)
        stockfish_path: Path to Stockfish executable
        num_games: Total number of games to play
        num_workers: Number of parallel worker processes
        time_limit: Time limit per move for Stockfish
        search_depth: Search depth for NN player
        games_file: Path to save all completed games (PGN)
    """
    print("=" * 70)
    print("BATCHED MULTIPROCESS MATCH: Neural Network vs Stockfish")
    print("=" * 70)
    print(f"Total games: {num_games}")
    print(f"Worker processes: {num_workers}")
    print(f"NN search depth: {search_depth} ply")
    print(f"Stockfish time: {time_limit}s per move")
    print(f"Batch interval: {BATCH_INTERVAL}s")
    print(f"Games saved to: {games_file}")
    print("=" * 70)

    # Create shared resources
    manager = Manager()
    request_queue = manager.Queue()
    response_dict = manager.dict()
    shutdown_event = manager.Event()
    stats_dict = manager.dict()

    # Start eval server process
    server_proc = mp.Process(
        target=eval_server_process,
        args=(request_queue, response_dict, shutdown_event, stats_dict)
    )
    server_proc.start()
    time.sleep(0.5)  # Give server time to start

    # Prepare game arguments
    game_args = [
        (game_num, game_num % 2 == 1, stockfish_path, time_limit, search_depth,
         request_queue, response_dict)
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
        f.write(f"; Batched Multiprocess Match: Neural Network vs Stockfish\n")
        f.write(f"; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"; NN Search Depth: {search_depth} ply\n")
        f.write(f"; Stockfish Time: {time_limit}s per move\n")
        f.write(f"; Worker Processes: {num_workers}\n")
        f.write(f"; Batch Interval: {BATCH_INTERVAL}s\n\n")

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

    finally:
        # Shutdown eval server
        shutdown_event.set()
        server_proc.join(timeout=5)
        if server_proc.is_alive():
            server_proc.terminate()

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

    # Server stats
    if 'batches_sent' in stats_dict:
        print(f"\nEval Server Stats:")
        print(f"  Batches sent: {stats_dict['batches_sent']}")
        print(f"  Total positions batched: {stats_dict['total_positions']:,}")
        print(f"  Cloud inference time: {stats_dict['cloud_time']:.1f}s")

    print(f"Games saved to: {games_file}")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # Configuration
    MODEL_PATH = "chess_model_epoch070.pt"
    STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
    NUM_GAMES = 100
    NUM_WORKERS = 4
    TIME_LIMIT = 5.0
    SEARCH_DEPTH = 3

    # Command line arguments
    if len(sys.argv) > 1:
        NUM_WORKERS = int(sys.argv[1])
    if len(sys.argv) > 2:
        NUM_GAMES = int(sys.argv[2])

    run_batched_multiprocess_match(
        MODEL_PATH,
        STOCKFISH_PATH,
        NUM_GAMES,
        NUM_WORKERS,
        TIME_LIMIT,
        SEARCH_DEPTH
    )
