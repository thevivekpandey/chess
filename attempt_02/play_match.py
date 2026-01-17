"""
Match: Neural Network (3-ply search) vs Stockfish (skill level 0, 1s/move)
Plays 100 games and reports results.
"""

import chess
import chess.pgn
import chess.engine
import chess.polyglot
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gzip
import json
import msgpack
import requests
from typing import Tuple, Optional, TextIO
from datetime import datetime
import time

# Cloud GPU inference endpoint
MODEL_INFERENCE_URL = "https://alba-nondedicative-roxann.ngrok-free.dev/evaluate"
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Encoding": "gzip",
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip"
}

# =============================================================================
# NEURAL NETWORK (copied from chess_engine.py)
# =============================================================================

NUM_PLANES = 18

PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert a FEN string to an 18-plane tensor representation."""
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    parts = fen.split()
    board_str = parts[0]
    side_to_move = parts[1] if len(parts) > 1 else 'w'
    castling = parts[2] if len(parts) > 2 else '-'
    en_passant = parts[3] if len(parts) > 3 else '-'

    row = 0
    col = 0
    for char in board_str:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            if char in PIECE_TO_PLANE:
                plane_idx = PIECE_TO_PLANE[char]
                planes[plane_idx, row, col] = 1.0
            col += 1

    if 'K' in castling:
        planes[12, :, :] = 1.0
    if 'Q' in castling:
        planes[13, :, :] = 1.0
    if 'k' in castling:
        planes[14, :, :] = 1.0
    if 'q' in castling:
        planes[15, :, :] = 1.0

    if side_to_move == 'w':
        planes[16, :, :] = 1.0

    if en_passant != '-':
        ep_col = ord(en_passant[0]) - ord('a')
        ep_row = 8 - int(en_passant[1])
        planes[17, ep_row, ep_col] = 1.0

    return torch.from_numpy(planes)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    def __init__(self, initial_channels: int = 512, res_channels: int = 256, num_res_blocks: int = 8):
        super().__init__()
        self.initial_conv = nn.Conv2d(NUM_PLANES, initial_channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(initial_channels)
        self.transition_conv = nn.Conv2d(initial_channels, res_channels, kernel_size=1, bias=False)
        self.transition_bn = nn.BatchNorm2d(res_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(res_channels) for _ in range(num_res_blocks)
        ])
        self.value_conv = nn.Conv2d(res_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = F.relu(self.transition_bn(self.transition_conv(out)))
        for block in self.res_blocks:
            out = block(out)
        out = F.relu(self.value_bn(self.value_conv(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out


# =============================================================================
# MINIMAX SEARCH (FULL TREE)
# =============================================================================

class NNPlayer:
    """Neural network player with minimax search using cloud GPU inference."""

    BATCH_SIZE = 1024 * 8
    OPENING_BOOK_FILE = "opening_book.csv"

    def __init__(self, model_path: str, device: str = 'auto', search_depth: int = 3):
        # model_path kept for API compatibility but not used (cloud handles model)
        print(f"Using cloud inference at {MODEL_INFERENCE_URL}")
        self.search_depth = search_depth
        self.nodes_evaluated = 0
        self.eval_cache = {}
        self.cloud_inference_time = 0.0  # Time spent on remote inference calls
        self.cache_hits = 0  # Cache hits for last move
        self.cache_requests = 0  # Total positions requested for last move
        self.used_opening_book = False  # True if last move came from opening book
        self.opening_book = self._load_opening_book()

    def _load_opening_book(self) -> dict:
        """Load opening book from CSV file if it exists."""
        import os
        import csv
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
        """Clear the evaluation cache. Call between games."""
        self.eval_cache = {}

    @staticmethod
    def compact_fen(fen: str) -> str:
        """Remove halfmove clock and fullmove number from FEN."""
        parts = fen.split()
        return ' '.join(parts[:4])

    def evaluate_batch_cloud(self, fens: list) -> dict:
        """
        Send batch of FENs to cloud GPU for evaluation.
        Returns dict mapping FEN -> score.
        Accumulates time spent on HTTP requests in self.cloud_inference_time.
        """
        results = {}
        for i in range(0, len(fens), self.BATCH_SIZE):
            batch = fens[i:i + self.BATCH_SIZE]
            #payload = gzip.compress(json.dumps({"fens": batch}).encode())
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

    def collect_leaf_positions(self, board: chess.Board, depth: int) -> list:
        """
        Collect all positions at leaf nodes (depth 0) for batch evaluation.
        Returns list of (zobrist_hash, compact_fen) tuples.
        FEN is None if position is already cached (skips expensive FEN generation).
        """
        # Terminal conditions - no position needed
        if board.is_game_over():
            return []

        # Leaf node - collect hash, only generate FEN if not cached
        if depth == 0:
            z_hash = chess.polyglot.zobrist_hash(board)
            if z_hash in self.eval_cache:
                return [(z_hash, None)]  # Already cached, skip FEN generation
            fen = self.compact_fen(board.fen())
            return [(z_hash, fen)]

        # Recurse through all legal moves
        positions = []
        for move in board.legal_moves:
            board.push(move)
            positions.extend(self.collect_leaf_positions(board, depth - 1))
            board.pop()
        return positions

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using cached cloud results.
        Returns score from white's perspective (positive = white better).
        """
        self.nodes_evaluated += 1
        z_hash = chess.polyglot.zobrist_hash(board)
        if z_hash in self.eval_cache:
            return self.eval_cache[z_hash]
        # Fallback: position not in cache (shouldn't happen if pre-collection is correct)
        raise ValueError(f"Position not found in cache: {z_hash}")

    def minimax(self, board: chess.Board, depth: int, maximizing: bool) -> float:
        """
        Full minimax search.

        Args:
            board: Current position
            depth: Remaining search depth
            maximizing: True if maximizing player (white)

        Returns:
            Position evaluation
        """
        # Terminal conditions
        if board.is_game_over():
            if board.is_checkmate():
                # Prefer faster mates by adding depth bonus
                return (-1.0 - depth * 0.01) if maximizing else (1.0 + depth * 0.01)
            return 0.0  # Stalemate or draw

        # Leaf node - use neural network
        if depth == 0:
            return self.evaluate(board)

        legal_moves = list(board.legal_moves)

        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
            return min_eval

    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Find the best move using minimax search with cloud batch evaluation."""
        self.nodes_evaluated = 0
        self.cloud_inference_time = 0.0  # Reset inference time for this move
        self.used_opening_book = False

        # Check opening book first
        if self.opening_book:
            fen = self.compact_fen(board.fen())
            if fen in self.opening_book:
                move_uci = self.opening_book[fen]
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    self.used_opening_book = True
                    return move

        # Step 1: Collect all leaf positions (hash, fen) tuples
        leaf_positions = self.collect_leaf_positions(board, self.search_depth)

        # Step 2: Deduplicate by hash
        unique_positions = {}  # hash -> fen (fen is None if already cached)
        for z_hash, fen in leaf_positions:
            if z_hash not in unique_positions:
                unique_positions[z_hash] = fen

        # Step 3: Filter to positions needing evaluation (fen is not None)
        # If fen is None, position was already in cache during collection
        new_positions = {h: f for h, f in unique_positions.items() if f is not None}
        self.cache_requests = len(unique_positions)
        self.cache_hits = len(unique_positions) - len(new_positions)

        # Step 4: Batch evaluate only NEW positions via cloud GPU
        if new_positions:
            # Send FENs to cloud, get scores back
            fens_to_eval = list(new_positions.values())
            hashes_to_eval = list(new_positions.keys())
            scores = self.evaluate_batch_cloud(fens_to_eval)
            # Store results keyed by hash
            for z_hash, fen in zip(hashes_to_eval, fens_to_eval):
                self.eval_cache[z_hash] = scores[fen]

        # Step 5: Run minimax search using cached evaluations
        best_move = None
        maximizing = board.turn == chess.WHITE

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

        return best_move


# =============================================================================
# MATCH RUNNER
# =============================================================================

def play_game(nn_player: NNPlayer, engine: chess.engine.SimpleEngine,
              nn_is_white: bool, game_num: int, pgn_file: TextIO,
              time_limit: float = 1.0) -> Tuple[str, int]:
    """
    Play a single game.

    Args:
        nn_player: Neural network player
        engine: Stockfish engine
        nn_is_white: True if NN plays white
        game_num: Game number for PGN header
        pgn_file: File handle to write PGN moves
        time_limit: Time limit per move for Stockfish

    Returns:
        Tuple of (result_string, winner) where winner is 1=white, -1=black, 0=draw
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "NN vs Stockfish Match"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_num)
    game.headers["White"] = "NeuralNet" if nn_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if nn_is_white else "NeuralNet"

    node = game
    move_count = 0

    while not board.is_game_over():
        move_count += 1

        nn_move = False
        if board.turn == chess.WHITE:
            if nn_is_white:
                nn_move = True
                eval_start = time.time()
                move = nn_player.get_best_move(board)
                eval_time = time.time() - eval_start
            else:
                result = engine.play(board, chess.engine.Limit(time=time_limit))
                move = result.move
        else:
            if nn_is_white:
                result = engine.play(board, chess.engine.Limit(time=time_limit))
                move = result.move
            else:
                nn_move = True
                eval_start = time.time()
                move = nn_player.get_best_move(board)
                eval_time = time.time() - eval_start

        # Get SAN notation before pushing the move
        move_san = board.san(move)
        player = "NN" if nn_move else "SF"

        if nn_move:
            if nn_player.used_opening_book:
                print(f"  Game {game_num}, Move {move_count} ({player}): {move_san} | "
                      f"[BOOK] {eval_time:.3f}s", flush=True)
            else:
                cloud_time = nn_player.cloud_inference_time
                local_time = eval_time - cloud_time
                evals_per_sec = nn_player.nodes_evaluated / cloud_time if cloud_time > 0 else 0
                cache_pct = 100 * nn_player.cache_hits / nn_player.cache_requests if nn_player.cache_requests > 0 else 0
                new_evals = nn_player.cache_requests - nn_player.cache_hits
                print(f"  Game {game_num}, Move {move_count} ({player}): {move_san} | "
                      f"Positions: {nn_player.cache_requests:,} (new: {new_evals:,}, cached: {nn_player.cache_hits:,}, {cache_pct:.0f}%) | "
                      f"Total: {eval_time:.2f}s (cloud: {cloud_time:.2f}s, local: {local_time:.2f}s) | "
                      f"{evals_per_sec:,.0f} evals/s", flush=True)
        else:
            print(f"  Game {game_num}, Move {move_count} ({player}): {move_san}", flush=True)

        # Add move to PGN game tree
        node = node.add_variation(move)
        board.push(move)

        # Write current game state to file and flush
        pgn_file.seek(0)
        pgn_file.truncate()
        pgn_file.write(str(game) + "\n\n")
        pgn_file.flush()

        # Safety limit
        if move_count > 300:
            game.headers["Result"] = "1/2-1/2"
            game.headers["Termination"] = "Move limit reached"
            return "Draw (move limit)", 0

    result = board.result()
    game.headers["Result"] = result

    if result == "1-0":
        winner = 1
    elif result == "0-1":
        winner = -1
    else:
        winner = 0

    return result, winner


def run_match(model_path: str, stockfish_path: str, num_games: int = 100,
              time_limit: float = 1.0, search_depth: int = 3,
              games_file: str = "match_games.pgn", current_game_file: str = "current_game.pgn"):
    """
    Run a match between the neural network and Stockfish.

    Args:
        model_path: Path to the trained model
        stockfish_path: Path to Stockfish executable
        num_games: Number of games to play
        time_limit: Time limit per move for Stockfish
        search_depth: Search depth for NN player
        games_file: Path to save all completed games (PGN)
        current_game_file: Path to save current game in progress (PGN)
    """
    print("=" * 70)
    print("MATCH: Neural Network vs Stockfish")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"NN search depth: {search_depth} ply")
    print(f"Stockfish: skill level 0, {time_limit}s per move")
    print(f"Games: {num_games}")
    print(f"All games saved to: {games_file}")
    print(f"Current game saved to: {current_game_file}")
    print("=" * 70)

    # Initialize players
    nn_player = NNPlayer(model_path, search_depth=search_depth)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Set Stockfish to skill level 0
    engine.configure({"Skill Level": 1})

    # Statistics
    nn_wins = 0
    sf_wins = 0
    draws = 0
    nn_white_wins = 0
    nn_black_wins = 0
    sf_white_wins = 0
    sf_black_wins = 0

    start_time = time.time()

    # Open file for all games (append mode)
    with open(games_file, 'w') as all_games_f:
        all_games_f.write(f"; Match: Neural Network vs Stockfish (Skill 0)\n")
        all_games_f.write(f"; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        all_games_f.write(f"; Model: {model_path}\n")
        all_games_f.write(f"; NN Search Depth: {search_depth} ply\n")
        all_games_f.write(f"; Stockfish Time: {time_limit}s per move\n\n")
        all_games_f.flush()

    try:
        for game_num in range(1, num_games + 1):
            # Clear cache between games
            nn_player.clear_cache()

            # Alternate colors
            nn_is_white = (game_num % 2 == 1)

            game_start = time.time()

            # Open current game file for this game (overwritten each move)
            with open(current_game_file, 'w') as current_f:
                result, winner = play_game(nn_player, engine, nn_is_white,
                                           game_num, current_f, time_limit)

            game_time = time.time() - game_start

            # Append completed game to all games file
            with open(current_game_file, 'r') as current_f:
                game_pgn = current_f.read()
            with open(games_file, 'a') as all_games_f:
                all_games_f.write(game_pgn)
                all_games_f.write("\n")
                all_games_f.flush()

            # Update statistics
            if nn_is_white:
                if winner == 1:
                    nn_wins += 1
                    nn_white_wins += 1
                    outcome = "NN wins"
                elif winner == -1:
                    sf_wins += 1
                    sf_black_wins += 1
                    outcome = "SF wins"
                else:
                    draws += 1
                    outcome = "Draw"
                color_info = "NN=White"
            else:
                if winner == -1:
                    nn_wins += 1
                    nn_black_wins += 1
                    outcome = "NN wins"
                elif winner == 1:
                    sf_wins += 1
                    sf_white_wins += 1
                    outcome = "SF wins"
                else:
                    draws += 1
                    outcome = "Draw"
                color_info = "NN=Black"

            print(f"Game {game_num:3d}/{num_games}: {result:12s} ({color_info}) - {outcome:8s} "
                  f"[{game_time:.1f}s, {nn_player.nodes_evaluated:,} nodes]")

    finally:
        engine.quit()

    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("MATCH RESULTS")
    print("=" * 70)
    print(f"Neural Network: {nn_wins} wins ({nn_wins/num_games*100:.1f}%)")
    print(f"  As White: {nn_white_wins}")
    print(f"  As Black: {nn_black_wins}")
    print(f"\nStockfish:      {sf_wins} wins ({sf_wins/num_games*100:.1f}%)")
    print(f"  As White: {sf_white_wins}")
    print(f"  As Black: {sf_black_wins}")
    print(f"\nDraws:          {draws} ({draws/num_games*100:.1f}%)")
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Games saved to: {games_file}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Configuration
    MODEL_PATH = "chess_model_epoch070.pt"
    STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # Adjust for your system
    NUM_GAMES = 100
    TIME_LIMIT = 5.0  # seconds per move for Stockfish
    SEARCH_DEPTH = 3  # ply for NN player

    # Check for command line arguments
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        STOCKFISH_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        NUM_GAMES = int(sys.argv[3])

    run_match(MODEL_PATH, STOCKFISH_PATH, NUM_GAMES, TIME_LIMIT, SEARCH_DEPTH)
