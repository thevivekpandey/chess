"""
Match: Neural Network (3-ply search) vs Stockfish (skill level 0, 1s/move)
Plays 100 games and reports results.
"""

import chess
import chess.pgn
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, TextIO
from datetime import datetime
import time

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
# MINIMAX SEARCH WITH ALPHA-BETA PRUNING
# =============================================================================

class NNPlayer:
    """Neural network player with minimax search."""

    def __init__(self, model_path: str, device: str = 'auto', search_depth: int = 3):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=8)

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.search_depth = search_depth
        self.nodes_evaluated = 0

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using neural network.
        Returns score from white's perspective (positive = white better).
        """
        self.nodes_evaluated += 1
        fen = board.fen()
        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.model(board_tensor).item()

        return score

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax search with alpha-beta pruning.

        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player (white)

        Returns:
            Position evaluation
        """
        # Terminal conditions
        if board.is_checkmate():
            # Prefer faster mates by adding depth bonus
            # Higher depth = found checkmate earlier = faster mate
            # Winning side wants fast mates (higher score), losing side wants slow mates
            return (-1.0 - depth * 0.01) if maximizing else (1.0 + depth * 0.01)
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0

        # Leaf node - use neural network
        if depth == 0:
            return self.evaluate(board)

        legal_moves = list(board.legal_moves)

        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Find the best move using minimax search."""
        self.nodes_evaluated = 0
        best_move = None
        maximizing = board.turn == chess.WHITE

        if maximizing:
            best_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, self.search_depth - 1, -float('inf'), float('inf'), False)
                board.pop()
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
        else:
            best_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, self.search_depth - 1, -float('inf'), float('inf'), True)
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

        if board.turn == chess.WHITE:
            if nn_is_white:
                move = nn_player.get_best_move(board)
            else:
                result = engine.play(board, chess.engine.Limit(time=time_limit))
                move = result.move
        else:
            if nn_is_white:
                result = engine.play(board, chess.engine.Limit(time=time_limit))
                move = result.move
            else:
                move = nn_player.get_best_move(board)

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
    TIME_LIMIT = 10.0  # seconds per move for Stockfish
    SEARCH_DEPTH = 3  # ply for NN player

    # Check for command line arguments
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        STOCKFISH_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        NUM_GAMES = int(sys.argv[3])

    run_match(MODEL_PATH, STOCKFISH_PATH, NUM_GAMES, TIME_LIMIT, SEARCH_DEPTH)
