"""
Evaluate trained model against Stockfish

Tests model strength by playing games against Stockfish at various levels.
Tracks win/loss/draw statistics and ELO estimation.
"""

import chess
import chess.engine
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import json
import os

from chess_engine import ChessNet, fen_to_tensor, get_legal_move_indices


@dataclass
class GameResult:
    """Result of a single game"""
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: int
    white_player: str
    black_player: str


def get_model_move(model: ChessNet, board: chess.Board, device: torch.device, temperature: float = 0.1) -> chess.Move:
    """
    Get move from model using policy network.

    Args:
        model: Neural network
        board: Current position
        device: Device to run on
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        Selected move
    """
    model.eval()
    with torch.no_grad():
        board_tensor = fen_to_tensor(board.fen()).unsqueeze(0).to(device)
        policy_logits, _ = model(board_tensor)
        policy_probs = torch.softmax(policy_logits[0] / temperature, dim=0).cpu().numpy()

        # Get legal moves and their probabilities
        legal_moves = list(board.legal_moves)
        legal_indices = get_legal_move_indices(board, legal_moves)
        legal_probs = np.array([policy_probs[idx] for idx in legal_indices])

        # Renormalize
        legal_probs = legal_probs / legal_probs.sum()

        # Select best move (argmax for low temperature)
        if temperature < 0.2:
            move_idx = np.argmax(legal_probs)
        else:
            move_idx = np.random.choice(len(legal_moves), p=legal_probs)

        return legal_moves[move_idx]


def play_game_vs_stockfish(
    model: ChessNet,
    stockfish_path: str,
    device: torch.device,
    model_color: chess.Color,
    stockfish_level: int = 10,
    stockfish_time: float = 0.1,
    model_temperature: float = 0.1,
    max_moves: int = 200
) -> GameResult:
    """
    Play a game between model and Stockfish.

    Args:
        model: Neural network model
        stockfish_path: Path to Stockfish binary
        device: Device to run model on
        model_color: chess.WHITE or chess.BLACK
        stockfish_level: Stockfish skill level (0-20)
        stockfish_time: Time per move for Stockfish (seconds)
        model_temperature: Sampling temperature for model
        max_moves: Maximum moves before declaring draw

    Returns:
        GameResult with outcome
    """
    board = chess.Board()

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf_engine:
        # Set Stockfish level
        sf_engine.configure({"Skill Level": stockfish_level})

        while not board.is_game_over() and len(board.move_stack) < max_moves:
            if board.turn == model_color:
                # Model's turn
                move = get_model_move(model, board, device, model_temperature)
            else:
                # Stockfish's turn
                result = sf_engine.play(board, chess.engine.Limit(time=stockfish_time))
                move = result.move

            board.push(move)

    # Determine result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            result_str = "0-1"  # Black won
        else:
            result_str = "1-0"  # White won
    else:
        result_str = "1/2-1/2"  # Draw

    white_player = "Model" if model_color == chess.WHITE else f"SF_L{stockfish_level}"
    black_player = f"SF_L{stockfish_level}" if model_color == chess.WHITE else "Model"

    return GameResult(
        result=result_str,
        moves=len(board.move_stack),
        white_player=white_player,
        black_player=black_player
    )


def evaluate_model(
    model_path: str,
    stockfish_path: str,
    num_games: int = 100,
    stockfish_level: int = 10,
    stockfish_time: float = 0.1,
    model_temperature: float = 0.1,
    device: str = "mps"
) -> Dict:
    """
    Evaluate model against Stockfish.

    Args:
        model_path: Path to model checkpoint
        stockfish_path: Path to Stockfish binary
        num_games: Number of games to play
        stockfish_level: Stockfish skill level (0-20)
        stockfish_time: Time per move for Stockfish
        model_temperature: Model sampling temperature
        device: Device to run model on

    Returns:
        Dictionary with evaluation statistics
    """
    print("=" * 80)
    print("EVALUATING MODEL VS STOCKFISH")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Stockfish level: {stockfish_level}")
    print(f"Stockfish time: {stockfish_time}s per move")
    print(f"Games: {num_games}")
    print(f"Model temperature: {model_temperature}")
    print()

    # Load model
    device_torch = torch.device(device)
    model = ChessNet()
    checkpoint = torch.load(model_path, map_location=device_torch, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_torch)
    model.eval()

    # Play games (alternating colors)
    results = []
    wins = 0
    losses = 0
    draws = 0

    for game_idx in tqdm(range(num_games), desc="Playing games"):
        # Alternate colors
        model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK

        result = play_game_vs_stockfish(
            model, stockfish_path, device_torch,
            model_color, stockfish_level, stockfish_time, model_temperature
        )

        results.append(result)

        # Count results from model's perspective
        if result.result == "1-0" and model_color == chess.WHITE:
            wins += 1
        elif result.result == "0-1" and model_color == chess.BLACK:
            wins += 1
        elif result.result == "1/2-1/2":
            draws += 1
        else:
            losses += 1

    # Calculate statistics
    total_games = len(results)
    win_rate = wins / total_games
    draw_rate = draws / total_games
    loss_rate = losses / total_games
    score = (wins + 0.5 * draws) / total_games

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total games: {total_games}")
    print(f"Wins:   {wins} ({win_rate:.1%})")
    print(f"Draws:  {draws} ({draw_rate:.1%})")
    print(f"Losses: {losses} ({loss_rate:.1%})")
    print(f"Score:  {score:.3f} ({score:.1%})")
    print()

    # Estimate ELO difference (rough approximation)
    # ELO difference ≈ -400 * log10(1/score - 1)
    if score > 0.01 and score < 0.99:
        elo_diff = -400 * np.log10(1/score - 1)
        print(f"Estimated ELO difference: {elo_diff:+.0f}")
        print(f"(Model ELO ≈ Stockfish L{stockfish_level} ELO {elo_diff:+.0f})")
    else:
        print("Score too extreme for reliable ELO estimate")

    return {
        'total_games': total_games,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'score': score,
        'stockfish_level': stockfish_level,
        'stockfish_time': stockfish_time
    }


def evaluate_model_progressive(
    model_path: str,
    stockfish_path: str,
    games_per_level: int = 50,
    device: str = "mps"
):
    """
    Evaluate model against multiple Stockfish levels to estimate strength.

    Args:
        model_path: Path to model checkpoint
        stockfish_path: Path to Stockfish binary
        games_per_level: Number of games per level
        device: Device to run model on
    """
    print("=" * 80)
    print("PROGRESSIVE EVALUATION VS STOCKFISH")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Games per level: {games_per_level}")
    print()

    levels = [1, 5, 10, 15, 20]
    all_results = []

    for level in levels:
        print(f"\n{'=' * 80}")
        print(f"TESTING VS STOCKFISH LEVEL {level}")
        print(f"{'=' * 80}")

        stats = evaluate_model(
            model_path=model_path,
            stockfish_path=stockfish_path,
            num_games=games_per_level,
            stockfish_level=level,
            stockfish_time=0.1,
            device=device
        )

        all_results.append({
            'level': level,
            **stats
        })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Level':<10} {'Score':<10} {'W':<6} {'D':<6} {'L':<6}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['level']:<10} {result['score']:<10.3f} {result['wins']:<6} {result['draws']:<6} {result['losses']:<6}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model against Stockfish")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stockfish", type=str, required=True, help="Path to Stockfish binary")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--level", type=int, default=10, help="Stockfish level (0-20)")
    parser.add_argument("--sf-time", type=float, default=0.1, help="Stockfish time per move")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu/cuda/mps)")
    parser.add_argument("--progressive", action="store_true", help="Test against multiple levels")

    args = parser.parse_args()

    if args.progressive:
        evaluate_model_progressive(
            model_path=args.model,
            stockfish_path=args.stockfish,
            games_per_level=args.games,
            device=args.device
        )
    else:
        evaluate_model(
            model_path=args.model,
            stockfish_path=args.stockfish,
            num_games=args.games,
            stockfish_level=args.level,
            stockfish_time=args.sf_time,
            model_temperature=args.temperature,
            device=args.device
        )
