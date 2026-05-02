#!/usr/bin/env python3
"""
Generate self-play games for training using MCTS.

Plays games where both sides use the neural engine with MCTS, collecting
training examples in the AlphaZero format:
  - Position (FEN)
  - MCTS visit distribution (policy target)
  - Game outcome (value target)

Saves training data to CSV files for later training.
"""

import argparse
import csv
import multiprocessing as mp
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn
import numpy as np

# Import MCTS engine from play_games_mcts
from play_games_mcts import (
    MCTSEngine,
    MCTSNode,
    _position_key,
)


def sample_move_with_temperature(
    moves: List[chess.Move],
    visit_counts: List[int],
    temperature: float = 1.0,
) -> chess.Move:
    """
    Sample a move based on visit counts with temperature.

    Args:
        moves: List of legal moves
        visit_counts: Visit count for each move
        temperature: Temperature parameter
            - temperature = 0: Pick move with highest visits (greedy)
            - temperature = 1: Sample proportional to visits
            - temperature > 1: More uniform sampling

    Returns:
        Selected move
    """
    if temperature == 0 or len(moves) == 1:
        # Greedy: pick move with highest visit count
        max_visits = max(visit_counts)
        best_moves = [moves[i] for i, v in enumerate(visit_counts) if v == max_visits]
        return random.choice(best_moves)  # Tie-break randomly

    # Apply temperature
    visits_array = np.array(visit_counts, dtype=np.float64)
    if temperature != 1.0:
        visits_array = visits_array ** (1.0 / temperature)

    # Normalize to probabilities
    total = visits_array.sum()
    if total == 0:
        # Uniform if no visits
        probs = np.ones(len(moves)) / len(moves)
    else:
        probs = visits_array / total

    # Sample
    return np.random.choice(moves, p=probs)


def play_selfplay_game(
    neural: MCTSEngine,
    n_simulations: int,
    mcts_batch_size: int,
    cpuct: float,
    fpu_reduction: float,
    reuse_tree: bool,
    max_plies: int,
    temperature_moves: int = 30,
    verbose: bool = False,
    early_exit_min_sims: int = 0,
) -> Tuple[chess.pgn.Game, List[Dict]]:
    """
    Play a self-play game where both sides use the neural engine.

    Args:
        temperature_moves: Number of opening moves to use temperature=1 sampling
                          (default: 30). After this, uses greedy selection.

    Returns:
        - chess.pgn.Game: The completed game
        - List of training examples: [{
            'fen': str,
            'move_uci_list': List[str],  # All legal moves in UCI
            'visit_counts': List[int],    # Visit count for each move
            'turn': chess.Color,          # Whose turn it was
          }, ...]
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play Training"
    game.headers["Date"] = datetime.utcnow().strftime("%Y.%m.%d")
    game.headers["White"] = "NeuralEngine"
    game.headers["Black"] = "NeuralEngine"

    mcts_root: Optional[MCTSNode] = neural.make_root(board) if reuse_tree else None
    position_history: List[str] = [_position_key(board)]

    training_examples = []
    pgn_node = game
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        # Make root if needed
        if not reuse_tree or mcts_root is None:
            mcts_root = neural.make_root(board)

        # Run MCTS search (ignore the returned move, we'll sample it ourselves)
        _, search_stats = neural.choose_move(
            mcts_root,
            n_sims=n_simulations,
            batch_size=mcts_batch_size,
            cpuct=cpuct,
            fpu_reduction=fpu_reduction,
            game_history=position_history,
            early_exit_min_sims=early_exit_min_sims,
        )

        # Extract visit distribution from top_moves
        top_moves = search_stats['top_moves']  # List[(move, visits, prior, q_white)]

        # Build visit distribution for all legal moves
        move_visits_dict = {mv: visits for mv, visits, _, _ in top_moves}

        # Get all legal moves and their visit counts
        legal_moves = list(board.legal_moves)
        move_uci_list = [mv.uci() for mv in legal_moves]
        visit_counts = [move_visits_dict.get(mv, 0) for mv in legal_moves]

        # Collect training example BEFORE making the move
        training_example = {
            'fen': board.fen(),
            'move_uci_list': move_uci_list,
            'visit_counts': visit_counts,
            'turn': board.turn,
        }
        training_examples.append(training_example)

        # Sample move with temperature
        # Use temperature=1 for first N moves, then greedy (temperature=0)
        temperature = 1.0 if ply < temperature_moves else 0.0
        move = sample_move_with_temperature(legal_moves, visit_counts, temperature)

        if verbose:
            san = board.san(move)
            move_prefix = (
                f"{board.fullmove_number}. " if board.turn == chess.WHITE
                else f"{board.fullmove_number}..."
            )
            print(f"{move_prefix}{san}")

        # Make the move
        board.push(move)
        position_history.append(_position_key(board))

        # Advance tree for reuse
        if reuse_tree and mcts_root is not None:
            mcts_root = neural.advance_root(mcts_root, move, position_history)

        pgn_node = pgn_node.add_variation(move)
        ply += 1

    # Set game result
    result = board.result(claim_draw=True)
    game.headers["Result"] = result

    return game, training_examples


def assign_game_outcome_to_examples(
    training_examples: List[Dict],
    game_result: str,
) -> List[Dict]:
    """
    Assign the game outcome to all training examples.

    Args:
        training_examples: List of examples from the game
        game_result: "1-0", "0-1", or "1/2-1/2"

    Returns:
        Updated examples with 'value_target' added (from current player's perspective)
    """
    # Parse result
    if game_result == "1-0":
        white_outcome = 1.0
        black_outcome = -1.0
    elif game_result == "0-1":
        white_outcome = -1.0
        black_outcome = 1.0
    elif game_result == "1/2-1/2":
        white_outcome = 0.0
        black_outcome = 0.0
    else:
        # Unknown result, treat as draw
        white_outcome = 0.0
        black_outcome = 0.0

    for example in training_examples:
        # Value target from current player's perspective
        if example['turn'] == chess.WHITE:
            example['value_target'] = white_outcome
        else:
            example['value_target'] = black_outcome

    return training_examples


def save_training_examples_to_csv(
    training_examples: List[Dict],
    csv_path: str,
    game_id: str,
):
    """
    Save training examples to CSV file.

    CSV format:
        fen, policy_moves, policy_visits, value_target, game_id
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header if new file
        if not file_exists:
            writer.writerow(['fen', 'policy_moves', 'policy_visits', 'value_target', 'game_id'])

        for example in training_examples:
            # Convert lists to comma-separated strings
            policy_moves = ','.join(example['move_uci_list'])
            policy_visits = ','.join(str(v) for v in example['visit_counts'])
            value_target = example['value_target']

            writer.writerow([
                example['fen'],
                policy_moves,
                policy_visits,
                value_target,
                game_id,
            ])

        csvfile.flush()


def generate_selfplay_games(
    model_path: str,
    num_games: int,
    output_csv: str,
    n_simulations: int = 800,
    mcts_batch_size: int = 16,
    cpuct: float = 2.0,
    fpu_reduction: float = 0.0,
    reuse_tree: bool = True,
    max_plies: int = 300,
    temperature_moves: int = 30,
    device: str = "cpu",
    eval_batch_size: int = 16,
    verbose: bool = False,
    early_exit_min_sims: int = 0,
):
    """
    Generate self-play games and save training data to CSV.

    Args:
        model_path: Path to neural network model checkpoint
        num_games: Number of self-play games to generate
        output_csv: Path to output CSV file
        n_simulations: MCTS simulations per move
        mcts_batch_size: Batch size for MCTS leaf-parallel evaluation
        cpuct: PUCT exploration constant
        fpu_reduction: First-play urgency reduction
        reuse_tree: Whether to reuse MCTS tree across moves
        max_plies: Maximum plies per game
        device: "cpu" or "cuda"
        eval_batch_size: Batch size for NN evaluation
        verbose: Print move-by-move output
        early_exit_min_sims: Early exit threshold
    """
    print(f"Initializing neural engine from {model_path}...")
    neural = MCTSEngine(
        model_path,
        device=device,
        eval_batch_size=eval_batch_size,
    )

    print(f"Generating {num_games} self-play games...")
    print(f"MCTS config: simulations={n_simulations}, batch={mcts_batch_size}, cpuct={cpuct}")
    print(f"Output: {output_csv}")
    print()

    total_positions = 0
    start_time = time.time()

    for game_idx in range(num_games):
        game_start = time.time()

        if verbose:
            print(f"\n=== Game {game_idx + 1}/{num_games} ===")

        # Play one self-play game
        game, training_examples = play_selfplay_game(
            neural=neural,
            n_simulations=n_simulations,
            mcts_batch_size=mcts_batch_size,
            cpuct=cpuct,
            fpu_reduction=fpu_reduction,
            reuse_tree=reuse_tree,
            max_plies=max_plies,
            temperature_moves=temperature_moves,
            verbose=verbose,
            early_exit_min_sims=early_exit_min_sims,
        )

        # Get game result and assign outcomes
        result = game.headers["Result"]
        training_examples = assign_game_outcome_to_examples(training_examples, result)

        # Save to CSV
        game_id = f"game_{game_idx:05d}"
        save_training_examples_to_csv(training_examples, output_csv, game_id)

        game_elapsed = time.time() - game_start
        total_positions += len(training_examples)

        # Extract moves in SAN notation
        moves_san = []
        node = game
        while node.variations:
            next_node = node.variation(0)
            moves_san.append(node.board().san(next_node.move))
            node = next_node

        # Format moves with move numbers (1. e4 e5 2. Nf3 Nc6 ...)
        moves_formatted = []
        for i in range(0, len(moves_san), 2):
            move_num = (i // 2) + 1
            if i + 1 < len(moves_san):
                # Both white and black moves
                moves_formatted.append(f"{move_num}. {moves_san[i]} {moves_san[i+1]}")
            else:
                # Only white's move (game ended after white's move)
                moves_formatted.append(f"{move_num}. {moves_san[i]}")

        # Print game summary with moves
        print(f"Game {game_idx + 1}/{num_games}: {result}, "
              f"{len(training_examples)} positions, {game_elapsed:.1f}s")
        print(f"  Moves: {' '.join(moves_formatted)}")

    total_elapsed = time.time() - start_time
    print(f"\nCompleted {num_games} games in {total_elapsed:.1f}s")
    print(f"Total positions: {total_positions}")
    print(f"Average: {total_positions/num_games:.0f} positions/game, "
          f"{total_elapsed/num_games:.1f}s/game")
    print(f"Training data saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play games for training"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to neural network model checkpoint"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of self-play games to generate (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="selfplay_data.csv",
        help="Output CSV file path (default: selfplay_data.csv)"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="MCTS simulations per move (default: 800)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="MCTS batch size (default: 16)"
    )
    parser.add_argument(
        "--cpuct",
        type=float,
        default=2.0,
        help="PUCT exploration constant (default: 2.0)"
    )
    parser.add_argument(
        "--temperature-moves",
        type=int,
        default=30,
        help="Number of opening moves to use temperature sampling (default: 30)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for neural network (default: cpu)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print move-by-move output"
    )

    args = parser.parse_args()

    generate_selfplay_games(
        model_path=args.model,
        num_games=args.games,
        output_csv=args.output,
        n_simulations=args.simulations,
        mcts_batch_size=args.batch_size,
        cpuct=args.cpuct,
        temperature_moves=args.temperature_moves,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
