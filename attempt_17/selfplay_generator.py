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
import cython_chess
import numpy as np
import torch

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
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
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

    # advance_root and make_root both maintain mcts_root.is_terminal, so the
    # loop guard can use it directly instead of paying for python-chess's
    # is_game_over(claim_draw=True) each ply (which iterates legal moves to
    # detect threefold via push/pop).
    mcts_root: MCTSNode = neural.make_root(board)
    position_history: List[str] = [_position_key(board)]

    training_examples = []
    pgn_node = game
    ply = 0

    while ply < max_plies and not mcts_root.is_terminal:
        # Run MCTS search (ignore the returned move, we'll sample it ourselves)
        _, search_stats = neural.choose_move(
            mcts_root,
            n_sims=n_simulations,
            batch_size=mcts_batch_size,
            cpuct=cpuct,
            fpu_reduction=fpu_reduction,
            game_history=position_history,
            early_exit_min_sims=early_exit_min_sims,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

        # Extract visit distribution from top_moves
        top_moves = search_stats['top_moves']  # List[(move, visits, prior, q_white)]

        # Build visit distribution for all legal moves
        move_visits_dict = {mv: visits for mv, visits, _, _ in top_moves}

        # Get all legal moves and their visit counts
        legal_moves = list(cython_chess.generate_legal_moves(board, chess.BB_ALL, chess.BB_ALL))
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

        # Refresh root so the next iteration's loop guard sees correct
        # is_terminal status. advance_root reuses the subtree; otherwise
        # rebuild from scratch.
        if reuse_tree:
            mcts_root = neural.advance_root(mcts_root, move, position_history)
        else:
            mcts_root = neural.make_root(board)

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
        Updated examples with 'value_target' added (from white's perspective,
        matching M0's supervised training convention and how MCTS interprets
        the value head — see play_games_mcts.py:640).
    """
    if game_result == "1-0":
        white_outcome = 1.0
    elif game_result == "0-1":
        white_outcome = -1.0
    else:
        white_outcome = 0.0  # draw or unknown

    for example in training_examples:
        example['value_target'] = white_outcome

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


# =============================================================================
# Multiprocessing workers for parallel self-play
# =============================================================================
#
# Each worker process loads its own MCTSEngine (its own model copy on the
# target device). Games are independent, so we just `imap_unordered` over them
# and write CSV from the main process to keep the file write serialized.

_WORKER_NEURAL: Optional[MCTSEngine] = None
_WORKER_PLAY_PARAMS: Optional[Dict] = None


def _format_moves_san(game: chess.pgn.Game) -> str:
    """Render a finished pgn Game as '1. e4 e5 2. Nf3 Nc6 ...'."""
    moves_san = []
    node = game
    while node.variations:
        next_node = node.variation(0)
        moves_san.append(node.board().san(next_node.move))
        node = next_node
    parts = []
    for i in range(0, len(moves_san), 2):
        move_num = (i // 2) + 1
        if i + 1 < len(moves_san):
            parts.append(f"{move_num}. {moves_san[i]} {moves_san[i+1]}")
        else:
            parts.append(f"{move_num}. {moves_san[i]}")
    return " ".join(parts)


def _init_selfplay_worker(
    model_path: str,
    device: str,
    eval_batch_size: int,
    play_params: Dict,
    base_seed: int,
):
    """Per-worker initialization for parallel self-play."""
    global _WORKER_NEURAL, _WORKER_PLAY_PARAMS

    # Cap torch intra-op threads so N workers don't all grab all 24 cores.
    torch.set_num_threads(1)

    # Decorrelate temperature sampling across workers — without this, every
    # worker would draw the same opening line from the same RNG state.
    pid = os.getpid()
    seed = (base_seed * 2654435761 + pid) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _WORKER_NEURAL = MCTSEngine(
        model_path,
        device=device,
        eval_batch_size=eval_batch_size,
        verbose=False,
    )
    _WORKER_PLAY_PARAMS = play_params


def _run_one_selfplay_game_in_worker(game_idx: int):
    """Play one self-play game; return data the main process needs for CSV/log."""
    global _WORKER_NEURAL, _WORKER_PLAY_PARAMS

    start = time.time()
    game, training_examples = play_selfplay_game(
        neural=_WORKER_NEURAL,
        **_WORKER_PLAY_PARAMS,
    )
    elapsed = time.time() - start

    result = game.headers["Result"]
    training_examples = assign_game_outcome_to_examples(training_examples, result)

    return game_idx, result, training_examples, elapsed


def generate_selfplay_games(
    model_path: str,
    num_games: int,
    output_csv: str,
    n_simulations: int = 800,
    mcts_batch_size: int = 64,
    cpuct: float = 2.0,
    fpu_reduction: float = 0.0,
    reuse_tree: bool = True,
    max_plies: int = 300,
    temperature_moves: int = 30,
    device: str = "cpu",
    eval_batch_size: int = 256,
    verbose: bool = False,
    early_exit_min_sims: int = 0,
    parallel_games: int = 1,
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
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
        eval_batch_size: Inner forward-pass chunk size (>= mcts_batch_size for
            no chunking).
        verbose: Print move-by-move output (sequential mode only — interleaved
            output from parallel workers would be unreadable).
        early_exit_min_sims: Early exit threshold
        parallel_games: Number of worker processes for concurrent self-play.
            1 = run in the main process (no fork overhead).
        dirichlet_alpha: Alpha for Dirichlet noise
        dirichlet_epsilon: Epsilon for Dirichlet noise
    """
    n_workers = max(1, min(parallel_games, num_games))

    print(f"Generating {num_games} self-play games...")
    print(f"MCTS config: simulations={n_simulations}, batch={mcts_batch_size}, cpuct={cpuct}")
    if dirichlet_epsilon > 0:
        print(f"Exploration: dirichlet_alpha={dirichlet_alpha}, dirichlet_epsilon={dirichlet_epsilon}")
    print(f"Parallel workers: {n_workers}")
    print(f"Output: {output_csv}")
    print()

    total_positions = 0
    start_time = time.time()

    play_params = dict(
        n_simulations=n_simulations,
        mcts_batch_size=mcts_batch_size,
        cpuct=cpuct,
        fpu_reduction=fpu_reduction,
        reuse_tree=reuse_tree,
        max_plies=max_plies,
        temperature_moves=temperature_moves,
        early_exit_min_sims=early_exit_min_sims,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    if n_workers == 1:
        # Sequential path: avoid pool/spawn overhead.
        print(f"Initializing neural engine from {model_path}...")
        neural = MCTSEngine(
            model_path, device=device, eval_batch_size=eval_batch_size,
        )

        for game_idx in range(num_games):
            game_start = time.time()

            if verbose:
                print(f"\n=== Game {game_idx + 1}/{num_games} ===")

            game, training_examples = play_selfplay_game(
                neural=neural,
                verbose=verbose,
                **play_params,
            )

            result = game.headers["Result"]
            training_examples = assign_game_outcome_to_examples(training_examples, result)

            game_id = f"game_{game_idx:05d}"
            save_training_examples_to_csv(training_examples, output_csv, game_id)

            game_elapsed = time.time() - game_start
            total_positions += len(training_examples)

            n_moves = len(training_examples)
            mps = n_moves / game_elapsed if game_elapsed > 0 else 0.0
            print(f"Game {game_idx + 1}/{num_games}: {result}, "
                  f"{n_moves} positions, {game_elapsed:.1f}s, {mps:.1f} moves/s")
    else:
        # Parallel path: workers each load their own MCTSEngine, play games
        # concurrently, return examples; main process serializes CSV writes.
        # spawn (not fork) is required for CUDA.
        play_params["verbose"] = False  # workers must not print per-move
        ctx = mp.get_context("spawn")
        base_seed = int(time.time_ns() & 0xFFFFFFFF)

        completed = 0
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_selfplay_worker,
            initargs=(model_path, device, eval_batch_size, play_params, base_seed),
        ) as pool:
            for game_idx, result, training_examples, game_elapsed in \
                    pool.imap_unordered(
                        _run_one_selfplay_game_in_worker, range(num_games)
                    ):
                game_id = f"game_{game_idx:05d}"
                save_training_examples_to_csv(training_examples, output_csv, game_id)

                completed += 1
                total_positions += len(training_examples)

                n_moves = len(training_examples)
                mps = n_moves / game_elapsed if game_elapsed > 0 else 0.0
                print(f"[{completed}/{num_games}] Game {game_idx + 1}: {result}, "
                      f"{n_moves} positions, {game_elapsed:.1f}s, {mps:.1f} moves/s")

    total_elapsed = time.time() - start_time
    overall_mps = total_positions / total_elapsed if total_elapsed > 0 else 0.0
    print(f"\nCompleted {num_games} games in {total_elapsed:.1f}s")
    print(f"Total positions: {total_positions}")
    print(f"Average: {total_positions/num_games:.0f} positions/game, "
          f"{total_elapsed/num_games:.1f}s/game wall, "
          f"{overall_mps:.1f} moves/s overall")
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
        default=64,
        help="MCTS leaf-parallel batch size (default: 64)"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="NN forward-pass chunk size (default: 256)"
    )
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=24,
        help="Number of worker processes for concurrent self-play (default: 24)"
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
    parser.add_argument(
        "--early-exit-min-sims",
        type=int,
        default=200,
        help="Minimum simulations before early exit (default: 200)"
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.0,
        help="Alpha for Dirichlet noise (default: 0.0, off)"
    )
    parser.add_argument(
        "--dirichlet-epsilon",
        type=float,
        default=0.0,
        help="Epsilon for Dirichlet noise (default: 0.0, off)"
    )

    args = parser.parse_args()

    generate_selfplay_games(
        model_path=args.model,
        num_games=args.games,
        output_csv=args.output,
        n_simulations=args.simulations,
        mcts_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        cpuct=args.cpuct,
        temperature_moves=args.temperature_moves,
        device=args.device,
        verbose=args.verbose,
        early_exit_min_sims=args.early_exit_min_sims,
        parallel_games=args.parallel_games,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
    )


if __name__ == "__main__":
    main()
