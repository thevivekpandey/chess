"""
Policy-Only Game Generator with Stockfish Labeling

Generates training data by:
1. Playing games using only the policy network (no MCTS)
2. Labeling each position with Stockfish multi-PV and evaluation
3. Converting SF output to policy and value targets

This is much faster than MCTS-based generation (~10ms vs 3-5s per move)
and provides diverse training data from a strong teacher.
"""

import chess
import chess.engine
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from chess_engine import ChessNet, fen_to_tensor, policy_index_to_move, get_legal_move_indices


@dataclass
class TrainingExample:
    """Single training example with FEN and targets"""
    fen: str
    policy_target: List[float]  # Length 4672 (73*64) - probability distribution
    value_target: float  # -1 to +1


def softmax_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature and softmax to logits"""
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def sample_move_from_policy(board: chess.Board, policy_probs: np.ndarray, temperature: float = 0.8) -> chess.Move:
    """
    Sample a legal move from policy network probabilities.

    Args:
        board: Current board position
        policy_probs: Policy network output (4672 probabilities)
        temperature: Sampling temperature (higher = more random)

    Returns:
        Sampled chess move
    """
    legal_moves = list(board.legal_moves)
    legal_indices = get_legal_move_indices(board, legal_moves)

    # Get probabilities for legal moves
    legal_probs = np.array([policy_probs[idx] for idx in legal_indices])

    # Apply temperature and renormalize
    if temperature > 0:
        legal_probs = softmax_temperature(np.log(legal_probs + 1e-10), temperature)

    # Sample move
    move_idx = np.random.choice(len(legal_moves), p=legal_probs)
    return legal_moves[move_idx]


def play_game_with_policy(model: ChessNet, device: torch.device, temperature: float = 0.8, max_moves: int = 200) -> List[str]:
    """
    Play a single game using only the policy network.

    Args:
        model: Neural network model
        device: Device to run model on
        temperature: Sampling temperature
        max_moves: Maximum moves before declaring draw

    Returns:
        List of FEN strings for each position in the game
    """
    board = chess.Board()
    positions = []

    model.eval()
    with torch.no_grad():
        while not board.is_game_over() and len(board.move_stack) < max_moves:
            positions.append(board.fen())

            # Get policy from network
            board_tensor = fen_to_tensor(board.fen()).unsqueeze(0).to(device)
            policy_logits, _ = model(board_tensor)
            policy_probs = torch.softmax(policy_logits[0], dim=0).cpu().numpy()

            # Sample move
            move = sample_move_from_policy(board, policy_probs, temperature)
            board.push(move)

    return positions


def get_stockfish_labels(fen: str, sf_path: str, depth: int = 20, multipv: int = 5) -> Tuple[List[Tuple[str, int]], int]:
    """
    Get Stockfish labels for a position.

    Args:
        fen: Position in FEN format
        sf_path: Path to Stockfish binary
        depth: Search depth
        multipv: Number of principal variations

    Returns:
        (policy_moves, value) where:
        - policy_moves: List of (move_uci, score_cp) tuples
        - value: Position evaluation in centipawns
    """
    with chess.engine.SimpleEngine.popen_uci(sf_path) as engine:
        board = chess.Board(fen)

        # Get multi-PV analysis
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)

        policy_moves = []
        for pv_info in info:
            move = pv_info['pv'][0]
            score = pv_info['score'].relative  # From side-to-move perspective (matches foundation data)

            # Convert score to centipawns
            if score.is_mate():
                # Mate scores: cap at +/- 10000 cp
                mate_in = score.mate()
                score_cp = 10000 - abs(mate_in) if mate_in > 0 else -10000 + abs(mate_in)
            else:
                score_cp = score.score()

            policy_moves.append((move.uci(), int(score_cp)))

        # Value is the score of the best move (in centipawns, from side-to-move perspective)
        value_cp = policy_moves[0][1] if policy_moves else 0

        return policy_moves, int(value_cp)


def centipawns_to_value(cp: float) -> float:
    """
    Convert centipawns to value in [-1, +1] range.
    Uses tanh scaling: value = tanh(cp / 400)

    This matches typical chess engine scaling where:
    - 400 cp advantage ≈ 76% win probability
    - 800 cp advantage ≈ 96% win probability
    """
    return np.tanh(cp / 400.0)


def policy_moves_to_target(board: chess.Board, policy_moves: List[Tuple[str, float]]) -> np.ndarray:
    """
    Convert Stockfish multi-PV output to policy target distribution.

    Strategy: Use softmax over move scores to create probability distribution.
    Temperature tuned so top move gets ~40-60% probability.

    Args:
        board: Chess board position
        policy_moves: List of (move_uci, score_cp) tuples from SF

    Returns:
        Policy target array of length 4672
    """
    policy_target = np.zeros(4672, dtype=np.float32)

    if not policy_moves:
        return policy_target

    # Build move -> score mapping
    move_scores = {}
    for move_uci, score_cp in policy_moves:
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                move_scores[move] = score_cp
        except:
            continue

    if not move_scores:
        return policy_target

    # Get all legal moves and their indices
    legal_moves = list(board.legal_moves)
    legal_indices = get_legal_move_indices(board, legal_moves)

    # Assign scores: SF moves get their scores, other legal moves get worst_score - 100
    scores = []
    worst_score = min(move_scores.values())
    for move in legal_moves:
        if move in move_scores:
            scores.append(move_scores[move])
        else:
            scores.append(worst_score - 100)  # Penalize moves not in SF top-N

    # Convert to probabilities with softmax (temperature = 100 cp)
    # This makes ~100 cp difference = 2.7x probability ratio
    scores = np.array(scores, dtype=np.float32)
    probs = softmax_temperature(scores / 100.0, temperature=1.0)

    # Fill policy target
    for move, idx, prob in zip(legal_moves, legal_indices, probs):
        policy_target[idx] = prob

    return policy_target


@dataclass
class SFLabel:
    """Stockfish label for a position (raw format for CSV output)"""
    fen: str
    eval_cp: int  # Centipawns evaluation
    moves: List[Tuple[str, int]]  # List of (move_uci, score_cp) tuples


def label_position(fen: str, sf_path: str, depth: int = 20, multipv: int = 5) -> Tuple[TrainingExample, SFLabel]:
    """
    Label a single position with Stockfish.

    Args:
        fen: Position in FEN format
        sf_path: Path to Stockfish binary
        depth: Search depth
        multipv: Number of principal variations

    Returns:
        Tuple of (TrainingExample, SFLabel) - neural targets and raw SF data
    """
    board = chess.Board(fen)

    # Get Stockfish labels
    policy_moves, value_cp = get_stockfish_labels(fen, sf_path, depth, multipv)

    # Convert to neural network targets
    policy_target = policy_moves_to_target(board, policy_moves)
    value_target = centipawns_to_value(value_cp)

    training_example = TrainingExample(
        fen=fen,
        policy_target=policy_target.tolist(),
        value_target=value_target
    )

    sf_label = SFLabel(
        fen=fen,
        eval_cp=value_cp,
        moves=policy_moves
    )

    return training_example, sf_label


def label_positions_batch(positions: List[str], sf_path: str, depth: int = 20, multipv: int = 5) -> Tuple[List[TrainingExample], List[SFLabel]]:
    """Label a batch of positions (for parallel processing)"""
    examples = []
    sf_labels = []
    for fen in positions:
        try:
            example, sf_label = label_position(fen, sf_path, depth, multipv)
            examples.append(example)
            sf_labels.append(sf_label)
        except Exception as e:
            print(f"Error labeling position {fen}: {e}")
            continue
    return examples, sf_labels


def generate_training_data(
    model_path: str,
    output_path: str,
    stockfish_path: str,
    num_games: int = 100,
    num_workers: int = 8,
    temperature: float = 0.8,
    sf_depth: int = 20,
    sf_multipv: int = 5,
    device: str = "mps"
):
    """
    Generate training data with policy-only game play and Stockfish labeling.

    Args:
        model_path: Path to neural network model checkpoint
        output_path: Path to save training data (JSON lines format)
        stockfish_path: Path to Stockfish binary
        num_games: Number of games to generate
        num_workers: Number of parallel workers for SF labeling
        temperature: Sampling temperature for policy network
        sf_depth: Stockfish search depth
        sf_multipv: Number of principal variations from SF
        device: Device to run neural network on
    """
    print(f"Generating training data with Stockfish labeling...")
    print(f"  Games: {num_games}")
    print(f"  Temperature: {temperature}")
    print(f"  SF depth: {sf_depth}, multi-PV: {sf_multipv}")
    print(f"  Workers: {num_workers}")

    # Load model
    device_torch = torch.device(device)
    model = ChessNet()
    checkpoint = torch.load(model_path, map_location=device_torch, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_torch)
    model.eval()

    # Step 1: Generate games using policy network
    print("\nStep 1: Generating games with policy network...")
    all_positions = []

    for game_idx in tqdm(range(num_games), desc="Playing games"):
        positions = play_game_with_policy(model, device_torch, temperature)
        all_positions.extend(positions)

    print(f"Generated {len(all_positions)} positions from {num_games} games")
    print(f"Average positions per game: {len(all_positions) / num_games:.1f}")

    # Step 2: Label positions with Stockfish in parallel
    print("\nStep 2: Labeling positions with Stockfish...")

    # Split positions into batches for parallel processing
    batch_size = max(1, len(all_positions) // (num_workers * 4))
    position_batches = [all_positions[i:i+batch_size] for i in range(0, len(all_positions), batch_size)]

    all_examples = []
    all_sf_labels = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in position_batches:
            future = executor.submit(label_positions_batch, batch, stockfish_path, sf_depth, sf_multipv)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Labeling positions"):
            examples, sf_labels = future.result()
            all_examples.extend(examples)
            all_sf_labels.extend(sf_labels)

    print(f"Successfully labeled {len(all_examples)} positions")

    # Step 3: Save to CSV file (matching attempt_02 format)
    print(f"\nStep 3: Saving to {output_path}...")

    with open(output_path, 'w') as f:
        # Write header
        f.write('fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5\n')

        # Write data
        for sf_label in all_sf_labels:
            # Convert eval to pawns
            eval_pawns = sf_label.eval_cp / 100.0

            # Build row
            row = [sf_label.fen, f"{eval_pawns:.2f}"]

            # Add up to 5 moves with scores
            for i in range(5):
                if i < len(sf_label.moves):
                    move_uci, score_cp = sf_label.moves[i]
                    row.extend([move_uci, str(score_cp)])
                else:
                    row.extend(['', ''])

            f.write(','.join(row) + '\n')

    print(f"✓ Saved {len(all_sf_labels)} training examples to {output_path}")

    # Print statistics
    values = [ex.value_target for ex in all_examples]
    print(f"\nValue statistics:")
    print(f"  Mean: {np.mean(values):.3f}")
    print(f"  Std:  {np.std(values):.3f}")
    print(f"  Min:  {np.min(values):.3f}")
    print(f"  Max:  {np.max(values):.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data with Stockfish labeling")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path for training data")
    parser.add_argument("--stockfish", type=str, required=True, help="Path to Stockfish binary")
    parser.add_argument("--games", type=int, default=100, help="Number of games to generate")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--sf-depth", type=int, default=20, help="Stockfish search depth")
    parser.add_argument("--sf-multipv", type=int, default=5, help="Stockfish multi-PV count")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    generate_training_data(
        model_path=args.model,
        output_path=args.output,
        stockfish_path=args.stockfish,
        num_games=args.games,
        num_workers=args.workers,
        temperature=args.temperature,
        sf_depth=args.sf_depth,
        sf_multipv=args.sf_multipv,
        device=args.device
    )
