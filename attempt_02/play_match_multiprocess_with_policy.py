"""
Multiprocess Match with Policy: Neural Network vs Stockfish
Uses policy predictions to filter candidate moves before searching.
Each process does cloud calls for evaluation + policy, then searches only suggested moves.
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

# Cloud GPU inference endpoint (passed as argument)
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Encoding": "gzip",
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip"
}

OPENING_BOOK_FILE = "opening_book.csv"


# =============================================================================
# STANDALONE EVALUATOR WITH POLICY (one per process)
# =============================================================================

class GameEvaluatorWithPolicy:
    """Evaluator for a single game using policy predictions - runs in its own process."""

    BATCH_SIZE = 4096

    def __init__(self, search_depth: int = 2, topk: int = 10,
                 prob_threshold: float = 1.0, min_moves: int = 1, server_url: str = None):
        """
        Args:
            search_depth: Depth for searching candidate moves
            topk: Number of moves to request from server and explore
            prob_threshold: Stop adding moves when cumulative probability exceeds this (0.0-1.0)
            min_moves: Minimum number of moves to explore regardless of probability
            server_url: URL of the inference server
        """
        self.search_depth = search_depth
        self.topk = topk
        self.prob_threshold = prob_threshold
        self.min_moves = min_moves
        self.server_url = server_url
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

    def evaluate_with_policy_cloud(self, fens: List[str]) -> Dict[str, dict]:
        """
        Send batch of FENs to cloud GPU for evaluation + policy prediction.

        Returns:
            Dict mapping FEN to {'score': float, 'top_moves': [[move_uci, prob], ...]}
        """
        results = {}
        for i in range(0, len(fens), self.BATCH_SIZE):
            batch = fens[i:i + self.BATCH_SIZE]
            payload = gzip.compress(msgpack.packb({"fens": batch, "top_k": self.topk}))

            # Retry loop
            while True:
                try:
                    request_start = time.time()
                    response = requests.post(
                        self.server_url + "/evaluate_with_policy",
                        data=payload,
                        headers=HEADERS,
                        timeout=120
                    )
                    response.raise_for_status()
                    data = response.json()
                    self.cloud_inference_time += time.time() - request_start

                    # Parse response - expects scores and top_moves
                    scores = data["scores"]
                    top_moves = data.get("top_moves", [[] for _ in batch])

                    for j, fen in enumerate(batch):
                        results[fen] = {
                            'score': scores[j],
                            'top_moves': top_moves[j] if j < len(top_moves) else []
                        }
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"  [Cloud error: {e}] Retrying in 5s...", flush=True)
                    time.sleep(5)
        return results

    def evaluate_batch_cloud(self, fens: List[str]) -> Dict[str, float]:
        """Send batch of FENs to cloud GPU for evaluation only (for leaf nodes)."""
        results = {}
        for i in range(0, len(fens), self.BATCH_SIZE):
            batch = fens[i:i + self.BATCH_SIZE]
            payload = gzip.compress(msgpack.packb({"fens": batch}))

            # Retry loop
            while True:
                try:
                    request_start = time.time()
                    # Use standard evaluate endpoint for leaf evaluations
                    eval_url = self.server_url + "/evaluate"
                    response = requests.post(
                        eval_url,
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

    def get_candidate_moves(
        self,
        board: chess.Board,
        top_moves: List[List]
    ) -> List[chess.Move]:
        """
        Get candidate moves based on policy predictions from server.
        Uses probability threshold to adaptively select moves.

        Args:
            board: Current board position
            top_moves: List of [move_uci, probability] from server

        Returns:
            List of candidate moves to search
        """
        candidate_moves = []
        cumulative_prob = 0.0

        for move_data in top_moves:
            if len(move_data) >= 2:
                move_uci, prob = move_data[0], move_data[1]
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        candidate_moves.append(move)
                        cumulative_prob += prob

                        # Stop if we've hit probability threshold (but ensure min_moves)
                        if len(candidate_moves) >= self.min_moves and cumulative_prob >= self.prob_threshold:
                            break

                        # Hard limit
                        if len(candidate_moves) >= self.topk:
                            break
                except ValueError:
                    pass

        # If no candidates found, fall back to all legal moves
        if not candidate_moves:
            candidate_moves = list(board.legal_moves)[:self.topk]

        return candidate_moves

    def minimax_with_policy(
        self,
        board: chess.Board,
        depth: int,
        maximizing: bool,
        policy_cache: Dict[int, List[chess.Move]],
        alpha: float = -float('inf'),
        beta: float = float('inf')
    ) -> float:
        """
        Minimax search with alpha-beta pruning using policy-guided moves.
        """
        if board.is_game_over():
            if board.is_checkmate():
                # Large checkmate value (much larger than any NN eval ~100)
                # Add depth bonus to prefer faster mates
                return (-10000.0 - depth) if maximizing else (10000.0 + depth)
            return 0.0

        z_hash = chess.polyglot.zobrist_hash(board)

        if depth == 0:
            return self.eval_cache.get(z_hash, 0.0)

        # Get moves from policy cache
        moves = policy_cache.get(z_hash, [])
        if not moves:
            # Fallback to cached evaluation if no policy moves
            return self.eval_cache.get(z_hash, 0.0)

        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                board.push(move)
                eval_score = self.minimax_with_policy(board, depth - 1, False, policy_cache, alpha, beta)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval_score = self.minimax_with_policy(board, depth - 1, True, policy_cache, alpha, beta)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def get_best_move(self, board: chess.Board) -> Tuple[chess.Move, dict]:
        """Get best move using policy-guided search at every level."""
        # Check opening book first
        book_move = self.get_opening_book_move(board)
        if book_move:
            return book_move, {'from_book': True, 'positions': 0, 'cloud_time': 0, 'candidates': 0}

        total_start = time.time()
        effective_depth = self.search_depth

        # Profiling accumulators
        time_collect_fens = 0.0
        time_api_calls = 0.0
        time_process_results = 0.0
        time_build_next_level = 0.0

        # policy_cache maps zobrist hash -> list of chess.Move objects
        policy_cache: Dict[int, List[chess.Move]] = {}
        total_positions = 0

        # Process level by level using BFS
        # Use dict to deduplicate by hash: hash -> board
        current_level = {chess.polyglot.zobrist_hash(board): board.copy()}

        for depth in range(effective_depth):
            is_leaf_level = (depth == effective_depth - 1)

            # Collect unique FENs that need querying
            t0 = time.time()
            fens_to_query = []
            hash_by_fen = {}  # fen -> hash
            board_by_hash = {}  # hash -> board (for getting candidate moves)

            for h, b in current_level.items():
                if b.is_game_over():
                    continue
                # Skip if already cached
                if h in self.eval_cache and (is_leaf_level or h in policy_cache):
                    continue

                fen = self.compact_fen(b.fen())
                if fen not in hash_by_fen:
                    fens_to_query.append(fen)
                    hash_by_fen[fen] = h
                    board_by_hash[h] = b
            time_collect_fens += time.time() - t0

            if fens_to_query:
                t0 = time.time()
                if is_leaf_level:
                    # Leaf level: just get evaluations
                    results = self.evaluate_batch_cloud(fens_to_query)
                else:
                    # Non-leaf: get policy + evaluation
                    results = self.evaluate_with_policy_cloud(fens_to_query)
                time_api_calls += time.time() - t0

                t0 = time.time()
                if is_leaf_level:
                    for fen, score in results.items():
                        h = hash_by_fen[fen]
                        self.eval_cache[h] = score
                else:
                    for fen, data in results.items():
                        h = hash_by_fen[fen]
                        b = board_by_hash[h]
                        self.eval_cache[h] = data['score']
                        candidate_moves = self.get_candidate_moves(b, data['top_moves'])
                        policy_cache[h] = candidate_moves
                time_process_results += time.time() - t0

                self.positions_evaluated += len(fens_to_query)
                total_positions += len(fens_to_query)

            if is_leaf_level:
                break

            # Build next level by expanding current level with policy moves
            t0 = time.time()
            next_level = {}
            for h, b in current_level.items():
                if b.is_game_over():
                    continue

                moves = policy_cache.get(h, [])
                for move in moves:
                    child = b.copy()
                    child.push(move)
                    child_hash = chess.polyglot.zobrist_hash(child)
                    if child_hash not in next_level:
                        next_level[child_hash] = child

            current_level = next_level
            time_build_next_level += time.time() - t0

        cloud_time = time_api_calls

        # Get root info
        root_hash = chess.polyglot.zobrist_hash(board)
        root_candidates = policy_cache.get(root_hash, [])
        num_candidates = len(root_candidates)

        # Run minimax with alpha-beta pruning
        t0 = time.time()
        maximizing = board.turn == chess.WHITE
        best_move = None
        move_scores = []  # List of (move_san, score) for all candidates
        alpha = -float('inf')
        beta = float('inf')

        if maximizing:
            best_eval = -float('inf')
            for move in root_candidates:
                move_san = board.san(move)
                board.push(move)
                eval_score = self.minimax_with_policy(board, effective_depth - 1, False, policy_cache, alpha, beta)
                board.pop()
                move_scores.append((move_san, eval_score))
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
            # Sort by score descending for white
            move_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            best_eval = float('inf')
            for move in root_candidates:
                move_san = board.san(move)
                board.push(move)
                eval_score = self.minimax_with_policy(board, effective_depth - 1, True, policy_cache, alpha, beta)
                board.pop()
                move_scores.append((move_san, eval_score))
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
            # Sort by score ascending for black (best for black first)
            move_scores.sort(key=lambda x: x[1])
        time_minimax = time.time() - t0

        # Fallback if no move found (shouldn't happen)
        if best_move is None and root_candidates:
            best_move = root_candidates[0]

        total_time = time.time() - total_start

        stats = {
            'from_book': False,
            'positions': total_positions,
            'cloud_time': cloud_time,
            'depth': effective_depth,
            'candidates': num_candidates,
            'move_scores': move_scores,
            # Profiling data
            'profile': {
                'total': total_time,
                'api': time_api_calls,
                'collect': time_collect_fens,
                'process': time_process_results,
                'build_tree': time_build_next_level,
                'minimax': time_minimax
            }
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
    game_num, nn_is_white, stockfish_path, time_limit, search_depth, skill_level, topk, prob_threshold, min_moves, server_url = args

    # Initialize evaluator (each process gets its own)
    evaluator = GameEvaluatorWithPolicy(
        search_depth=search_depth,
        topk=topk,
        prob_threshold=prob_threshold,
        min_moves=min_moves,
        server_url=server_url
    )

    # Initialize Stockfish engine (each process gets its own)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": skill_level})

    # Initialize game
    board = chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "NN (Policy) vs Stockfish Match"
    pgn_game.headers["Site"] = "Local"
    pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    pgn_game.headers["Round"] = str(game_num)
    pgn_game.headers["White"] = "NeuralNet-Policy" if nn_is_white else "Stockfish"
    pgn_game.headers["Black"] = "Stockfish" if nn_is_white else "NeuralNet-Policy"
    pgn_node = pgn_game

    move_count = 0
    nn_move_count = 0

    # Profile accumulators
    profile_totals = {
        'total': 0.0,
        'api': 0.0,
        'collect': 0.0,
        'process': 0.0,
        'build_tree': 0.0,
        'minimax': 0.0
    }

    try:
        while not board.is_game_over() and move_count < 300:
            is_nn_turn = (board.turn == chess.WHITE) == nn_is_white
            move_label = format_move_number(move_count)

            if is_nn_turn:
                move_start = time.time()
                move, stats = evaluator.get_best_move(board)
                move_time = time.time() - move_start
                move_san = board.san(move)
                if stats['from_book']:
                    print(f"  Game {game_num}, {move_label} (NN): {move_san} | [BOOK] ({move_time:.1f}s)", flush=True)
                else:
                    cand_info = f"cands={stats['candidates']}"
                    info_parts = [f"pos={stats['positions']:,}", cand_info]
                    print(f"  Game {game_num}, {move_label} (NN): {move_san} | {' '.join(info_parts)} ({move_time:.1f}s)", flush=True)
                    # Accumulate profiling data
                    if 'profile' in stats:
                        for key in profile_totals:
                            profile_totals[key] += stats['profile'].get(key, 0.0)
                        nn_move_count += 1
            else:
                move_start = time.time()
                result = engine.play(board, chess.engine.Limit(time=time_limit))
                move_time = time.time() - move_start
                move = result.move
                move_san = board.san(move)
                print(f"  Game {game_num}, {move_label} (SF): {move_san} ({move_time:.1f}s)", flush=True)

            pgn_node = pgn_node.add_variation(move)
            board.push(move)
            move_count += 1

            # Get Stockfish evaluation of the position after the move (from white's perspective)
            if not board.is_game_over():
                info = engine.analyse(board, chess.engine.Limit(depth=12))
                score = info["score"].white()
                if score.is_mate():
                    sf_eval = f"M{score.mate()}"
                else:
                    sf_eval = f"{score.score() / 100:+.2f}"
                print(f"  Game {game_num}, -> SF eval: {sf_eval}", flush=True)

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

    # Print profile summary for the game
    if nn_move_count > 0:
        print(f"  Game {game_num} [PROFILE] {nn_move_count} NN moves | "
              f"total={profile_totals['total']:.1f}s | api={profile_totals['api']:.1f}s | "
              f"collect={profile_totals['collect']:.1f}s | process={profile_totals['process']:.1f}s | "
              f"build_tree={profile_totals['build_tree']:.1f}s | minimax={profile_totals['minimax']:.1f}s", flush=True)

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
    search_depth: int = 2,
    skill_level: int = 3,
    topk: int = 10,
    prob_threshold: float = 1.0,
    min_moves: int = 1,
    server_url: str = None,
    games_file: str = "match_games_policy.pgn"
):
    """
    Run a match with multiple games in parallel processes.
    Uses policy predictions to guide move selection.

    Args:
        model_path: Path to the trained model (unused, for compatibility)
        stockfish_path: Path to Stockfish executable
        num_games: Total number of games to play
        num_workers: Number of parallel worker processes
        time_limit: Time limit per move for Stockfish
        search_depth: Search depth for NN player (after policy filtering)
        skill_level: Stockfish skill level (0-20)
        topk: Number of moves to request from server and explore
        prob_threshold: Stop adding moves when cumulative probability exceeds this
        min_moves: Minimum number of moves to explore regardless of probability
        server_url: URL of the inference server
        games_file: Path to save all completed games (PGN)
    """
    print("=" * 70)
    print("MULTIPROCESS MATCH WITH POLICY: Neural Network vs Stockfish")
    print("=" * 70)
    print(f"Total games: {num_games}")
    print(f"Worker processes: {num_workers}")
    print(f"NN search depth: {search_depth} ply (on policy-filtered moves)")
    print(f"Policy top-k: {topk}")
    print(f"Probability threshold: {prob_threshold:.0%}, min_moves={min_moves}")
    print(f"Server URL: {server_url}")
    print(f"Stockfish skill level: {skill_level}")
    print(f"Stockfish time: {time_limit}s per move")
    print(f"Games saved to: {games_file}")
    print("=" * 70)

    # Prepare game arguments
    game_args = [
        (game_num, game_num % 2 == 1, stockfish_path, time_limit, search_depth, skill_level, topk, prob_threshold, min_moves, server_url)
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
        f.write(f"; Multiprocess Match with Policy: Neural Network vs Stockfish\n")
        f.write(f"; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"; NN Search Depth: {search_depth} ply (on policy-filtered moves)\n")
        f.write(f"; Policy Top-K: {topk}\n")
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
        description='Run a match between Neural Network (with Policy) and Stockfish',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python play_match_multiprocess_with_policy.py --skill 3 --games 100 --workers 4
    python play_match_multiprocess_with_policy.py --skill 5 --games 50 --depth 2 --topk 10
        """
    )
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of parallel worker processes (default: 4)')
    parser.add_argument('--games', '-g', type=int, default=100,
                        help='Number of games to play (default: 100)')
    parser.add_argument('--skill', '-s', type=int, default=3,
                        help='Stockfish skill level 0-20 (default: 3)')
    parser.add_argument('--depth', '-d', type=int, default=2,
                        help='NN search depth in ply on filtered moves (default: 2)')
    parser.add_argument('--topk', '-k', type=int, default=10,
                        help='Number of moves to request from server and explore (default: 10)')
    parser.add_argument('--prob-threshold', type=float, default=1.0,
                        help='Stop adding moves when cumulative prob exceeds this (default: 1.0 = disabled)')
    parser.add_argument('--min-moves', type=int, default=1,
                        help='Minimum moves to explore regardless of probability (default: 1)')
    parser.add_argument('--server-url', type=str, required=True,
                        help='Inference server URL (required)')
    parser.add_argument('--time', '-t', type=float, default=5.0,
                        help='Stockfish time per move in seconds (default: 5.0)')
    parser.add_argument('--stockfish', type=str, default="/opt/homebrew/bin/stockfish",
                        help='Path to Stockfish executable')
    parser.add_argument('--output', '-o', type=str, default="match_games_policy.pgn",
                        help='Output PGN file (default: match_games_policy.pgn)')

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
        args.topk,
        args.prob_threshold,
        args.min_moves,
        args.server_url,
        args.output
    )
