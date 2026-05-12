"""
Evaluate a trained model against Stockfish.

Plays games with the policy network (legal-move-masked argmax / sampled from the
policy head) against Stockfish at a chosen skill level and reports W/D/L, score
and a rough ELO-difference estimate.

With num_workers > 1 the games are split across worker processes (each loads the
model on CPU and runs its own Stockfish) - eval is otherwise pure serial Stockfish
time. Batch-1 CPU inference is cheap relative to the 0.1s/move Stockfish budget, so
keeping the model off the GPU here avoids CUDA-in-subprocess fragility for free.
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

import chess
import chess.engine
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional - fall back to a no-op wrapper
    def tqdm(iterable=None, total=None, desc=None, **_kwargs):
        return iterable if iterable is not None else range(total or 0)

from chess_engine import ChessNet, fen_to_tensor, move_to_policy_index


@dataclass
class GameResult:
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: int
    white_player: str
    black_player: str


def get_model_move(model: ChessNet, board: chess.Board, device: torch.device,
                   temperature: float = 0.1) -> chess.Move:
    """Pick a legal move from the policy head (legal-move masked)."""
    with torch.no_grad():
        board_tensor = fen_to_tensor(board.fen()).unsqueeze(0).to(device)
        _value, policy_logits = model(board_tensor)          # (1, 73, 8, 8)
        flat = policy_logits.reshape(-1).float().cpu().numpy()  # (4672,)

    legal_moves = list(board.legal_moves)
    move_logits = np.full(len(legal_moves), -1e9, dtype=np.float64)
    for i, mv in enumerate(legal_moves):
        idx = move_to_policy_index(mv.uci())
        if idx is not None:
            src_row, src_col, plane = idx
            move_logits[i] = flat[plane * 64 + src_row * 8 + src_col]

    if temperature is None or temperature < 0.2:
        return legal_moves[int(np.argmax(move_logits))]

    move_logits = move_logits / float(temperature)
    move_logits -= move_logits.max()
    probs = np.exp(move_logits)
    probs /= probs.sum()
    return legal_moves[int(np.random.choice(len(legal_moves), p=probs))]


def play_game_vs_stockfish(
    model: ChessNet,
    stockfish_path: str,
    device: torch.device,
    model_color: chess.Color,
    stockfish_level: int = 10,
    stockfish_time: float = 0.1,
    model_temperature: float = 0.1,
    max_moves: int = 200,
) -> GameResult:
    board = chess.Board()
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as sf_engine:
        try:
            sf_engine.configure({"Skill Level": int(stockfish_level)})
        except chess.engine.EngineError:
            pass  # some builds clamp / rename this; ignore and use defaults

        while not board.is_game_over() and len(board.move_stack) < max_moves:
            if board.turn == model_color:
                move = get_model_move(model, board, device, model_temperature)
            else:
                move = sf_engine.play(board, chess.engine.Limit(time=stockfish_time)).move
            board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result_str = "1/2-1/2"  # hit max_moves
    elif outcome.winner is chess.WHITE:
        result_str = "1-0"
    elif outcome.winner is chess.BLACK:
        result_str = "0-1"
    else:
        result_str = "1/2-1/2"

    white_player = "Model" if model_color == chess.WHITE else f"SF_L{stockfish_level}"
    black_player = f"SF_L{stockfish_level}" if model_color == chess.WHITE else "Model"
    return GameResult(result=result_str, moves=len(board.move_stack),
                      white_player=white_player, black_player=black_player)


def _model_outcome_value(result: str, model_color: chess.Color) -> float:
    """1.0 win / 0.5 draw / 0.0 loss, from the model's perspective."""
    if result == "1/2-1/2":
        return 0.5
    model_won = (result == "1-0" and model_color == chess.WHITE) or \
                (result == "0-1" and model_color == chess.BLACK)
    return 1.0 if model_won else 0.0


# ---------------------------------------------------------------------------
# Parallel eval workers (each worker process: one model + one Stockfish per game)
# ---------------------------------------------------------------------------

_WORKER_MODEL = None
_WORKER_DEVICE = None


def _eval_worker_init(model_path: str, device: str):
    global _WORKER_MODEL, _WORKER_DEVICE
    torch.set_num_threads(1)  # one torch thread per worker - avoid oversubscription
    try:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
    except Exception:  # noqa: BLE001
        dev = torch.device("cpu")
    _WORKER_DEVICE = dev
    model = ChessNet()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_WORKER_DEVICE)
    model.eval()
    _WORKER_MODEL = model


def _eval_worker_play(args):
    game_idx, stockfish_path, stockfish_level, stockfish_time, model_temperature = args
    model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
    result = play_game_vs_stockfish(
        _WORKER_MODEL, stockfish_path, _WORKER_DEVICE, model_color,
        stockfish_level, stockfish_time, model_temperature,
    )
    return _model_outcome_value(result.result, model_color)


def _play_games_parallel(model_path, stockfish_path, num_games, stockfish_level,
                         stockfish_time, model_temperature, num_workers, device):
    """Play num_games across num_workers processes; return (wins, draws, losses)."""
    n = max(1, min(int(num_workers), num_games))
    # 'spawn' so each worker gets a clean CUDA context (the parent may already
    # have initialized CUDA - forking that would break torch in the child).
    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        ctx = None
    print(f"  (running {num_games} games across {n} worker process(es) on {device})", flush=True)
    outcomes = []
    report_every = max(1, num_games // 10)
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                             initializer=_eval_worker_init,
                             initargs=(model_path, device)) as executor:
        futures = [
            executor.submit(_eval_worker_play,
                            (gi, stockfish_path, stockfish_level, stockfish_time, model_temperature))
            for gi in range(num_games)
        ]
        for fut in as_completed(futures):
            outcomes.append(fut.result())
            done = len(outcomes)
            if done % report_every == 0 or done == num_games:
                print(f"  [eval] {done}/{num_games} games done", flush=True)
    wins = sum(1 for v in outcomes if v == 1.0)
    draws = sum(1 for v in outcomes if v == 0.5)
    losses = sum(1 for v in outcomes if v == 0.0)
    return wins, draws, losses


def _play_games_sequential(model_path, stockfish_path, num_games, stockfish_level,
                           stockfish_time, model_temperature, device):
    """Play num_games sequentially in-process (model on `device`); return (wins, draws, losses)."""
    device_torch = torch.device(device)
    model = ChessNet()
    checkpoint = torch.load(model_path, map_location=device_torch, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_torch)
    model.eval()

    wins = draws = losses = 0
    for game_idx in tqdm(range(num_games), desc="Playing games"):
        model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        result = play_game_vs_stockfish(
            model, stockfish_path, device_torch, model_color,
            stockfish_level, stockfish_time, model_temperature,
        )
        v = _model_outcome_value(result.result, model_color)
        if v == 1.0:
            wins += 1
        elif v == 0.5:
            draws += 1
        else:
            losses += 1
    return wins, draws, losses


def evaluate_model(
    model_path: str,
    stockfish_path: str,
    num_games: int = 100,
    stockfish_level: int = 10,
    stockfish_time: float = 0.1,
    model_temperature: float = 0.1,
    device: str = "cuda",
    num_workers: int = 1,
) -> Dict:
    print("=" * 80)
    print("EVALUATING MODEL VS STOCKFISH")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Stockfish level: {stockfish_level}  time: {stockfish_time}s/move  games: {num_games}  "
          f"model temperature: {model_temperature}  workers: {num_workers}")
    print()

    if num_workers and num_workers > 1 and num_games > 1:
        wins, draws, losses = _play_games_parallel(
            model_path, stockfish_path, num_games, stockfish_level,
            stockfish_time, model_temperature, num_workers, device,
        )
    else:
        wins, draws, losses = _play_games_sequential(
            model_path, stockfish_path, num_games, stockfish_level,
            stockfish_time, model_temperature, device,
        )

    total_games = wins + draws + losses
    win_rate = wins / total_games if total_games else 0.0
    draw_rate = draws / total_games if total_games else 0.0
    loss_rate = losses / total_games if total_games else 0.0
    score = (wins + 0.5 * draws) / total_games if total_games else 0.0

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total games: {total_games}")
    print(f"Wins:   {wins} ({win_rate:.1%})")
    print(f"Draws:  {draws} ({draw_rate:.1%})")
    print(f"Losses: {losses} ({loss_rate:.1%})")
    print(f"Score:  {score:.3f} ({score:.1%})")
    if 0.01 < score < 0.99:
        elo_diff = -400 * np.log10(1.0 / score - 1.0)
        print(f"Estimated ELO difference vs SF L{stockfish_level}: {elo_diff:+.0f}")
    else:
        print("Score too extreme for a reliable ELO estimate")

    return {
        "total_games": total_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "score": score,
        "stockfish_level": stockfish_level,
        "stockfish_time": stockfish_time,
    }


def evaluate_model_progressive(model_path: str, stockfish_path: str,
                               games_per_level: int = 50, device: str = "cuda",
                               num_workers: int = 1):
    print("=" * 80)
    print("PROGRESSIVE EVALUATION VS STOCKFISH")
    print("=" * 80)
    print(f"Model: {model_path}  games/level: {games_per_level}  workers: {num_workers}\n")

    levels = [1, 5, 10, 15, 20]
    all_results = []
    for level in levels:
        print(f"\n{'=' * 80}\nTESTING VS STOCKFISH LEVEL {level}\n{'=' * 80}")
        stats = evaluate_model(
            model_path=model_path, stockfish_path=stockfish_path,
            num_games=games_per_level, stockfish_level=level,
            stockfish_time=0.1, device=device, num_workers=num_workers,
        )
        all_results.append({"level": level, **stats})

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Level':<8} {'Score':<10} {'W':<6} {'D':<6} {'L':<6}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['level']:<8} {r['score']:<10.3f} {r['wins']:<6} {r['draws']:<6} {r['losses']:<6}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model against Stockfish")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stockfish", type=str, required=True, help="Path to Stockfish binary")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--level", type=int, default=10, help="Stockfish skill level (0-20)")
    parser.add_argument("--sf-time", type=float, default=0.1, help="Stockfish time per move (s)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model sampling temperature")
    parser.add_argument("--workers", type=int, default=8, help="Parallel game-playing workers (model runs on CPU when >1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for the sequential (workers<=1) path")
    parser.add_argument("--progressive", action="store_true", help="Test against multiple levels")
    args = parser.parse_args()

    if args.progressive:
        evaluate_model_progressive(
            model_path=args.model, stockfish_path=args.stockfish,
            games_per_level=args.games, device=args.device, num_workers=args.workers,
        )
    else:
        evaluate_model(
            model_path=args.model, stockfish_path=args.stockfish,
            num_games=args.games, stockfish_level=args.level,
            stockfish_time=args.sf_time, model_temperature=args.temperature,
            device=args.device, num_workers=args.workers,
        )
