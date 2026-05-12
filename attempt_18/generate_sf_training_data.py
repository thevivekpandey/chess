"""
Policy-Only Game Generator with Stockfish Labeling

Generates training data by:
1. Playing games using only the policy network (no MCTS, ~10ms/move)
2. Labeling each visited (non-terminal) position with Stockfish multi-PV + eval
3. Writing rows in the project CSV format used by chess_engine.load_data_from_csv:

   fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5

   - eval  : position evaluation in PAWNS, from WHITE's perspective
             (matches the foundation dataset and the ChessNet value head, which
             chess_engine.ChessDataset feeds straight through normalize_eval()).
   - moveN : UCI strings for Stockfish's top-N principal variations.
   - scoreN: evaluation in CENTIPAWNS after that move, from the SIDE-TO-MOVE's
             perspective, so the best move always has the highest score. This is
             what chess_engine.moves_to_policy_sparse() expects, and matches the
             foundation dataset.

Labeling cost is controlled by --sf-time / --sf-nodes / --sf-depth. Depth-only
search blows up on the messy positions a weak policy net reaches, so the default
is a fixed ~100 ms per position (with --sf-depth acting as an early-exit ceiling).
A few hundred ms of Stockfish is already far stronger than this network needs as
a teacher, and it keeps wall-clock predictable.
"""

import argparse
import multiprocessing as mp
import time

import chess
import chess.engine
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional - fall back to a no-op wrapper
    def tqdm(iterable=None, total=None, desc=None, **_kwargs):
        return iterable if iterable is not None else range(total or 0)

from chess_engine import ChessNet, fen_to_tensor, move_to_policy_index, ensure_parent_dir

MATE_CP = 10000           # centipawn magnitude used to represent forced mates
MATE_PAWNS_CLAMP = 100.0  # clamp the white-pov eval column to +/- this (pawns)


# ---------------------------------------------------------------------------
# Step 1: play games with the policy network
# ---------------------------------------------------------------------------

def _legal_move_logits(model, board, device):
    """Return (legal_moves, logits_for_those_moves) using the policy head."""
    with torch.no_grad():
        board_tensor = fen_to_tensor(board.fen()).unsqueeze(0).to(device)
        _value, policy_logits = model(board_tensor)        # (1, 73, 8, 8)
        flat = policy_logits.reshape(-1).float().cpu().numpy()  # (4672,)

    legal_moves = list(board.legal_moves)
    move_logits = np.full(len(legal_moves), -1e9, dtype=np.float64)
    for i, mv in enumerate(legal_moves):
        idx = move_to_policy_index(mv.uci())
        if idx is not None:
            src_row, src_col, plane = idx
            move_logits[i] = flat[plane * 64 + src_row * 8 + src_col]
    return legal_moves, move_logits


def _sample_move(model, board, device, temperature):
    legal_moves, move_logits = _legal_move_logits(model, board, device)
    temperature = max(float(temperature), 1e-3)
    move_logits = move_logits / temperature
    move_logits -= move_logits.max()
    probs = np.exp(move_logits)
    probs /= probs.sum()
    return legal_moves[int(np.random.choice(len(legal_moves), p=probs))]


def play_game_with_policy(model, device, temperature=0.8, max_moves=160):
    """
    Play one game with the policy network; return list of visited (non-terminal) FENs.

    Ends on checkmate/stalemate, on a *claimable* draw (threefold repetition or the
    50-move rule - a weak net shuffles into these constantly), or after `max_moves`
    plies. Without the claimable-draw check, games routinely run to the ply cap with
    ~100 near-identical shuffling positions that just waste Stockfish time.
    """
    board = chess.Board()
    positions = []
    model.eval()
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_moves:
        positions.append(board.fen())
        board.push(_sample_move(model, board, device, temperature))
    return positions


# ---------------------------------------------------------------------------
# Parallel game generation (each worker process: one model, plays games)
# ---------------------------------------------------------------------------

_GEN_MODEL = None
_GEN_DEVICE = None


def _gen_worker_init(model_path: str, device: str):
    global _GEN_MODEL, _GEN_DEVICE
    torch.set_num_threads(1)
    try:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
    except Exception:  # noqa: BLE001
        dev = torch.device("cpu")
    _GEN_DEVICE = dev
    model = ChessNet()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_GEN_DEVICE)
    model.eval()
    _GEN_MODEL = model


def _gen_worker_play(args):
    _game_idx, temperature, max_moves = args
    return play_game_with_policy(_GEN_MODEL, _GEN_DEVICE, temperature, max_moves)


def _play_games_parallel(model_path, device, num_games, temperature, max_moves, num_workers):
    """Play num_games across num_workers processes; return the flat list of visited FENs."""
    n = max(1, min(int(num_workers), num_games))
    # 'spawn' so each worker gets a clean CUDA context (the parent may already
    # have CUDA initialized from a previous iteration's training - forking that
    # would leave the child unable to put the model on the GPU).
    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        ctx = None
    print(f"  (playing {num_games} games across {n} worker process(es) on {device})", flush=True)
    all_positions = []
    report_every = max(1, num_games // 10)
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                             initializer=_gen_worker_init,
                             initargs=(model_path, device)) as executor:
        futures = [executor.submit(_gen_worker_play, (gi, temperature, max_moves))
                   for gi in range(num_games)]
        done = 0
        for future in as_completed(futures):
            all_positions.extend(future.result())
            done += 1
            if done % report_every == 0 or done == num_games:
                print(f"  [play] {done}/{num_games} games done | {len(all_positions)} positions", flush=True)
    return all_positions


# ---------------------------------------------------------------------------
# Step 2: label positions with Stockfish (runs in worker processes)
# ---------------------------------------------------------------------------

def _build_limit(sf_time=0.1, sf_depth=0, sf_nodes=0):
    """Pick a chess.engine.Limit from the (time, depth, nodes) knobs."""
    sf_time = float(sf_time or 0.0)
    sf_depth = int(sf_depth or 0)
    sf_nodes = int(sf_nodes or 0)
    if sf_nodes > 0:
        return chess.engine.Limit(nodes=sf_nodes)
    if sf_time > 0:
        # depth acts as an early-exit ceiling so easy positions don't burn the
        # full time budget; on hard positions the time cap kicks in.
        return chess.engine.Limit(time=sf_time, depth=sf_depth if sf_depth > 0 else None)
    if sf_depth > 0:
        return chess.engine.Limit(depth=sf_depth)
    return chess.engine.Limit(time=0.1)


def label_positions_batch(positions, sf_path, sf_time=0.1, sf_depth=0, sf_nodes=0,
                          multipv=5, sf_hash=128):
    """
    Label a batch of FENs with Stockfish using a single engine process.

    Returns a list of (fen, eval_pawns_white, [(uci, score_cp_stm), ...]).
    """
    rows = []
    limit = _build_limit(sf_time, sf_depth, sf_nodes)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[label] could not start Stockfish: {exc}", flush=True)
        return rows

    try:
        try:
            engine.configure({"Threads": 1, "Hash": int(sf_hash)})
        except chess.engine.EngineError:
            pass

        for fen in positions:
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            n_legal = board.legal_moves.count()
            if n_legal == 0:
                continue
            try:
                info = engine.analyse(board, limit, multipv=min(multipv, n_legal))
            except chess.engine.EngineTerminatedError:
                break  # engine died; salvage what we have
            except chess.engine.EngineError:
                continue

            if isinstance(info, dict):  # defensive: some versions return a dict for multipv=1
                info = [info]

            moves = []
            for pv in info:
                pv_line = pv.get("pv")
                if not pv_line:
                    continue
                score_cp = pv["score"].relative.score(mate_score=MATE_CP)
                if score_cp is None:
                    continue
                moves.append((pv_line[0].uci(), int(score_cp)))
            if not moves:
                continue

            white_cp = info[0]["score"].white().score(mate_score=MATE_CP)
            if white_cp is None:
                continue
            eval_pawns = white_cp / 100.0
            eval_pawns = max(-MATE_PAWNS_CLAMP, min(MATE_PAWNS_CLAMP, eval_pawns))

            rows.append((fen, eval_pawns, moves))
    finally:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _describe_limit(sf_time, sf_depth, sf_nodes):
    if sf_nodes and sf_nodes > 0:
        return f"{sf_nodes} nodes/position"
    if sf_time and sf_time > 0:
        return f"{sf_time:g}s/position" + (f" (depth<={sf_depth} ceiling)" if sf_depth and sf_depth > 0 else "")
    if sf_depth and sf_depth > 0:
        return f"depth {sf_depth}/position"
    return "0.1s/position"


def generate_training_data(
    model_path: str,
    output_path: str,
    stockfish_path: str,
    num_games: int = 100,
    num_workers: int = 8,
    temperature: float = 0.8,
    sf_depth: int = 20,
    sf_multipv: int = 5,
    sf_time: float = 0.1,
    sf_nodes: int = 0,
    sf_hash: int = 128,
    max_moves: int = 160,
    device: str = "cuda",
):
    limit_desc = _describe_limit(sf_time, sf_depth, sf_nodes)
    print("Generating training data with Stockfish labeling...")
    print(f"  Games: {num_games}  Temperature: {temperature}  Max plies/game: {max_moves}")
    print(f"  SF budget: {limit_desc}  multi-PV: {sf_multipv}  Workers: {num_workers}  Hash: {sf_hash}MB/worker")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_path}")

    play_seconds = 0.0
    label_seconds = 0.0

    # Step 1: play games
    print("\nStep 1: playing games with the policy network...", flush=True)
    t0 = time.time()
    if num_workers and num_workers > 1 and num_games > 1:
        # Parallel: worker processes load their own model; the parent stays
        # CUDA-free here, which keeps the 'spawn' pool clean.
        all_positions = _play_games_parallel(
            model_path, device, num_games, temperature, max_moves, num_workers
        )
    else:
        device_torch = torch.device(device)
        model = ChessNet()
        checkpoint = torch.load(model_path, map_location=device_torch, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device_torch)
        model.eval()
        all_positions = []
        for _ in tqdm(range(num_games), desc="Playing games"):
            all_positions.extend(play_game_with_policy(model, device_torch, temperature, max_moves))

    # De-duplicate while preserving order (the start position appears in every game).
    seen = set()
    unique_positions = []
    for fen in all_positions:
        if fen not in seen:
            seen.add(fen)
            unique_positions.append(fen)
    play_seconds = time.time() - t0
    print(f"  Collected {len(all_positions)} positions ({len(unique_positions)} unique) from {num_games} games "
          f"in {play_seconds:.1f}s", flush=True)
    if num_games:
        print(f"  Average plies per game: {len(all_positions) / num_games:.1f}", flush=True)

    # Step 2: label with Stockfish in parallel
    print("\nStep 2: labeling positions with Stockfish...", flush=True)
    n_chunks = max(1, num_workers * 4)
    chunk_size = max(1, (len(unique_positions) + n_chunks - 1) // n_chunks)
    batches = [unique_positions[i:i + chunk_size] for i in range(0, len(unique_positions), chunk_size)]

    rows = []
    if batches:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(label_positions_batch, batch, stockfish_path,
                                sf_time, sf_depth, sf_nodes, sf_multipv, sf_hash)
                for batch in batches
            ]
            done = 0
            for future in as_completed(futures):
                rows.extend(future.result())
                done += 1
                elapsed = time.time() - t0
                rate = len(rows) / elapsed if elapsed > 0 else 0.0
                eta = (len(unique_positions) - len(rows)) / rate if rate > 0 else 0.0
                print(f"  [label] {done}/{len(futures)} batches | {len(rows)}/{len(unique_positions)} positions "
                      f"| {elapsed:.0f}s elapsed | ~{rate:.0f} pos/s | ETA ~{eta:.0f}s", flush=True)
        label_seconds = time.time() - t0
    print(f"Labeled {len(rows)} positions in {label_seconds:.1f}s")

    # Step 3: write CSV
    print(f"\nStep 3: writing {output_path}...")
    ensure_parent_dir(output_path)
    with open(output_path, "w") as f:
        f.write("fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5\n")
        for fen, eval_pawns, moves in rows:
            cells = [fen, f"{eval_pawns:.2f}"]
            for i in range(5):
                if i < len(moves):
                    cells.extend([moves[i][0], str(moves[i][1])])
                else:
                    cells.extend(["", ""])
            f.write(",".join(cells) + "\n")
    print(f"Saved {len(rows)} training examples to {output_path}")

    if rows:
        evals = np.array([r[1] for r in rows], dtype=np.float32)
        print("\nEval column (white pov, pawns):")
        print(f"  mean {evals.mean():.2f}  std {evals.std():.2f}  min {evals.min():.2f}  max {evals.max():.2f}")

    return {
        "positions": len(rows),
        "games": num_games,
        "play_seconds": play_seconds,
        "label_seconds": label_seconds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data with Stockfish labeling")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path for training data (CSV)")
    parser.add_argument("--stockfish", type=str, required=True, help="Path to Stockfish binary")
    parser.add_argument("--games", type=int, default=100, help="Number of games to generate")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel labeling workers (Stockfish is 1 thread each)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Policy sampling temperature")
    parser.add_argument("--sf-time", type=float, default=0.1, help="Seconds of Stockfish per position (0 to disable; default 0.1)")
    parser.add_argument("--sf-nodes", type=int, default=0, help="Stockfish nodes per position (overrides --sf-time/--sf-depth if >0)")
    parser.add_argument("--sf-depth", type=int, default=20, help="Stockfish depth: a search ceiling when --sf-time>0, or the budget when --sf-time=0")
    parser.add_argument("--sf-multipv", type=int, default=5, help="Stockfish multi-PV count")
    parser.add_argument("--sf-hash", type=int, default=128, help="Hash (MB) per Stockfish worker")
    parser.add_argument("--max-moves", type=int, default=160, help="Max plies per game before cutting it short")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda/mps)")
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
        sf_time=args.sf_time,
        sf_nodes=args.sf_nodes,
        sf_hash=args.sf_hash,
        max_moves=args.max_moves,
        device=args.device,
    )
