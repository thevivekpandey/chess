"""
Policy-Only Game Generator with Stockfish Labeling

Generates training data by:
1. Playing games using only the policy network (no MCTS, ~10ms/move)
2. Labeling each visited (non-terminal) position with Stockfish multi-PV + eval
3. Writing rows in the project CSV format used by chess_engine.load_data_from_csv:

   fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5

   - eval  : position evaluation in PAWNS, from WHITE's perspective.
   - moveN : UCI strings for Stockfish's top-N principal variations.
   - scoreN: WHITE-POV evaluation DELTA in PAWNS after the move, i.e.
             score_i = move_eval_white - eval. Positive deltas mean the move
             makes White's position better; negative deltas mean it makes
             White's position worse. This matches the convention produced by
             attempt_02/augment_with_top_moves.py (the foundation dataset).
             The downstream loader (chess_engine.moves_to_policy_sparse) flips
             the sign on Black's turn so that the best move for the side to
             move always softmaxes to the highest probability.

Labeling cost is controlled by --sf-time / --sf-nodes / --sf-depth. Depth-only
search blows up on the messy positions a weak policy net reaches, so the default
is a fixed ~100 ms per position (with --sf-depth acting as an early-exit ceiling).
A few hundred ms of Stockfish is already far stronger than this network needs as
a teacher, and it keeps wall-clock predictable.
"""

import argparse
import gc
import multiprocessing as mp
import os
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed


def _terminate_pool(executor):
    """Force-kill any worker processes that didn't exit cleanly after shutdown.

    PyTorch's CUDA-context finalize in a spawn worker can stall, so a worker
    that has received the executor's sentinel may keep its Python process (and
    its GPU memory mapping) alive long after `shutdown(wait=True)` returns.
    Capture the worker handles BEFORE shutdown (it nulls `_processes`), then
    explicitly terminate any survivor."""
    procs_dict = getattr(executor, "_processes", None) or {}
    procs = list(procs_dict.values())
    executor.shutdown(wait=True)
    for p in procs:
        if not p.is_alive():
            continue
        try:
            p.terminate()
            p.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        if p.is_alive():
            try:
                p.kill()
                p.join(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional - fall back to a no-op wrapper
    def tqdm(iterable=None, total=None, desc=None, **_kwargs):
        return iterable if iterable is not None else range(total or 0)

from chess_engine import ChessNet, ensure_parent_dir
from move_search import choose_move

MATE_CP = 10000           # centipawn magnitude used to represent forced mates
MATE_PAWNS_CLAMP = 100.0  # clamp the white-pov eval column to +/- this (pawns)


# ---------------------------------------------------------------------------
# Step 1: play games with the network
# ---------------------------------------------------------------------------

def play_game_with_policy(model, device, temperature=0.05, max_moves=160, search_depth=1,
                          early_temperature_plies=5):
    """
    Play one self-play game with the network.

    Move selection goes through move_search.choose_move: `search_depth` plies of
    value-head negamax (search_depth=1 = pick the move whose resulting position the
    value head likes best), or the raw policy head when search_depth<=0.

    Returns ``(positions, game_record)`` where ``positions`` is the list of visited
    (non-terminal) FENs and ``game_record`` is ``{"moves": [uci, ...], "result": str,
    "plies": int}`` describing the actual self-play game (so it can be written to PGN).

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
        t = temperature if len(board.move_stack) < early_temperature_plies else 0.0
        board.push(choose_move(model, board, device, search_depth, t))
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    moves_uci = [mv.uci() for mv in board.move_stack]
    return positions, {"moves": moves_uci, "result": result, "plies": len(moves_uci)}


def _write_games_pgn(games, path, model_label="", temperature=None):
    """Write self-play `games` (the dicts returned by play_game_with_policy) to a PGN file."""
    if not games:
        return 0
    ensure_parent_dir(path)
    written = 0
    with open(path, "w") as f:
        for i, g in enumerate(games, 1):
            game = chess.pgn.Game()
            game.headers["Event"] = "policy-net self-play"
            game.headers["Site"] = "attempt_18"
            game.headers["Round"] = str(i)
            game.headers["White"] = model_label or "ChessNet"
            game.headers["Black"] = model_label or "ChessNet"
            game.headers["Result"] = g.get("result", "*")
            if temperature is not None:
                game.headers["Temperature"] = f"{temperature:g}"
            node = game
            ok = True
            for uci in g.get("moves", []):
                try:
                    node = node.add_variation(chess.Move.from_uci(uci))
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            game.headers["PlyCount"] = str(g.get("plies", len(g.get("moves", []))))
            print(game, file=f, end="\n\n")
            written += 1
    return written


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
    _game_idx, temperature, max_moves, search_depth, early_temperature_plies = args
    return play_game_with_policy(_GEN_MODEL, _GEN_DEVICE, temperature, max_moves, search_depth,
                                 early_temperature_plies=early_temperature_plies)


def _play_games_parallel(model_path, device, num_games, temperature, max_moves, num_workers, search_depth,
                         early_temperature_plies):
    """Play num_games across num_workers processes; return (flat list of visited FENs, list of game records)."""
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
    games = []
    report_every = max(1, num_games // 10)
    executor = ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                                   initializer=_gen_worker_init,
                                   initargs=(model_path, device))
    try:
        futures = [executor.submit(_gen_worker_play,
                                   (gi, temperature, max_moves, search_depth, early_temperature_plies))
                   for gi in range(num_games)]
        done = 0
        for future in as_completed(futures):
            positions, game_rec = future.result()
            all_positions.extend(positions)
            games.append(game_rec)
            done += 1
            if done % report_every == 0 or done == num_games:
                print(f"  [play] {done}/{num_games} games done | {len(all_positions)} positions", flush=True)
    finally:
        _terminate_pool(executor)
    return all_positions, games


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

            white_cp = info[0]["score"].white().score(mate_score=MATE_CP)
            if white_cp is None:
                continue
            eval_pawns = white_cp / 100.0
            eval_pawns = max(-MATE_PAWNS_CLAMP, min(MATE_PAWNS_CLAMP, eval_pawns))

            # Foundation convention: per-move score is the white-POV pawn DELTA
            # from the position's base eval. The loader flips sign on Black's
            # turn, so storing white-POV here keeps the data symmetric.
            moves = []
            for pv in info:
                pv_line = pv.get("pv")
                if not pv_line:
                    continue
                move_white_cp = pv["score"].white().score(mate_score=MATE_CP)
                if move_white_cp is None:
                    continue
                move_white_pawns = move_white_cp / 100.0
                move_white_pawns = max(-MATE_PAWNS_CLAMP, min(MATE_PAWNS_CLAMP, move_white_pawns))
                delta = round(move_white_pawns - eval_pawns, 2)
                moves.append((pv_line[0].uci(), delta))
            if not moves:
                continue

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
    temperature: float = 0.05,
    early_temperature_plies: int = 5,
    sf_depth: int = 20,
    sf_multipv: int = 5,
    sf_time: float = 0.1,
    sf_nodes: int = 0,
    sf_hash: int = 128,
    max_moves: int = 160,
    device: str = "cuda",
    save_games_path: str = None,
    search_depth: int = 1,
):
    limit_desc = _describe_limit(sf_time, sf_depth, sf_nodes)
    move_desc = (f"{search_depth}-ply value search" if search_depth and search_depth >= 1
                 else "policy head only")
    print("Generating training data with Stockfish labeling...")
    print(f"  Games: {num_games}  Move selection: {move_desc}  "
          f"Temperature: {temperature} (first {early_temperature_plies} plies, then argmax)  "
          f"Max plies/game: {max_moves}")
    print(f"  SF budget: {limit_desc}  multi-PV: {sf_multipv}  Workers: {num_workers}  Hash: {sf_hash}MB/worker")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_path}")
    if save_games_path:
        print(f"  Self-play PGN: {save_games_path}")

    play_seconds = 0.0
    label_seconds = 0.0
    games = []

    # Step 1: play games
    print(f"\nStep 1: playing games ({move_desc})...", flush=True)
    t0 = time.time()
    if num_workers and num_workers > 1 and num_games > 1:
        # Parallel: worker processes load their own model; the parent stays
        # CUDA-free here, which keeps the 'spawn' pool clean.
        all_positions, games = _play_games_parallel(
            model_path, device, num_games, temperature, max_moves, num_workers, search_depth,
            early_temperature_plies=early_temperature_plies,
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
            positions, game_rec = play_game_with_policy(model, device_torch, temperature, max_moves, search_depth,
                                                        early_temperature_plies=early_temperature_plies)
            all_positions.extend(positions)
            games.append(game_rec)

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

    # Save the self-play games as PGN (the actual move sequences + results).
    games_written = 0
    if save_games_path and games:
        model_label = os.path.splitext(os.path.basename(model_path))[0] if model_path else "ChessNet"
        games_written = _write_games_pgn(games, save_games_path, model_label=model_label, temperature=temperature)
        print(f"  Saved {games_written} self-play games to {save_games_path}", flush=True)

    # Step 2: label with Stockfish in parallel
    print("\nStep 2: labeling positions with Stockfish...", flush=True)
    n_chunks = max(1, num_workers * 4)
    chunk_size = max(1, (len(unique_positions) + n_chunks - 1) // n_chunks)
    batches = [unique_positions[i:i + chunk_size] for i in range(0, len(unique_positions), chunk_size)]

    rows = []
    if batches:
        t0 = time.time()
        # 'spawn' so label workers don't inherit the parent's mmap'd CUDA libs
        # (forked labelers showed up in nvidia-smi holding ~900 MB each once the
        # parent had touched CUDA in an earlier iteration's training phase).
        try:
            label_ctx = mp.get_context("spawn")
        except ValueError:
            label_ctx = None
        executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=label_ctx)
        try:
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
        finally:
            _terminate_pool(executor)
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
        "games_written": games_written,
        "games_pgn": save_games_path if games_written else None,
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
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Policy sampling temperature (used only for the first --early-temperature-plies plies; argmax thereafter)")
    parser.add_argument("--early-temperature-plies", type=int, default=5,
                        help="Number of opening plies that use --temperature; subsequent plies use argmax (0 = argmax everywhere)")
    parser.add_argument("--sf-time", type=float, default=0.1, help="Seconds of Stockfish per position (0 to disable; default 0.1)")
    parser.add_argument("--sf-nodes", type=int, default=0, help="Stockfish nodes per position (overrides --sf-time/--sf-depth if >0)")
    parser.add_argument("--sf-depth", type=int, default=20, help="Stockfish depth: a search ceiling when --sf-time>0, or the budget when --sf-time=0")
    parser.add_argument("--sf-multipv", type=int, default=5, help="Stockfish multi-PV count")
    parser.add_argument("--sf-hash", type=int, default=128, help="Hash (MB) per Stockfish worker")
    parser.add_argument("--max-moves", type=int, default=160, help="Max plies per game before cutting it short")
    parser.add_argument("--search-depth", type=int, default=1,
                        help="Plies of value-head negamax for move selection during self-play "
                             "(1 = pick the move whose resulting position the value head likes best; 0 = policy head only)")
    parser.add_argument("--save-games", type=str, default=None, help="If set, write the self-play games to this PGN file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    generate_training_data(
        model_path=args.model,
        output_path=args.output,
        stockfish_path=args.stockfish,
        num_games=args.games,
        num_workers=args.workers,
        temperature=args.temperature,
        early_temperature_plies=args.early_temperature_plies,
        sf_depth=args.sf_depth,
        sf_multipv=args.sf_multipv,
        sf_time=args.sf_time,
        sf_nodes=args.sf_nodes,
        sf_hash=args.sf_hash,
        max_moves=args.max_moves,
        device=args.device,
        save_games_path=args.save_games,
        search_depth=args.search_depth,
    )
