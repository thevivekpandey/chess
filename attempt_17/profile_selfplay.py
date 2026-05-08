"""Profile a few self-play games to find CPU bottlenecks.

Mirrors the production self-play config (single worker — cProfile doesn't
follow multiprocessing). Saves the raw .prof for later inspection and prints
top-N by both cumulative and own-time.
"""

import cProfile
import io
import pstats
import time

import torch

from play_games_mcts import MCTSEngine
from selfplay_generator import play_selfplay_game


# Match the production command-line config so the profile reflects reality.
N_SIMULATIONS = 400
MCTS_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512
EARLY_EXIT_MIN_SIMS = 200
NUM_GAMES = 3
MAX_PLIES = 200
TOP_N = 30
PROF_OUT = "selfplay.prof"


def profile_selfplay():
    model_path = "attempt_14b_epoch032.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, profiling on CPU")

    print(f"Initializing engine on {device}...")
    engine = MCTSEngine(model_path, device=device, eval_batch_size=EVAL_BATCH_SIZE)

    # Warm up: torch.compile, cuDNN benchmark, channels-last all benefit from a
    # untimed warm-up pass so the profile reflects steady-state behaviour.
    print("Warm-up game (not profiled)...")
    play_selfplay_game(
        neural=engine,
        n_simulations=N_SIMULATIONS,
        mcts_batch_size=MCTS_BATCH_SIZE,
        cpuct=2.0,
        fpu_reduction=0.0,
        reuse_tree=True,
        max_plies=40,
        temperature_moves=30,
        verbose=False,
        early_exit_min_sims=EARLY_EXIT_MIN_SIMS,
    )

    pr = cProfile.Profile()
    print(f"Profiling {NUM_GAMES} games at {N_SIMULATIONS} sims, "
          f"mcts_batch={MCTS_BATCH_SIZE}, eval_batch={EVAL_BATCH_SIZE}...")

    total_plies = 0
    pr.enable()
    t0 = time.time()
    for i in range(NUM_GAMES):
        _, examples = play_selfplay_game(
            neural=engine,
            n_simulations=N_SIMULATIONS,
            mcts_batch_size=MCTS_BATCH_SIZE,
            cpuct=2.0,
            fpu_reduction=0.0,
            reuse_tree=True,
            max_plies=MAX_PLIES,
            temperature_moves=30,
            verbose=False,
            early_exit_min_sims=EARLY_EXIT_MIN_SIMS,
        )
        total_plies += len(examples)
        print(f"  game {i + 1}/{NUM_GAMES}: {len(examples)} plies")
    elapsed = time.time() - t0
    pr.disable()

    moves_per_sec = total_plies / elapsed if elapsed > 0 else 0
    print(f"\nProfiled {NUM_GAMES} games in {elapsed:.1f}s "
          f"({total_plies} plies, {moves_per_sec:.1f} moves/s)")

    pr.dump_stats(PROF_OUT)
    print(f"Raw profile saved to {PROF_OUT}\n")

    for sort_key, label in [
        (pstats.SortKey.CUMULATIVE, "Cumulative Time"),
        (pstats.SortKey.TIME, "Own Time (tottime)"),
    ]:
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(sort_key).print_stats(TOP_N)
        print(f"--- Profile Results ({label}) ---")
        print(s.getvalue())


if __name__ == "__main__":
    profile_selfplay()
