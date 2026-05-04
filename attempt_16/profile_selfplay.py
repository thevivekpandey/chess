
import cProfile
import pstats
import io
import torch
import time
from selfplay_generator import play_selfplay_game
from play_games_mcts import MCTSEngine

def profile_selfplay():
    model_path = "attempt_14b_epoch032.pt"  # Assumes this exists based on user prompt
    if not torch.cuda.is_available():
        print("CUDA not available, profiling on CPU")
        device = "cpu"
    else:
        device = "cuda"

    print(f"Initializing engine on {device}...")
    engine = MCTSEngine(model_path, device=device, eval_batch_size=256)
    
    # Run one game with profiling
    pr = cProfile.Profile()
    print("Starting profiled game...")
    pr.enable()
    
    start_time = time.time()
    game, examples = play_selfplay_game(
        neural=engine,
        n_simulations=800,
        mcts_batch_size=64,
        cpuct=2.0,
        fpu_reduction=0.0,
        reuse_tree=True,
        max_plies=100,  # Limit for profiling speed
        temperature_moves=30,
        verbose=False
    )
    end_time = time.time()
    
    pr.disable()
    print(f"Game finished in {end_time - start_time:.2f} seconds.")
    print(f"Total plies: {len(examples)}")
    
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)  # Top 20 functions
    
    print("\n--- Profile Results (Cumulative Time) ---")
    print(s.getvalue())
    
    # Also look at internal stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
    ps.print_stats(20)
    print("\n--- Profile Results (Own Time) ---")
    print(s.getvalue())

if __name__ == "__main__":
    profile_selfplay()
