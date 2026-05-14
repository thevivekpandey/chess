"""
Benchmark self-play game generation speed (no Stockfish labeling, no training).

Wraps `generate_sf_training_data._play_games_parallel` so we can measure exactly
how long the play step takes for a given (model, search_depth, workers, games)
combo. Useful for sizing search-depth vs throughput trade-offs.

Example:
    ~/myvenv/bin/python benchmark_play.py \\
        --model ../attempt_17/m0.pt --games 100 --workers 20 \\
        --search-depth 2 --device cuda
"""

import argparse
import time

from generate_sf_training_data import _play_games_parallel


def main():
    parser = argparse.ArgumentParser(description="Benchmark self-play speed")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--workers", type=int, default=8, help="Parallel game workers")
    parser.add_argument("--temperature", type=float, default=0.8, help="Move sampling temperature")
    parser.add_argument("--max-moves", type=int, default=160, help="Max plies per game")
    parser.add_argument("--search-depth", type=int, default=1,
                        help="Plies of value-head search per move (0 = policy head only)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda/mps)")
    parser.add_argument("--save-games", type=str, default=None,
                        help="Optional PGN output path for the played games")
    args = parser.parse_args()

    move_desc = (f"{args.search_depth}-ply value search"
                 if args.search_depth and args.search_depth >= 1
                 else "policy head only")
    print(f"Model:        {args.model}")
    print(f"Games:        {args.games}")
    print(f"Workers:      {args.workers}")
    print(f"Device:       {args.device}")
    print(f"Move select:  {move_desc}")
    print(f"Temperature:  {args.temperature}")
    print(f"Max moves:    {args.max_moves}")
    print()

    t0 = time.time()
    positions, games = _play_games_parallel(
        model_path=args.model,
        device=args.device,
        num_games=args.games,
        temperature=args.temperature,
        max_moves=args.max_moves,
        num_workers=args.workers,
        search_depth=args.search_depth,
    )
    elapsed = time.time() - t0

    plies = [len(g.get("moves", [])) for g in games]
    avg_plies = sum(plies) / len(plies) if plies else 0.0
    total_plies = sum(plies)
    games_per_min = (args.games / elapsed) * 60.0 if elapsed > 0 else 0.0
    plies_per_sec = total_plies / elapsed if elapsed > 0 else 0.0
    sec_per_game = elapsed / args.games if args.games > 0 else 0.0

    print()
    print("=" * 60)
    print(f"Total time:        {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"Games played:      {args.games}")
    print(f"Positions visited: {len(positions)}")
    print(f"Total plies:       {total_plies}  (avg {avg_plies:.1f} plies/game)")
    print(f"Throughput:        {games_per_min:.2f} games/min  ({sec_per_game:.2f} s/game)")
    print(f"                   {plies_per_sec:.2f} plies/sec")
    print("=" * 60)

    if args.save_games:
        from generate_sf_training_data import _write_games_pgn
        import os
        label = os.path.basename(args.model).replace(".pt", "").replace(".pth", "")
        _write_games_pgn(games, args.save_games, model_label=label,
                         temperature=args.temperature)
        print(f"Saved {len(games)} games to {args.save_games}")


if __name__ == "__main__":
    main()
