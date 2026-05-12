# Quick Reference - Supervised Learning Pipeline

## Start Training

```bash
cd /Users/vivekpandey/chess/attempt_18

# Edit run_example.sh to set paths:
# - STOCKFISH: Path to Stockfish binary
# - INITIAL_MODEL: Your starting model
# - FOUNDATION_DATA: Path to training_data_augmented.csv
# - SF_EVAL_LEVEL: Stockfish difficulty (start with 10)

./run_example.sh
```

## What Happens

1. **Iteration 1:** Establishes baseline (M0 vs SF level 0)
2. **Each iteration:**
   - Generate ~5K positions (policy network + SF labeling)
   - Mix with 30% foundation data → ~7K training examples
   - Train for 10 epochs
   - Evaluate candidate vs SF current level
   - **Promote if candidate > baseline**, else reject and reset trunk
3. **Auto-advance:** When baseline > 60%, level increases (0→1→2→...→8+)
4. **Level skipping:** If new level already >60%, skip immediately to next

## Training Automatically Advances At 60%

```
🎯 LEVEL COMPLETE!
  Baseline score: 61.2% (exceeds 60% threshold at level 3)

  Automatically advancing to Stockfish level 4
  Training continues without manual intervention

  Level 3 completed in 9 iterations
  New baseline at level 4 will be established...
```

## Continuing Training (If Interrupted)

If training was interrupted, continue from where you left off:

```bash
# Resume training from best model
python supervised_pipeline.py \
  --base-dir runs/continued_run \
  --model runs/run_01/models/model_best.pth \
  --stockfish /path/to/stockfish \
  --foundation-data /path/to/foundation.csv \
  --initial-level 5 \
  --foundation-ratio 0.3 \
  --iterations 100 \
  --games 100 \
  --epochs 10 \
  --device mps
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--initial-level` | 0 | Starting Stockfish level (auto-advances when >60%) |
| `--foundation-ratio` | 0.3 | % of foundation data (prevents forgetting) |
| `--iterations` | 100 | Max iterations (usually stops naturally at plateau) |
| `--games` | 100 | Games per iteration |
| `--epochs` | 10 | Training epochs per iteration |
| `--eval-games` | 100 | Evaluation games (more = less noise) |
| `--sf-depth` | 20 | SF search depth for labeling |
| `--sf-multipv` | 5 | Number of candidate moves from SF |

## Expected Timeline

Per iteration (~30-40 minutes):
- Data generation: ~20-30 min (100 games, 8 workers, SF depth 20)
- Training: ~5-10 min (10 epochs, 256 batch size)
- Evaluation: ~3-5 min (100 games vs SF)

Until 60% saturation (~7-10 iterations):
- **Total time: 3.5-6.5 hours per difficulty level**

## Progressive Training Roadmap (Automatic)

Training automatically progresses through levels starting from 0:

| Levels | Expected Behavior | Iterations Per Level |
|--------|------------------|---------------------|
| 0-2    | Skipped (too easy) | 0 (baseline already >60%) |
| 3-5    | Active training | 5-15 iterations each |
| 6-7    | Harder challenges | 15-30 iterations each |
| 8+     | Model plateaus | May not reach 60% (capacity ceiling) |

Note: Always start with `--initial-level 0` - system automatically finds right difficulty.

## Output Structure

```
runs/run_01/
├── data/
│   ├── iter1_raw.csv              # Generated positions
│   ├── iter1_train.csv            # Mixed with foundation (30%)
│   └── iter1_val.csv              # Validation split
├── models/
│   ├── model_M0.pth               # Initial model
│   ├── model_M1.pth               # Iteration 1 candidate
│   ├── model_best.pth             # Best promoted model
│   └── iter1/
│       └── model_epoch_10.pth     # Training checkpoints
└── logs/
    ├── training_log.json          # Full iteration history
    └── level_history.json         # Level progression tracking
```

## Promotion Logic

```python
if candidate_score > baseline_score:
    ✓ PROMOTE
    - Save as best model
    - Advance training trunk
    - Update baseline
else:
    ✗ REJECT
    - Discard candidate
    - Reset trunk to best model
    - Keep baseline unchanged
```

## Monitoring Progress

Watch for:
- **Baseline increasing:** Good! Model is improving
- **Many rejections:** Normal, model exploring
- **Baseline stuck:** May need more games or different hyperparams
- **Baseline > 60%:** Training stops, increase SF level

## Troubleshooting

**Problem:** Training stops immediately (baseline > 60% on iteration 1)
- **Fix:** Your model is already too strong for this SF level. Start with higher level.

**Problem:** Baseline not improving (stuck at X% for many iterations)
- **Fix:** Try increasing games (--games 200) or training longer (--epochs 20)

**Problem:** Many consecutive rejections
- **Fix:** Normal! Model is exploring. If > 10 rejections, consider:
  - Reducing learning rate (--lr 0.0005)
  - Increasing foundation ratio (--foundation-ratio 0.4)

**Problem:** Out of memory during training
- **Fix:** Reduce batch size (--batch-size 128 or 64)

## Example Session

```bash
# Single run - automatically progresses through all levels
./run_example.sh

# Output:
# - Skips levels 0-2 (already >60%)
# - Trains at level 3 until 60% (~5-10 iterations)
# - Auto-advances to level 4, trains until 60% (~10-15 iterations)
# - Auto-advances to level 5, trains until 60% (~15-20 iterations)
# - Continues until level 7-8 where model plateaus
# - Total time: varies based on model capacity
# - No manual intervention needed!
```

## Reading Training Logs

```json
{
  "iteration": 3,
  "model": "M3",
  "candidate_score": 0.489,
  "baseline_score": 0.456,
  "improvement": 0.033,
  "wins": 38,
  "draws": 20,
  "losses": 42,
  "promoted": true,
  "time_seconds": 1943.5
}
```

- `candidate_score`: New model's win rate
- `baseline_score`: Current baseline to beat
- `improvement`: Difference (positive = better)
- `promoted`: true if candidate advanced
- `stopped_early`: true if hit 60% threshold

## Complete Training Example

```bash
# Set up
cd /Users/vivekpandey/chess/attempt_18

# Start training with adaptive difficulty
python supervised_pipeline.py \
  --base-dir runs/run01 \
  --model ../attempt_17/best_model.pth \
  --stockfish /opt/homebrew/bin/stockfish \
  --foundation-data ../attempt_02/training_data_augmented.csv \
  --foundation-ratio 0.3 \
  --initial-level 0 \
  --iterations 100 \
  --games 100 \
  --epochs 10 \
  --device mps

# Training automatically progresses through levels:
# - Skips levels 0-2 if already >60%
# - Trains at level 3 until 60%
# - Auto-advances to level 4, trains until 60%
# - Continues until model plateaus or 100 iterations
# Output: runs/run01/models/model_best.pth
```

## Success Criteria

You know it's working when:
- ✓ Baseline increases over iterations
- ✓ Some models promoted, some rejected (not all one or the other)
- ✓ Training stops automatically at 60%
- ✓ Restarting at higher level shows initial drop (harder task) then climb back to 60%

This indicates **genuine improvement** through progressive curriculum!
