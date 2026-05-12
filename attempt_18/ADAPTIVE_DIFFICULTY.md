# Adaptive Difficulty System

## Overview

The pipeline now uses **automatic difficulty progression** - no need to manually set or increase Stockfish level. The system starts at level 0 and automatically advances when the model exceeds 60% win rate.

## How It Works

### 1. Start at Level 0

```
python supervised_pipeline.py \
  --base-dir runs/adaptive_run01 \
  --model initial_model.pth \
  --stockfish /path/to/stockfish \
  --foundation-data foundation.csv \
  --initial-level 0 \
  --iterations 100
```

The system starts at Stockfish level 0 (very weak).

### 2. Evaluate at Current Level

After each training iteration:
- Evaluate candidate model vs Stockfish at current level
- Compare against baseline (best model at this level)
- Promote if candidate beats baseline

###3. Auto-Advance When > 60%

When baseline exceeds 60% win rate:
```
✓ PROMOTED! Candidate beats baseline.
  Baseline: 62.3%

🎯 Baseline exceeds 60%! Advancing to Stockfish level 1
```

The system automatically:
- Records level completion (level 0 → 62.3%)
- Advances to next level (1)
- Resets baseline (will be established at level 1)
- Continues training

### 4. Skip Levels if Already Strong

If model is already strong at a level:
```
Iteration X: Establishing baseline at level 5...
  Baseline score: 0.823 (82.3%)

🚀 Baseline already exceeds 60%! Skipping training, advancing to level 6
```

No training happens - immediately advances to next level.

### 5. Continue Until Plateau

Training continues automatically through levels:
- Level 0 → Level 1 → Level 2 → ... → Level 7
- Eventually model will struggle to reach 60% (plateau)
- Runs until max iterations (default: 100) or manual stop

## Example Session

```bash
./run_example.sh
```

**Output:**
```
ITERATION 1
  Establishing baseline at level 0...
  Baseline: 98.5%

🚀 Baseline already exceeds 60%! Advancing to level 1

ITERATION 2
  Establishing baseline at level 1...
  Baseline: 87.2%

🚀 Baseline already exceeds 60%! Advancing to level 2

ITERATION 3
  Establishing baseline at level 2...
  Baseline: 71.5%

🚀 Baseline already exceeds 60%! Advancing to level 3

ITERATION 4
  Establishing baseline at level 3...
  Baseline: 48.3%

  Evaluating candidate (M4)...
  Score: 51.2%

✓ PROMOTED! Improvement: +2.9%

ITERATION 5
  Candidate: 53.8%
✓ PROMOTED!

...

ITERATION 12
  Candidate: 62.1%
✓ PROMOTED!

🎯 Baseline exceeds 60%! Advancing to level 4

ITERATION 13
  Establishing baseline at level 4...
  Baseline: 42.7%

...continues until model plateaus...
```

## Expected Progression

For a 20M parameter model without MCTS:

| Level | Approx ELO | Expected | Notes |
|-------|-----------|----------|-------|
| 0-2   | <1500     | Skip immediately | Too easy |
| 3     | ~1600     | Train 5-10 iterations | First challenge |
| 4     | ~1750     | Train 10-15 iterations | |
| 5     | ~1900     | Train 15-20 iterations | |
| 6     | ~2050     | Train 20-30 iterations | Getting harder |
| 7     | ~2200     | Train 30-50 iterations | Near ceiling |
| 8     | ~2350     | May not reach 60% | Plateau |

Total: ~100-150 iterations across all levels.

## Advantages

### 1. **Zero Configuration**
- No need to guess starting level
- No manual level adjustments
- Just run and let it find the right difficulty

### 2. **Automatic Curriculum**
- Progressively harder challenges
- Skips levels that are too easy
- Trains more at appropriate difficulty

### 3. **Continuous Progress**
- Never stops at arbitrary iteration count
- Runs until model naturally plateaus
- Clear measure of model capacity (max level reached)

### 4. **Level History Tracking**

The system saves complete progression:

```json
// level_history.json
[
  {
    "level": 0,
    "final_score": 0.985,
    "iterations": 0,
    "skipped": true
  },
  {
    "level": 1,
    "final_score": 0.872,
    "iterations": 0,
    "skipped": true
  },
  {
    "level": 3,
    "final_score": 0.623,
    "iterations": 9,
    "skipped": false
  },
  ...
]
```

## Training Output

```
TRAINING COMPLETE
================================================================================
Status: Completed 47 iterations

Adaptive Difficulty Progression:
  Level 0: 98.5% (SKIPPED - already >60%)
  Level 1: 87.2% (SKIPPED - already >60%)
  Level 2: 71.5% (SKIPPED - already >60%)
  Level 3: 62.3% (9 iterations)
  Level 4: 61.8% (14 iterations)
  Level 5: 60.2% (18 iterations)
  Current Level: 6
  Current Baseline: 47.3%

Results summary:
Iter   Model    Level  Score    Baseline    Δ        W    D    L    Status
-------------------------------------------------------------------------------------
1      M1       0      0.985    0.985       +0.000   97   3    0    ✗ REJECTED [LVL+]
2      M2       1      0.872    0.872       +0.000   84   9    7    ✗ REJECTED [LVL+]
3      M3       2      0.715    0.715       +0.000   65   22   13   ✗ REJECTED [LVL+]
4      M4       3      0.512    0.483       +0.029   38   26   36   ✓ PROMOTED
5      M5       3      0.538    0.512       +0.026   41   23   36   ✓ PROMOTED
...
22     M22      3      0.623    0.615       +0.008   48   19   33   ✓ PROMOTED [LVL+]
23     M23      4      0.427    0.427       +0.000   30   25   45   ✓ PROMOTED
...

Best model: runs/adaptive_run01/models/model_best.pth
Reached Stockfish level: 6
Current baseline at L6: 47.3%
```

## When Model Plateaus

Model will eventually struggle to improve at a certain level:

```
Iteration 95: Level 7, Score 48.2% ✗ REJECTED
Iteration 96: Level 7, Score 49.1% ✗ REJECTED
Iteration 97: Level 7, Score 47.8% ✗ REJECTED
...
```

This indicates:
- Model has reached its capacity limit
- Needs more parameters, better architecture, or MCTS to go further
- Level 7 is the "ceiling" for this model configuration

## Comparison: Fixed vs Adaptive

### Fixed Level (Old Approach)
```
User: "Start at level 10"
→ Baseline 42% at L10
→ Train until 60%
→ Stop
User: "Now start at level 15"
→ Repeat
```

**Problems:**
- Need to guess starting level
- Manual restarts
- Wasted time if level too easy/hard

### Adaptive Level (New Approach)
```
System: "Starting at level 0"
→ Skips levels 0-2 (too easy)
→ Trains at level 3 until 60%
→ Auto-advances to level 4
→ Trains at level 4 until 60%
→ ...continues...
→ Plateaus at level 7 (47%)
```

**Benefits:**
- ✓ Zero configuration
- ✓ Automatic progression
- ✓ Finds model's natural ceiling
- ✓ Complete history of all levels

## Usage

### Basic (Recommended)

```bash
./run_example.sh
```

Starts at level 0, auto-advances, runs until iteration 100 or manual stop.

### Custom Starting Level

If you know your model is already strong:

```bash
python supervised_pipeline.py \
  --base-dir runs/run01 \
  --model strong_model.pth \
  --stockfish /path/to/stockfish \
  --foundation-data foundation.csv \
  --initial-level 5 \
  --iterations 100
```

Starts at level 5 instead of 0 (skips easy levels).

### Continuing From Previous Run

If training was interrupted:

```bash
python supervised_pipeline.py \
  --base-dir runs/run02 \
  --model runs/run01/models/model_best.pth \
  --stockfish /path/to/stockfish \
  --foundation-data foundation.csv \
  --initial-level 6 \  # Start where you left off
  --iterations 100
```

## Monitoring Progress

Key metrics to watch:

1. **Level progression** - Are levels advancing?
2. **Iterations per level** - More iterations = model learning
3. **Current baseline** - Is it climbing toward 60%?
4. **Rejection rate** - Many rejections = model exploring

Good signs:
- ✓ Levels advance every 5-20 iterations
- ✓ Baseline increases within each level
- ✓ Mix of promotions and rejections

Bad signs:
- ✗ Stuck at same level for 50+ iterations
- ✗ Baseline not improving
- ✗ All rejections or all promotions

## Complete Feature Set

The adaptive system combines:

1. ✅ **Policy-only game generation** - Fast data (10ms vs 3-5s)
2. ✅ **Stockfish labeling** - Strong teacher signal
3. ✅ **Foundation data mixing** - Prevents forgetting (30%)
4. ✅ **Baseline promotion** - Only genuine improvement
5. ✅ **Trunk reset on rejection** - No compounding degradation
6. ✅ **Adaptive difficulty** - Auto-advances through levels
7. ✅ **Level skipping** - Skips trivial levels
8. ✅ **Progress tracking** - Complete level history

This creates a **fully automatic training pipeline** that finds your model's natural strength ceiling!
