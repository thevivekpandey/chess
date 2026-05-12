# Promotion System with Adaptive Difficulty

## Overview

The training pipeline uses a **baseline-comparison promotion system** combined with **adaptive difficulty progression**. Models are promoted only when they beat the current baseline, and Stockfish difficulty automatically increases as the model improves.

## Core Promotion Logic

### Baseline Comparison

At each Stockfish level, we maintain a baseline score that candidates must beat:

```
Baseline (best model at level K): 45.3%
Candidate (new model):            48.7%
Improvement:                      +3.4%

✓ PROMOTED (candidate > baseline)
```

**If promoted:**
- Candidate becomes new best model
- Training trunk advances to promoted model
- Baseline updates to new score (48.7%)

**If rejected:**
- Candidate discarded
- **Training trunk resets** to best model (prevents compounding degradation)
- Baseline stays unchanged (45.3%)

This is exactly the AlphaZero approach - only promote on genuine improvement.

## Adaptive Difficulty Integration

### Level Progression

When baseline exceeds 60% at current level:

```
Iteration 12: Level 3, Score 62.1% ✓ PROMOTED

🎯 Baseline exceeds 60%! Advancing to Stockfish level 4
```

The system:
1. Records level 3 completion (62.1%)
2. Advances to level 4
3. **Resets baseline** (will be established at level 4)
4. Continues training

### First Evaluation at New Level

When entering a new level, establish baseline:

```
Iteration 13: Level 4
  Establishing baseline at level 4...
  Current best model vs SF L4: 42.7%

  Baseline set: 42.7%
```

If baseline already > 60%, skip training entirely:

```
Iteration 2: Level 1
  Establishing baseline at level 1...
  Current best model vs SF L1: 87.2%

🚀 Already exceeds 60%! Skipping training, advancing to level 2
```

## Complete Flow

### Iteration Workflow

```
ITERATION N at Level K:

1. Generate training data (policy + SF labeling)
2. Mix with foundation data (30%)
3. Train for 10 epochs
4. Evaluate candidate vs SF level K

5. PROMOTION DECISION:
   if candidate_score > baseline_score:
       ✓ PROMOTE
       - Update baseline = candidate_score
       - Advance trunk to candidate

       if baseline_score > 0.60:
           🎯 ADVANCE LEVEL
           - Record level K completion
           - K = K + 1
           - Reset baseline = None
   else:
       ✗ REJECT
       - Discard candidate
       - Reset trunk to best model
       - Keep baseline unchanged

6. Continue to next iteration
```

### Example Progression

```
Level 3:
  Iteration 4:  Baseline 48.3%
  Iteration 5:  Score 51.2% ✓ PROMOTED (baseline → 51.2%)
  Iteration 6:  Score 49.8% ✗ REJECTED (baseline stays 51.2%)
  Iteration 7:  Score 53.6% ✓ PROMOTED (baseline → 53.6%)
  ...
  Iteration 12: Score 62.1% ✓ PROMOTED (baseline → 62.1%)
                🎯 Advancing to level 4

Level 4:
  Iteration 13: Establish baseline 42.7%
  Iteration 14: Score 45.1% ✓ PROMOTED (baseline → 45.1%)
  ...
```

## Why This Works

### 1. Prevents False Progress

Without baseline comparison:
```
M1: 50% (lucky run)
M2: 48% (promoted because > initial 45%)
M3: 46% (promoted because > initial 45%)
```
Model degrading but still "better than M0"!

With baseline comparison:
```
M1: 50% ✓ PROMOTED (baseline → 50%)
M2: 48% ✗ REJECTED (< baseline 50%)
M3: 52% ✓ PROMOTED (baseline → 52%)
```
Only promotes on genuine improvement.

### 2. Prevents Compounding Degradation

**Critical feature:** Trunk resets on rejection.

Bad approach (attempt_17):
```
M1 rejected → but trunk still at M1
M2 trains from M1 → makes it worse
M3 trains from M2 → even worse
Compounding degradation!
```

Good approach (attempt_18):
```
M1 rejected → trunk resets to M0
M2 trains from M0 → fresh attempt
M3 trains from M0 → another fresh attempt
No compounding!
```

### 3. Provides Curriculum Learning

Adaptive difficulty creates natural curriculum:

```
Level 3: Model struggles, needs ~10 iterations to reach 60%
Level 4: Model struggles more, needs ~15 iterations
Level 5: Model struggles even more, needs ~20 iterations
...
Level 8: Model plateaus, can't reach 60% (capacity ceiling)
```

Model learns progressively harder challenges.

## Comparison with AlphaZero

From the AlphaZero paper (Silver et al., 2017):

> "The best player always uses the latest network; when the network is updated, its win rate is evaluated against the previous best network. If it wins by a margin greater than 55%, it replaces the best network."

Our implementation:

| AlphaZero | attempt_18 |
|-----------|------------|
| Threshold: > 55% | Threshold: > 50% (strictly greater) |
| Fixed opponent strength | Adaptive opponent (SF level increases) |
| Self-play for data | Policy + SF labeling |
| Manual stop | Continues through levels automatically |

Core idea is identical: **only promote when candidate beats current best**.

## Output Format

### Iteration Log

```
Iter   Model    Level  Score    Baseline    Δ        W    D    L    Status
-------------------------------------------------------------------------------------
4      M4       3      0.512    0.483       +0.029   38   26   36   ✓ PROMOTED
5      M5       3      0.498    0.512       -0.014   36   25   39   ✗ REJECTED
6      M6       3      0.538    0.512       +0.026   41   23   36   ✓ PROMOTED
...
12     M12      3      0.621    0.615       +0.006   48   19   33   ✓ PROMOTED [LVL+]
13     M13      4      0.451    0.427       +0.024   32   27   41   ✓ PROMOTED
```

Legend:
- `✓ PROMOTED` - Candidate beats baseline
- `✗ REJECTED` - Candidate doesn't beat baseline
- `[LVL+]` - Level advanced after this iteration

### Level History

```json
{
  "level": 3,
  "final_score": 0.621,
  "iterations": 9,
  "skipped": false
}
```

- `final_score`: Best score achieved at this level
- `iterations`: Number of training iterations
- `skipped`: true if level was too easy (baseline already > 60%)

## Practical Implications

### Promotion Rate

Expect ~40-60% promotion rate:
- Too many promotions (>80%) → training too conservative, increase learning rate
- Too many rejections (>80%) → training too aggressive, decrease learning rate or add more data

### Iterations Per Level

Typical pattern:
- Easy levels (0-2): 0 iterations (skipped)
- Medium levels (3-5): 10-20 iterations each
- Hard levels (6-8): 20-50 iterations each
- Plateau level: 50+ iterations, never reaches 60%

### Training Trunk Behavior

The trunk (what we train from) follows this pattern:

```
Iteration 1: Trunk = M0, trains → M1
  M1 promoted → Trunk = M1

Iteration 2: Trunk = M1, trains → M2
  M2 rejected → Trunk = M1 (reset)

Iteration 3: Trunk = M1, trains → M3
  M3 promoted → Trunk = M3

Iteration 4: Trunk = M3, trains → M4
  M4 promoted → Trunk = M4
```

Trunk always represents the "best known model" at current level.

## Advantages of Combined System

1. **Automatic curriculum** - Difficulty increases as model improves
2. **Genuine improvement** - Only promotes when truly better
3. **Stable training** - Trunk resets prevent degradation
4. **Clear progress metric** - Can see advancement through levels
5. **Natural stopping point** - Training plateaus when model reaches capacity
6. **No manual intervention** - Runs fully automatically

## Monitoring Tips

### Good Signs
- ✓ Baseline increasing within each level
- ✓ Mix of promotions and rejections (40-60% each)
- ✓ Levels advancing every 10-30 iterations
- ✓ Higher levels take more iterations

### Warning Signs
- ⚠️ Baseline not improving for 20+ iterations
- ⚠️ All promotions (training too easy)
- ⚠️ All rejections (training too hard)
- ⚠️ Stuck at same level for 50+ iterations

### Action Items
- If stuck: Try more data (--games 200), longer training (--epochs 20)
- If all rejections: Lower learning rate (--lr 0.0005)
- If all promotions: Higher learning rate (--lr 0.002)
- If forgetting: Increase foundation ratio (--foundation-ratio 0.4)

## Summary

The promotion system ensures:
1. **Only genuine improvement** gets promoted
2. **Failed attempts don't compound** (trunk reset)
3. **Difficulty auto-adjusts** (level progression)
4. **Training is fully automatic** (no manual stops)

This creates a robust, self-managing training pipeline that progressively challenges the model until it reaches its natural capacity ceiling.
