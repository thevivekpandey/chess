# Supervised Learning with Stockfish (Plan 2)

This directory implements **Plan 2** from our chess training strategy: supervised learning from Stockfish as a teacher.

## Approach

Unlike self-play RL (which suffered from catastrophic forgetting), this approach uses:

1. **Policy-only game generation**: Play games using just the policy network (no MCTS) - very fast (~10ms per move)
2. **Stockfish labeling**: Label each position with SF's multi-PV analysis and evaluation
3. **Supervised learning**: Train on SF's labels using cross-entropy (policy) + MSE (value)

**Key advantages:**
- Much faster than MCTS-based generation (10ms vs 3-5s per move)
- Direct learning from a strong teacher (Stockfish)
- Foundation data mixing prevents catastrophic forgetting (30% diverse positions)
- Baseline promotion system ensures genuine improvement
- Adaptive difficulty progression through SF levels (0→1→2→...→8+)
- Diverse positions from policy mistakes + SF corrections

## Files

- `generate_sf_training_data.py` - Generate games and label with Stockfish (CSV output)
- `train_supervised.py` - Train on Stockfish-labeled data (CSV format)
- `mix_datasets.py` - Mix foundation data with new data (prevents forgetting)
- `evaluate_vs_stockfish.py` - Test model strength vs Stockfish
- `supervised_pipeline.py` - Full training pipeline with adaptive difficulty
- `chess_engine.py` - Neural network architecture (copied from attempt_17)
- `run_example.sh` - Example script to run complete pipeline

## Requirements

```bash
pip install torch chess python-chess tqdm numpy
```

You also need:
- Stockfish binary (download from https://stockfishchess.org/)
- Initial model checkpoint (e.g., from attempt_17)

## Quick Start

### Option 1: Full Pipeline (Recommended)

**Easiest way - use the example script:**

```bash
# Edit run_example.sh to set your paths
./run_example.sh
```

**Or run directly:**

```bash
python supervised_pipeline.py \
  --base-dir runs/run_01 \
  --model /path/to/initial_model.pth \
  --stockfish /path/to/stockfish \
  --foundation-data /path/to/foundation.csv \
  --foundation-ratio 0.3 \
  --initial-level 0 \
  --iterations 100 \
  --games 100 \
  --epochs 10 \
  --device mps
```

This will:
1. Start at Stockfish level 0 (auto-advances when win rate > 60%)
2. Each iteration:
   - Generate 100 games with policy network
   - Label positions with Stockfish (depth 20, multi-PV 5)
   - Mix with 30% foundation data (prevents forgetting)
   - Train for 10 epochs
   - Evaluate vs Stockfish at current level
   - Promote only if candidate beats baseline
3. Automatically advance through levels until model plateaus

**Output structure:**
```
runs/run_01/
├── data/
│   ├── iter1_raw.csv
│   ├── iter1_train.csv
│   ├── iter1_val.csv
│   └── ...
├── models/
│   ├── model_M0.pth (initial)
│   ├── model_M1.pth
│   ├── model_best.pth
│   └── iter1/
│       └── model_epoch_10.pth
└── logs/
    ├── training_log.json
    └── level_history.json
```

### Option 2: Step-by-Step

#### Step 1: Generate Training Data

```bash
python generate_sf_training_data.py \
  --model /path/to/model.pth \
  --output data/training_data.csv \
  --stockfish /path/to/stockfish \
  --games 100 \
  --workers 8 \
  --temperature 0.8 \
  --sf-depth 20 \
  --sf-multipv 5 \
  --device mps
```

**Parameters:**
- `--games`: Number of games to generate (100 games ≈ 4000-6000 positions)
- `--workers`: Parallel workers for SF labeling (8 recommended)
- `--temperature`: Policy sampling temperature (0.8 = some randomness for diversity)
- `--sf-depth`: Stockfish search depth (20 = strong, ~0.5s per position)
- `--sf-multipv`: Number of candidate moves from SF (5 = top 5 moves)

**Time estimate:** 100 games with 8 workers ≈ 20-30 minutes

#### Step 2: Mix with Foundation Data (Optional but Recommended)

```bash
python mix_datasets.py \
  --foundation /path/to/foundation_data.csv \
  --new-data data/training_data.csv \
  --output-train data/train.csv \
  --output-val data/val.csv \
  --foundation-ratio 0.3
```

This mixes 30% foundation data with 70% new data to prevent catastrophic forgetting.

#### Step 3: Train Model

```bash
python train_supervised.py \
  --train-data data/train.csv \
  --val-data data/val.csv \
  --model /path/to/initial_model.pth \
  --output models/iter1 \
  --epochs 10 \
  --batch-size 256 \
  --lr 0.001 \
  --device mps
```

**Training time:** ~5-10 minutes per epoch on MPS (M1/M2 Mac)

#### Step 4: Evaluate

```bash
# Single evaluation
python evaluate_vs_stockfish.py \
  --model models/iter1/model_best.pth \
  --stockfish /path/to/stockfish \
  --games 100 \
  --level 10 \
  --device mps

# Progressive evaluation (multiple levels)
python evaluate_vs_stockfish.py \
  --model models/iter1/model_best.pth \
  --stockfish /path/to/stockfish \
  --games 50 \
  --progressive \
  --device mps
```

## Hyperparameters

### Data Generation
- **Games per iteration**: 100-200 (balance speed vs diversity)
- **Temperature**: 0.8 (some randomness for exploration)
- **SF depth**: 20 (strong but not too slow)
- **SF multi-PV**: 5 (captures main candidate moves)

### Training
- **Epochs**: 10 (balance overfitting vs underfitting)
- **Batch size**: 256 (fits well on GPU/MPS)
- **Learning rate**: 0.001 (Adam default)
- **Loss weights**: 1.0 policy + 1.0 value (equal weighting)

### Evaluation
- **Games**: 100+ (50 as White, 50 as Black)
- **SF level**: 10 (medium strength, ~1800-2000 ELO)
- **SF time**: 0.1s per move (fast evaluation)

## Expected Results

Based on similar approaches (e.g., "Grandmaster-Level Chess Without Search" paper):

- **After 1 iteration**: Modest improvement, model learns basic SF preferences
- **After 3-5 iterations**: Noticeable improvement in tactical play
- **After 10+ iterations**: Should reach ~Stockfish level 5-10 strength (policy-only)

**Key metric:** Score vs Stockfish level 10 (50% = equal strength)

## Design Decisions

### Why policy-only game generation?

**Pros:**
- 200-500x faster than MCTS (10ms vs 3-5s per move)
- SF corrects mistakes, so weak positions still provide learning signal
- More diverse positions (policy makes errors, explores broadly)

**Cons:**
- Some unrealistic positions (but SF still labels them correctly)
- Distribution mismatch with target inference (if using MCTS later)

**Verdict:** For supervised learning, speed + diversity > perfect realism

### Why Stockfish multi-PV?

SF doesn't naturally output a policy distribution (just best move). Multi-PV gives us:
- Top 5 moves with evaluations
- Convert to probabilities via softmax over scores
- Captures "these moves are roughly equal" vs "this is clearly best"

Alternative: Use uniform distribution over legal moves + weight by SF scores (simpler but loses information)

### Why depth 20?

- **Depth 15**: Fast (~0.2s) but misses some tactics
- **Depth 20**: Strong (~0.5s) with good tactical vision ← **Recommended**
- **Depth 25**: Very strong (~1-2s) but slow

Trade-off: Deeper = better labels but slower generation. Depth 20 is the sweet spot.

### Why temperature 0.8?

- **Temperature 0.1**: Near-greedy, follows policy closely (less diversity)
- **Temperature 0.8**: Some randomness, explores more ← **Recommended**
- **Temperature 1.5**: High randomness, reaches bad positions (might waste SF compute)

We want diverse but not completely random games.

## Comparison with Self-Play (Plan 1)

| Aspect | Self-Play (attempt_17) | Supervised (attempt_18) |
|--------|------------------------|-------------------------|
| **Speed** | 3-5s per move (MCTS) | 10ms per move (policy) |
| **Learning signal** | Self-play outcomes | Stockfish labels |
| **Stability** | Catastrophic forgetting | Stable (external teacher) |
| **Exploration** | MCTS search | Policy errors + randomness |
| **Data efficiency** | Needs replay buffer | Can train on all data |
| **Failure mode** | Model degradation | Plateaus at SF knowledge |

## Troubleshooting

### Data generation is slow

- Reduce `--workers` if CPU-bound (memory limit)
- Reduce `--sf-depth` to 15 (faster but weaker labels)
- Reduce `--games` (quality > quantity)

### Training loss not decreasing

- Check data quality (print some examples)
- Reduce learning rate (try 0.0001)
- Increase epochs (try 20)
- Check for bugs in policy target conversion

### Model not improving vs Stockfish

- Generate more data (try 200-500 games per iteration)
- Increase SF depth to 25 (better labels)
- Train longer (try 20 epochs)
- Check evaluation setup (ensure fair time controls)

### Out of memory

- Reduce batch size (try 128 or 64)
- Reduce workers (try 4)
- Free up GPU memory (close other apps)

## Next Steps

After training with Plan 2, you can:

1. **Test vs Plan 1**: Compare supervised learning vs self-play RL
2. **Hybrid approach**: Use supervised learning to bootstrap, then switch to self-play
3. **Plan 3**: Learn from blunders (human-like analysis)
4. **Increase target strength**: Use deeper SF (depth 30) or newer SF versions
5. **Policy distillation with MCTS**: Train policy to mimic MCTS search (bridge to AlphaZero)

## References

- "Grandmaster-Level Chess Without Search" - Policy-based chess with supervised learning
- AlphaZero paper - Original self-play approach (Plan 1)
- Leela Chess Zero - Combination of self-play + supervised learning
