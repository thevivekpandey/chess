# Foundation Data Mixing

## Problem: Catastrophic Forgetting

When training with only policy-generated games + SF labels, the model may forget important chess knowledge from the original supervised training:

- **Your original model** was trained on 3.6M diverse examples:
  - 16M+ from games (various eval ranges)
  - 4M from puzzles (tactical patterns, mate in 1/2/3)
  - Balanced distribution across eval ranges

- **New SF-labeled data** comes from policy network games:
  - Only ~5K positions per iteration (100 games)
  - Biased by current policy (may miss tactics, endgames, extreme positions)
  - Risk of forgetting diverse patterns

**Result:** Model might lose tactical sharpness, miss mates, or perform poorly on positions not similar to its own games.

## Solution: Mix Foundation Data

We now support mixing your original 3.6M examples with new SF-labeled data to maintain diverse knowledge while improving from SF.

### How it Works

```
Training Data = 30% Foundation + 70% New SF Data
```

For each iteration:
1. Generate ~5K new positions with policy + SF labeling
2. Sample ~2.1K positions from foundation dataset (30%)
3. Combine and shuffle → ~7.1K total training examples
4. Train on mixed dataset

This ensures the model:
- ✓ Maintains tactical knowledge (puzzles, mates)
- ✓ Remembers diverse positions (various eval ranges)
- ✓ Learns new SF preferences (70% new data)
- ✓ Doesn't overfit to its own policy mistakes

## Usage

### With Foundation Data (Recommended)

```bash
python supervised_pipeline.py \
  --base-dir runs/run_01 \
  --model ../attempt_17/best_model.pth \
  --stockfish /path/to/stockfish \
  --foundation-data ../attempt_02/training_data_augmented.csv \
  --foundation-ratio 0.3 \
  --initial-level 0 \
  --iterations 100 \
  --games 100 \
  --epochs 10 \
  --device mps
```

### Without Foundation Data (Not Recommended - May Forget)

```bash
python supervised_pipeline.py \
  --base-dir runs/run_01 \
  --model ../attempt_17/best_model.pth \
  --stockfish /path/to/stockfish \
  --initial-level 0 \
  --iterations 100 \
  --games 100 \
  --epochs 10 \
  --device mps
```

## Foundation Ratio Tuning

The `--foundation-ratio` parameter controls how much foundation data to include:

| Ratio | Foundation | New Data | Use Case |
|-------|-----------|----------|----------|
| 0.0   | 0%        | 100%     | No mixing (may forget) |
| 0.2   | 20%       | 80%      | Aggressive SF learning |
| 0.3   | 30%       | 70%      | **Recommended balance** |
| 0.4   | 40%       | 60%      | Conservative (more retention) |
| 0.5   | 50%       | 50%      | Equal mix |

**Recommendation:** Start with 0.3 (30% foundation). Adjust based on:
- If tactical accuracy drops → increase to 0.4-0.5
- If not improving vs SF → decrease to 0.2
- If improving well → keep at 0.3

## Monitoring

Track these metrics to ensure mixing is working:

### 1. Tactical Test Suite
Run your model on a tactical test set (mate in 1/2/3):
```bash
# Assuming you have a tactics test file
python test_tactics.py --model runs/run_01/models/model_M3.pth
```

**Expected:** Accuracy should stay constant or improve (not degrade)

### 2. Eval Distribution
Check the distribution of evals in generated positions:
```bash
# In your training logs
grep "Value statistics" runs/run_01/logs/*.txt
```

**Expected:** Values should span -1 to +1, not cluster around 0

### 3. Performance vs SF
Standard metric from eval:
```bash
# Already tracked in training_log.json
cat runs/run_01/logs/training_log.json | grep score
```

**Expected:** Steady improvement across iterations

## Implementation Details

### Data Format

All data uses CSV format for consistency:
```csv
fen,eval,move1,score1,move2,score2,move3,score3,move4,score4,move5,score5
r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4,0.31,c4b5,56,c4e2,54,c4b3,42,c4d3,37,c4a2,25
```

- `eval`: Position evaluation in pawns (from SF)
- `move1-5`: Top 5 moves from SF multi-PV
- `score1-5`: Centipawn scores for each move

### Mixing Strategy

The `mix_datasets.py` script:
1. Reads foundation CSV (3.6M rows)
2. Reads new SF CSV (~5K rows)
3. Samples foundation to achieve desired ratio
4. Combines and shuffles
5. Splits into train/val (90/10)

### Training Dataset

The `StockfishDataset` class (in `train_supervised.py`):
1. Reads CSV rows
2. Converts eval to value target: `value = tanh(eval / 4.0)`
3. Converts moves+scores to policy target using softmax
4. Returns (board_tensor, policy_target, value_target)

## Files

- `mix_datasets.py` - Utility to mix foundation + new data
- `supervised_pipeline.py` - Main pipeline (updated to support mixing)
- `train_supervised.py` - Training script (updated to read CSV)
- `generate_sf_training_data.py` - Data generator (updated to output CSV)

## Example Output

```
Mixing datasets...
  Foundation: ../attempt_02/training_data_augmented.csv
  New data: runs/run_01/data/iter1_raw.csv
  Foundation ratio: 30%
  Val split: 10%

Loading datasets...
  Foundation: 3600000 examples
  New data: 5234 examples
  Sampled foundation: 2243 examples
  Mixed total: 7477 examples
  Train: 6729 examples
  Val: 748 examples

Final statistics:
  Train - Foundation: 2018 (30.0%)
  Train - New: 4711 (70.0%)

✓ Saved train to: runs/run_01/data/iter1_train.csv
✓ Saved val to: runs/run_01/data/iter1_val.csv
```

## Benefits

1. **Prevents catastrophic forgetting** - Maintains tactical and positional knowledge
2. **Balanced learning** - 70% SF improvement + 30% knowledge retention
3. **Adaptive** - Can adjust ratio based on observed metrics
4. **Proven approach** - Used by Leela Chess Zero and other projects
5. **Minimal overhead** - Sampling is fast (~1-2 seconds)

## Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **No mixing** | Faster iterations | May forget tactics, mates |
| **30% foundation** | Balanced improvement | Slightly slower training |
| **50% foundation** | Maximum retention | Slower SF learning |

**Recommended:** Start with 30% foundation mixing for best balance of improvement and retention.
