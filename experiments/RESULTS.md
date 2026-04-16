# Results (April 2026)

All results below were produced after the label-alignment fix. Previous results from the buggy codebase are superseded.

## Bug Fixed: Double-Shift in Loss Computation

A loss-alignment bug affected all example scripts. `targets = ids[1:seq_len+1]` already encodes the next-token shift, but the loss applied a second shift (`logits[:-1]` vs `targets[1:]`), supervising every position against the wrong token.

**Impact by masking mode:**
- **Label-only** (SST-2, IMDB): total signal loss on the single supervised position. The mask landed on the colon instead of the answer token. Non-DP partially recovered via attention bleed across positions; DP clipped that indirect channel, causing the 16pp gap we originally observed.
- **Full-sequence** (canary, PubMedQA): 1/seq_len of supervision wasted per position. Still degraded memorization signal — canary non-DP AUC jumped 0.79 → 0.917, and PubMedQA non-DP AUC rose from 0.75–0.80 to 0.861 after the fix. The alignment fix made both the attack stronger and DP more clearly effective.

**Files changed:** `sst2_dp.py`, `imdb_dp.py`, `canary_frontier.py`, `pubmedqa_dp.py`, `experiments/mia/{train,demo,attack}.py`, `tests/test_lora_dp.py`

**Additional fixes:**
- DP loss logging moved from post-update to pre-update (consistent with non-DP)
- SST-2 MIA pool changed from train-vs-validation splits to disjoint subsets of `raw["train"]`
- Regression tests added: `tests/test_label_alignment.py`

## Gradient Norm Diagnostic

Post-fix per-sample gradient norms on SST-2 (label-only, 2 batches of 16):

| Percentile | Norm |
|---|---|
| min | 0.75 |
| p25 | 1.28 |
| median | 2.34 |
| p75 | 10.89 |
| p90 | 29.77 |
| max | 32.08 |

**C=1.0 clips 88% of samples.** The current results are conservative — raising the clip toward the median (~2.5) would preserve more gradient signal without materially increasing privacy cost. This is free utility left on the table, not a fundamental DP limitation.

## Canary Frontier

Qwen2.5-0.5B LoRA, 400 canaries (200 train / 200 test), 5 epochs, seed 42.

| Setting | ε | ROC-AUC | Bal. Accuracy | TPR@1%FPR | Mem. Loss | NM Loss |
|---|---|---|---|---|---|---|
| non_dp | ∞ | **0.917** | 0.843 | 0.545 | 0.582 | 0.673 |
| dp_mid | 15.43 | 0.507 | 0.525 | 0.005 | 1.346 | 1.353 |
| dp_strong | 1.08 | 0.514 | 0.535 | 0.010 | 2.969 | 2.975 |

Artifacts: `experiments/canary/baseline/results.json`

## SST-2 Sentiment

Qwen2.5-0.5B LoRA, 2000 train / 800 test (balanced, same source pool), label-only mask, seq_len=96, logical_batch=16, microbatch=4, 2 epochs. **5 seeds.**

| Setting | ε | Accuracy | MIA AUC |
|---|---|---|---|
| non_dp | ∞ | **87.4 ± 0.9%** | 0.661 |
| dp_mid | 8.76 | **82.5 ± 2.1%** | 0.507 |

Utility gap: **5.0pp ± 2.8pp** (range 0.5–7.9pp).

Per-seed breakdown:

| Seed | non_dp Acc | dp_mid Acc | Gap | dp_mid MIA AUC |
|---|---|---|---|---|
| 42 | 88.75% | 80.87% | 7.9pp | 0.514 |
| 123 | 86.62% | 82.00% | 4.6pp | 0.515 |
| 456 | 86.88% | 86.38% | 0.5pp | 0.485 |
| 7 | 86.88% | 81.25% | 5.6pp | 0.514 |
| 99 | 88.00% | 81.87% | 6.1pp | 0.509 |

The seed variance is real: DP noise dominates the per-seed gap at this training budget. Every seed has MIA AUC near 0.50.

Artifacts: `experiments/sst2/baseline/seed{42,123,456,7,99}.json`

## IMDB Sentiment

Qwen2.5-0.5B LoRA, 1500 train / 1500 test, label-only mask, seq_len=128, logical_batch=8, microbatch=2, 5 epochs. **3 seeds.**

| Setting | ε | Accuracy | MIA AUC | Mem. Loss | NM Loss |
|---|---|---|---|---|---|
| non_dp | ∞ | **78.0 ± 0.8%** | 0.926 | 0.025 | 0.811 |
| dp_mid | 9.72 | **77.3 ± 1.7%** | 0.500 | 0.859 | 0.838 |

Utility gap: **0.6pp ± 2.4pp** (range -1.9 to 3.0pp). DP outperforms non-DP on seed 123 — noise acts as a regularizer when non-DP overfits at 5 epochs.

The non-DP member loss (0.025) is 32x lower than nonmember loss (0.811). This is severe memorization on a real classification task: the model has memorized which reviews it saw, detectable at MIA AUC 0.926, even though accuracy looks normal.

Per-seed breakdown:

| Seed | non_dp Acc | dp_mid Acc | Gap | non_dp MIA | dp_mid MIA |
|---|---|---|---|---|---|
| 42 | 78.5% | 75.5% | 3.0pp | 0.890 | 0.497 |
| 123 | 76.9% | 78.8% | -1.9pp | 0.943 | 0.506 |
| 456 | 78.4% | 77.6% | 0.8pp | 0.945 | 0.499 |

Artifacts: `experiments/imdb/baseline/seed{42,123,456}.json`

## PubMedQA

Qwen2.5-0.5B LoRA, ~500 train / ~500 test, response-only mask, seq_len=128, 5 epochs, seed 42.

| Setting | ε | MIA AUC | Mem. Loss | NM Loss |
|---|---|---|---|---|
| non_dp | ∞ | **0.861** | 0.045 | 0.497 |
| dp_mid | 8.80 | 0.538 | 0.618 | 0.736 |
| dp_strong | 0.47 | 0.538 | 0.566 | 0.680 |

The alignment fix made the attack measurably more effective on real text (prior codebase: AUC 0.75–0.80; post-fix: 0.861). DP defeats a stronger attack than the old code was capable of mounting.

Artifacts: `experiments/pubmedqa/baseline/results.json`

## Summary

| Benchmark | Task | non_dp MIA AUC | dp_mid MIA AUC | Utility Gap |
|---|---|---|---|---|
| Canary | Memorization | 0.917 | 0.507 | N/A (intended) |
| SST-2 (5 seeds) | Classification | 0.661 | 0.507 | 5.0 ± 2.8pp |
| IMDB (3 seeds) | Classification | 0.926 | 0.500 | 0.6 ± 2.4pp |
| PubMedQA | QA (no learning) | 0.861 | 0.538 | N/A (no utility) |

DP consistently reduces MIA AUC to near-random (0.50) across all benchmarks. Classification utility costs are small (0.6–5.0pp) and conservative — `C=1.0` clips 88% of gradients, so raising the clip would narrow the gap further.
