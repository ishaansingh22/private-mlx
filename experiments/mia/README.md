# MIA Experiment

Loss-threshold membership inference attack on DP vs non-DP LoRA adapters.

## Quick start

```bash
# Single-seed canary run (fast, ~3 min):
python run_experiment.py --epochs 5 --seed 42

# Multi-seed with full diagnostics (~20 min):
python multi_seed.py --seeds 5

# Real corpus:
python run_experiment.py --corpus my_data.jsonl --epochs 5 --seed 42

# Personal demo (local-only, not committed):
python demo.py --corpus my_writing.jsonl --noise-multiplier 1.0 --epochs 3
```

## Results (5 seeds)

| Setting | ε (measured) | ROC-AUC | Bal. Accuracy | TPR @ 1% FPR |
|---|---|---|---|---|
| No DP | ∞ | 0.790 ± 0.024 | 0.727 ± 0.029 | 0.157 ± 0.075 |
| DP (σ=0.5) | 15.43 | 0.493 ± 0.006 | 0.519 ± 0.005 | 0.034 ± 0.006 |
| DP (σ=1.5) | 1.08 | 0.493 ± 0.015 | 0.523 ± 0.010 | 0.024 ± 0.016 |

ε computed via RDP accountant at δ=1e-5 (Poisson subsampling assumption; conservative).

**Verbatim extraction:** 0/20 canary codes extractable via greedy decoding from the non-DP model, even with tokenization-correct prefixes (training token IDs used directly as prompts). Teacher-forced next-token accuracy reaches 97.4%, confirming the memorization exists but doesn't surface through autoregressive generation. Loss-based MIA is the appropriate detection tool for LoRA memorization.

**Canary ablation:** Unseen canaries (novel codes, same template) cluster with nonmembers (loss 0.857 vs 0.786), not members (0.696). The attack detects per-example memorization, not template recognition.

**Response-only scoring:** Full-sequence AUC (0.793) slightly exceeds response-only AUC (0.768). The attack is well-specified; prompt tokens carry marginal signal.

## Corpus selection criteria

The MIA signal depends on the non-DP model memorizing. A corpus that the base model already knows produces no leakage gap and a flat frontier. Pick a corpus where:

- **200-500 examples.** Small enough to memorize in a few epochs with LoRA.
- **Base model is plausibly weak on this domain.** Check: compute base model perplexity on the corpus vs generic text. If the gap is small, the model already "knows" the content and won't memorize much during fine-tuning.
- **No overlap with common pretraining data.** Avoid Wikipedia, major news, StackOverflow, popular GitHub repos. These are in every LLM's training set.
- **Candidates that tend to work:** niche technical Q&A post-cutoff, stylized fiction from a low-traffic author, non-English minority language text, private/internal documents, synthetic canaries with unique identifiers.

**Sanity check before running the full frontier:** Train one quick non-DP adapter for a few hundred steps. If holdout loss barely drops below the base model's holdout loss, or if MIA ROC-AUC is already < 0.65 with no DP, the corpus is too easy. Switch or fall back to canaries.

## Corpus format

JSONL, one example per line:

```json
{"text": "The secret code for entry 0 is A7B3K9Z2. Remember it carefully."}
{"text": "Patient 1 was diagnosed with condition X8M2P4Q1 on January 15th."}
```

## Claims policy

When writing about results from this experiment, avoid:

- "fastest" or speed comparisons without measurement
- "production-ready"
- "matches Opacus" (different frameworks, different hardware, not apples-to-apples)
- "formal privacy guarantees" without the conservative-ε caveat in the same sentence
- "provably private" (the ε bound is conservative due to sampling mismatch)

Safe phrasings: "DP-SGD with RDP accounting", "conservative ε estimate", "per-sample gradient clipping with calibrated noise", "membership inference attack collapses to chance under DP".
