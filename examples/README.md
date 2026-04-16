# Examples

## Canary frontier (privacy stress test)

Synthetic corpus with unique identifiers. The strongest MIA signal.

```bash
python3 examples/canary_frontier.py
```

**Runtime:** ~4 min on M1 Pro 16 GB.

## SST-2 (utility benchmark)

Short-input binary sentiment. Label-only loss masking, microbatched DP.

```bash
python3 examples/sst2_dp.py --seed 42 --epochs 2 --logical-batch-size 16 --microbatch-size 4 --n-train 2000 --n-test 800
```

## IMDB (long-input utility)

Longer reviews, label-only masking, microbatched DP.

```bash
python3 examples/imdb_dp.py --seed 42 --epochs 5
```

## PubMedQA (real-text privacy validation)

Medical QA. Validates DP prevents leakage even when the model fails to generalize.

```bash
python3 examples/pubmedqa_dp.py
```

## Requirements

All scripts require the `lora` extras:

```bash
pip install -e ".[lora]"    # mlx-lm, numpy
```

PubMedQA, SST-2, and IMDB download data from HuggingFace on first run.
