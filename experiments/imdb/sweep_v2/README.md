# IMDB Sweep v2 (Decision-Gated)

This directory stores auditable artifacts for the post-alignment IMDB sweep.

- Objective: label-only assistant-token loss.
- DP path: `step_microbatched()` with logical batch > microbatch.
- Grid:
  - `seq_len`: 96, 128, 160
  - `logical_batch`: 8, 16
  - `microbatch`: 2
  - `seeds`: 42, 123, 456
- Run budget: `epochs=2`, `n_train=256`, `n_test=256`.

## Result

Best `dp_mid` candidate:

- `seq_len=128`, `logical_batch=8`, `microbatch=2`
- accuracy `0.6028 ± 0.1093`
- MIA AUC `0.5055`
- epsilon `13.25`

Baseline (`non_dp`) on same config:

- accuracy `0.6836 ± 0.0345`
- MIA AUC `0.5429`

Hard-stop decision:

- `fallback_required`
- gap: `8.08pp` (utility gap acceptable)
- stability: **failed** (`std=10.93pp` > `8pp`)

IMDB remains too unstable for flagship utility claims under this run budget, so the benchmark was switched to a short-input replacement (`examples/sst2_dp.py`).
