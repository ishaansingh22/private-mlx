"""Decision-gated IMDB DP sweep for public-result stabilization.

Phase 1:
  - Run dp_mid over (seq_len x logical_batch x microbatch) for one seed.
  - Pick the better microbatch per (seq_len, logical_batch).

Phase 2:
  - Run remaining seeds for the selected (seq_len, logical_batch, best_microbatch)
    configs and compute 3-seed aggregates.

Phase 3:
  - Run non-DP baseline for the best dp_mid config across the same seeds.
  - Apply stop/go criteria for whether IMDB remains viable as the utility story.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IMDB_SCRIPT = REPO_ROOT / "examples" / "imdb_dp.py"


@dataclass(frozen=True)
class RunKey:
    setting: str
    seed: int
    seq_len: int
    logical_batch: int
    microbatch: int


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _result_path(run_dir: Path, key: RunKey) -> Path:
    return run_dir / (
        f"{key.setting}_seed{key.seed}_sl{key.seq_len}_lb{key.logical_batch}"
        f"_mb{key.microbatch}.json"
    )


def _run_config(
    key: RunKey,
    *,
    run_dir: Path,
    epochs: int,
    n_train: int,
    n_test: int,
    mask_mode: str,
    max_review_tokens: int,
    dry_run: bool,
) -> dict:
    out_path = _result_path(run_dir, key)
    if out_path.exists():
        with open(out_path) as f:
            payload = json.load(f)
        run = payload["results"][0]
        run["from_cache"] = True
        return run

    cmd = [
        "python",
        str(IMDB_SCRIPT),
        "--seed",
        str(key.seed),
        "--epochs",
        str(epochs),
        "--seq-len",
        str(key.seq_len),
        "--logical-batch-size",
        str(key.logical_batch),
        "--microbatch-size",
        str(key.microbatch),
        "--max-review-tokens",
        str(max_review_tokens),
        "--mask-mode",
        mask_mode,
        "--n-train",
        str(n_train),
        "--n-test",
        str(n_test),
        "--settings",
        key.setting,
        "--output",
        str(out_path),
    ]

    if dry_run:
        return {
            "setting": key.setting,
            "epsilon": None,
            "accuracy": 0.0,
            "roc_auc": 0.5,
            "tpr@fpr=0.01": 0.0,
            "elapsed_s": 0.0,
            "logical_batch_size": key.logical_batch,
            "microbatch_size": key.microbatch,
            "dry_run": True,
        }

    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    elapsed = time.perf_counter() - t0

    with open(out_path) as f:
        payload = json.load(f)
    run = payload["results"][0]
    run["wall_elapsed_s"] = round(elapsed, 1)
    run["from_cache"] = False
    return run


def _microbatch_screen_score(run: dict) -> float:
    # Higher is better: prioritize utility, then closeness to chance-level MIA.
    return float(run["accuracy"]) - 0.5 * abs(float(run["roc_auc"]) - 0.5)


def _aggregate_runs(runs: list[dict]) -> dict:
    accs = [float(r["accuracy"]) for r in runs]
    aucs = [float(r["roc_auc"]) for r in runs]
    tprs = [float(r["tpr@fpr=0.01"]) for r in runs]
    eps = [float(r["epsilon"]) for r in runs if r.get("epsilon") is not None]
    return {
        "n_runs": len(runs),
        "accuracy_mean": statistics.mean(accs),
        "accuracy_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
        "mia_auc_mean": statistics.mean(aucs),
        "mia_auc_std": statistics.pstdev(aucs) if len(aucs) > 1 else 0.0,
        "tpr1_mean": statistics.mean(tprs),
        "epsilon_mean": statistics.mean(eps) if eps else None,
    }


def _decision(dp_best: dict, non_dp: dict) -> dict:
    gap_pp = (non_dp["accuracy_mean"] - dp_best["accuracy_mean"]) * 100.0
    stable = dp_best["accuracy_std"] * 100.0 <= 8.0
    utility_ok = gap_pp <= 10.0
    privacy_ok = 0.45 <= dp_best["mia_auc_mean"] <= 0.55
    stop = not (stable and utility_ok and privacy_ok)
    return {
        "status": "fallback_required" if stop else "imdb_viable",
        "gap_pp": round(gap_pp, 2),
        "stable": stable,
        "utility_ok": utility_ok,
        "privacy_ok": privacy_ok,
    }


def _write_markdown(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# IMDB DP Sweep Summary")
    lines.append("")
    cfg = summary["config"]
    lines.append(
        f"- seeds: {cfg['seeds']}\n"
        f"- seq_lens: {cfg['seq_lens']}\n"
        f"- logical_batches: {cfg['logical_batches']}\n"
        f"- microbatches: {cfg['microbatches']}\n"
        f"- epochs: {cfg['epochs']}, n_train: {cfg['n_train']}, n_test: {cfg['n_test']}"
    )
    lines.append("")
    lines.append("## DP-Mid Candidates (3-seed)")
    lines.append("")
    lines.append("| seq_len | logical_batch | microbatch | acc_mean | acc_std | mia_auc_mean | epsilon_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in summary["dp_candidates"]:
        agg = c["aggregate"]
        lines.append(
            f"| {c['seq_len']} | {c['logical_batch']} | {c['microbatch']} | "
            f"{agg['accuracy_mean']:.4f} | {agg['accuracy_std']:.4f} | "
            f"{agg['mia_auc_mean']:.4f} | {agg['epsilon_mean']:.2f} |"
        )
    lines.append("")
    lines.append("## Headline Comparison")
    lines.append("")
    best = summary["dp_best"]
    non_dp = summary["non_dp_baseline"]
    d = summary["decision"]
    lines.append(
        f"- dp_mid best: sl={best['seq_len']} lb={best['logical_batch']} mb={best['microbatch']} "
        f"acc={best['aggregate']['accuracy_mean']:.4f}±{best['aggregate']['accuracy_std']:.4f} "
        f"auc={best['aggregate']['mia_auc_mean']:.4f} eps={best['aggregate']['epsilon_mean']:.2f}"
    )
    lines.append(
        f"- non_dp baseline: acc={non_dp['aggregate']['accuracy_mean']:.4f}±"
        f"{non_dp['aggregate']['accuracy_std']:.4f} auc={non_dp['aggregate']['mia_auc_mean']:.4f}"
    )
    lines.append(
        f"- decision: {d['status']} (gap_pp={d['gap_pp']}, stable={d['stable']}, "
        f"utility_ok={d['utility_ok']}, privacy_ok={d['privacy_ok']})"
    )
    lines.append("")
    if d["status"] == "fallback_required":
        lines.append("## Stop-Go Outcome")
        lines.append("")
        lines.append(
            "IMDB remains unstable or too far below non-DP. Stop further IMDB tuning "
            "and move to a short-input, short-label benchmark."
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--seq-lens", default="96,128,160")
    parser.add_argument("--logical-batches", default="8,16")
    parser.add_argument("--microbatches", default="2,4")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=600)
    parser.add_argument("--mask-mode", choices=["assistant", "label"], default="label")
    parser.add_argument("--max-review-tokens", type=int, default=96)
    parser.add_argument("--output-dir", default="examples/.cache/imdb_sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    seq_lens = _parse_int_list(args.seq_lens)
    logical_batches = _parse_int_list(args.logical_batches)
    microbatches = _parse_int_list(args.microbatches)

    run_dir = REPO_ROOT / args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    phase1 = []
    best_micro_per_pair: dict[tuple[int, int], dict] = {}

    # Phase 1: one-seed microbatch screening across full grid.
    seed0 = seeds[0]
    for seq_len in seq_lens:
        for lb in logical_batches:
            candidates = []
            for mb in microbatches:
                if mb > lb:
                    continue
                key = RunKey("dp_mid", seed0, seq_len, lb, mb)
                run = _run_config(
                    key,
                    run_dir=run_dir,
                    epochs=args.epochs,
                    n_train=args.n_train,
                    n_test=args.n_test,
                    mask_mode=args.mask_mode,
                    max_review_tokens=min(args.max_review_tokens, seq_len - 16),
                    dry_run=args.dry_run,
                )
                row = {
                    "seed": seed0,
                    "seq_len": seq_len,
                    "logical_batch": lb,
                    "microbatch": mb,
                    **run,
                }
                phase1.append(row)
                candidates.append(row)
            if not candidates:
                continue
            best = max(candidates, key=_microbatch_screen_score)
            best_micro_per_pair[(seq_len, lb)] = best

    # Phase 2: complete 3-seed runs for selected seq_len/lb with screened microbatch.
    dp_runs_by_cfg: dict[tuple[int, int, int], list[dict]] = {}
    for (seq_len, lb), screened in best_micro_per_pair.items():
        mb = int(screened["microbatch"])
        cfg_key = (seq_len, lb, mb)
        cfg_runs = [screened]
        for seed in seeds[1:]:
            key = RunKey("dp_mid", seed, seq_len, lb, mb)
            run = _run_config(
                key,
                run_dir=run_dir,
                epochs=args.epochs,
                n_train=args.n_train,
                n_test=args.n_test,
                mask_mode=args.mask_mode,
                max_review_tokens=min(args.max_review_tokens, seq_len - 16),
                dry_run=args.dry_run,
            )
            cfg_runs.append(
                {
                    "seed": seed,
                    "seq_len": seq_len,
                    "logical_batch": lb,
                    "microbatch": mb,
                    **run,
                }
            )
        dp_runs_by_cfg[cfg_key] = cfg_runs

    dp_candidates = []
    for (seq_len, lb, mb), runs in sorted(dp_runs_by_cfg.items()):
        dp_candidates.append(
            {
                "seq_len": seq_len,
                "logical_batch": lb,
                "microbatch": mb,
                "runs": runs,
                "aggregate": _aggregate_runs(runs),
            }
        )

    # Pick headline dp_mid config: highest mean accuracy, tie-break by AUC closeness to 0.5.
    dp_best = max(
        dp_candidates,
        key=lambda c: (
            c["aggregate"]["accuracy_mean"],
            -abs(c["aggregate"]["mia_auc_mean"] - 0.5),
        ),
    )

    # Phase 3: non-DP baseline on the same seq_len/lb across all seeds.
    non_dp_runs = []
    for seed in seeds:
        key = RunKey(
            "non_dp",
            seed,
            int(dp_best["seq_len"]),
            int(dp_best["logical_batch"]),
            int(dp_best["microbatch"]),
        )
        run = _run_config(
            key,
            run_dir=run_dir,
            epochs=args.epochs,
            n_train=args.n_train,
            n_test=args.n_test,
            mask_mode=args.mask_mode,
            max_review_tokens=min(args.max_review_tokens, int(dp_best["seq_len"]) - 16),
            dry_run=args.dry_run,
        )
        non_dp_runs.append(
            {
                "seed": seed,
                "seq_len": int(dp_best["seq_len"]),
                "logical_batch": int(dp_best["logical_batch"]),
                "microbatch": int(dp_best["microbatch"]),
                **run,
            }
        )

    non_dp_baseline = {
        "seq_len": int(dp_best["seq_len"]),
        "logical_batch": int(dp_best["logical_batch"]),
        "microbatch": int(dp_best["microbatch"]),
        "runs": non_dp_runs,
        "aggregate": _aggregate_runs(non_dp_runs),
    }

    decision = _decision(dp_best["aggregate"], non_dp_baseline["aggregate"])

    summary = {
        "config": {
            "seeds": seeds,
            "seq_lens": seq_lens,
            "logical_batches": logical_batches,
            "microbatches": microbatches,
            "epochs": args.epochs,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "mask_mode": args.mask_mode,
            "max_review_tokens": args.max_review_tokens,
            "output_dir": str(run_dir),
        },
        "phase1": phase1,
        "dp_candidates": dp_candidates,
        "dp_best": dp_best,
        "non_dp_baseline": non_dp_baseline,
        "decision": decision,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(summary_md, summary)

    print(f"Sweep summary written to {summary_json}")
    print(f"Human-readable summary written to {summary_md}")
    print(f"Decision: {decision['status']} (gap_pp={decision['gap_pp']})")


if __name__ == "__main__":
    main()
