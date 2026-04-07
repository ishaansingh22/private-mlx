"""End-to-end MIA experiment: prepare data, train 3 adapters, score, chart.

Usage:
    python run_experiment.py --corpus corpus.jsonl [--output-dir mia_run]

If no corpus is provided, generates a synthetic canary corpus for testing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def generate_canary_corpus(output_path: str, n: int = 400, seed: int = 42):
    """Generate a synthetic corpus with unique canary phrases."""
    rng = np.random.RandomState(seed)
    templates = [
        "The secret code for entry {i} is {code}. Remember it carefully.",
        "Patient {i} was diagnosed with condition {code} on January 15th.",
        "Account number {i} has balance {code} as of last quarter.",
        "Employee {i} rated performance score {code} in their review.",
        "Document {i} contains classified reference {code} on page three.",
    ]
    corpus = []
    for i in range(n):
        t = templates[i % len(templates)]
        code = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"), size=8))
        corpus.append({"text": t.format(i=i, code=code)})

    with open(output_path, "w") as f:
        for item in corpus:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {n} canary examples -> {output_path}")
    return n


def create_split(n_total: int, member_fraction: float, seed: int, output_path: str):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)
    n_members = int(n_total * member_fraction)
    split = {
        "member_ids": sorted(indices[:n_members].tolist()),
        "nonmember_ids": sorted(indices[n_members:].tolist()),
        "seed": seed,
    }
    with open(output_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Split: {n_members} members, {n_total - n_members} nonmembers -> {output_path}")
    return split


def run_cmd(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"FAILED: {desc} (exit {result.returncode})")
        sys.exit(1)


def collect_results(result_dir: str, settings: list[str]) -> list[dict]:
    results = []
    for s in settings:
        path = os.path.join(result_dir, f"{s}.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def print_summary(results: list[dict]):
    print(f"\n{'='*60}")
    print("  MIA Frontier Summary")
    print(f"{'='*60}")
    print(f"{'Setting':<12} {'ε':>8} {'ROC-AUC':>10} {'Bal.Acc':>10} {'Mem.Loss':>10} {'Non.Loss':>10}")
    print("-" * 62)
    for r in results:
        eps = r.get("epsilon")
        eps_str = f"{eps:.2f}" if eps is not None else "∞"
        print(
            f"{r['setting']:<12} {eps_str:>8} {r['roc_auc']:>10.4f} "
            f"{r['balanced_accuracy']:>10.4f} {r['member_loss_mean']:>10.4f} "
            f"{r['nonmember_loss_mean']:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", help="Path to corpus JSONL. If omitted, generates canaries.")
    parser.add_argument("--output-dir", default="mia_run")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only score")
    args = parser.parse_args()

    base = args.output_dir
    os.makedirs(base, exist_ok=True)

    corpus_path = args.corpus
    if corpus_path is None:
        corpus_path = os.path.join(base, "canary_corpus.jsonl")
        n_total = generate_canary_corpus(corpus_path, n=400, seed=args.seed)
    else:
        with open(corpus_path) as f:
            n_total = sum(1 for _ in f)

    split_path = os.path.join(base, "split.json")
    create_split(n_total, member_fraction=0.5, seed=args.seed, output_path=split_path)

    settings = ["non_dp", "dp_mid", "dp_strong"]
    py = sys.executable

    if not args.skip_train:
        for s in settings:
            run_cmd(
                [py, "train.py",
                 "--setting", s,
                 "--output", os.path.join(base, "runs", s),
                 "--split", split_path,
                 "--corpus", corpus_path,
                 "--epochs", str(args.epochs),
                 "--batch-size", str(args.batch_size),
                 "--seed", str(args.seed)],
                f"Train: {s}",
            )

    result_dir = os.path.join(base, "results")
    os.makedirs(result_dir, exist_ok=True)

    for s in settings:
        run_dir = os.path.join(base, "runs", s)
        if not os.path.exists(os.path.join(run_dir, "adapter.npz")):
            print(f"Skipping score for {s}: no adapter found")
            continue
        run_cmd(
            [py, "attack.py",
             "--run-dir", run_dir,
             "--split", split_path,
             "--corpus", corpus_path,
             "--output", os.path.join(result_dir, f"{s}.json")],
            f"Score: {s}",
        )

    results = collect_results(result_dir, settings)
    if results:
        print_summary(results)

        summary_path = os.path.join(base, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
