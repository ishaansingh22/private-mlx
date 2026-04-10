"""IMDB sentiment DP-LoRA: utility + privacy on real classification.

Trains Qwen2.5-0.5B LoRA on IMDB binary sentiment under three privacy
settings, evaluates forced-choice accuracy, and runs loss-threshold MIA.

Usage:  python3 examples/imdb_dp.py [--seed 42] [--epochs 5]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import Adam

from mlx_private import DPOptimizer, make_private_loss
from mlx_private._patch import ensure_attention_backend_for_per_sample_grads

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEQ_LEN = 320
BATCH_SIZE = 2
EPOCHS = 5
LR = 2e-4
TARGET_DELTA = 1e-5
SEED = 42
N_TRAIN = 1500
N_TEST = 1500

LORA_RANK = 16
LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

SETTINGS = {
    "non_dp": {"noise_multiplier": 0.0, "l2_norm_clip": 1e10},
    "dp_mid": {"noise_multiplier": 0.5, "l2_norm_clip": 1.0},
    "dp_strong": {"noise_multiplier": 1.5, "l2_norm_clip": 1.0},
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
PROMPT = "Is the sentiment of this review positive? Answer Yes or No."


# ---- data ----------------------------------------------------------------

def load_imdb_reviews():
    cache = os.path.join(CACHE_DIR, "imdb_raw.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", split="train")
    items = []
    for row in ds:
        text = html.unescape(row["text"]).replace("<br />", "\n").replace("<br/>", "\n")
        items.append({"text": text, "label": int(row["label"])})

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(items, f)
    print(f"  cached {len(items)} reviews", flush=True)
    return items


def filter_by_length(items, tokenizer, seq_len):
    """Keep reviews that fit in seq_len after chat formatting (no truncation)."""
    kept = []
    for item in items:
        msgs = [
            {"role": "user", "content": f"{PROMPT}\n\n{item['text']}"},
            {"role": "assistant", "content": "Yes"},
        ]
        full = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        if len(tokenizer.encode(full)) <= seq_len + 1:
            kept.append(item)
    return kept


def balanced_split(items, n_train, n_test, seed):
    rng = np.random.RandomState(seed)
    pos = [x for x in items if x["label"] == 1]
    neg = [x for x in items if x["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_per_class = (n_train + n_test) // 2
    balanced = pos[:n_per_class] + neg[:n_per_class]
    rng.shuffle(balanced)
    return balanced[:n_train], balanced[n_train : n_train + n_test]


def format_and_tokenize(items, tokenizer, seq_len):
    """Chat-format + tokenize with full-sequence loss mask (all non-padding).

    Returns (x, y) where y[:, 0, :] = targets, y[:, 1, :] = loss mask.
    """
    all_x, all_y = [], []
    for item in items:
        answer = "Yes" if item["label"] == 1 else "No"
        msgs = [
            {"role": "user", "content": f"{PROMPT}\n\n{item['text']}"},
            {"role": "assistant", "content": answer},
        ]
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        ids = tokenizer.encode(full_text)

        orig_len = len(ids)
        if orig_len < seq_len + 1:
            ids = ids + [0] * (seq_len + 1 - orig_len)
        ids = ids[: seq_len + 1]

        targets = ids[1 : seq_len + 1]
        mask = [0] * seq_len
        for j in range(min(orig_len - 1, seq_len)):
            mask[j] = 1

        all_x.append(ids[:seq_len])
        all_y.append([targets, mask])

    return mx.array(all_x), mx.array(all_y)


# ---- model ----------------------------------------------------------------

def build_model(patch_attention=False):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = load(MODEL_NAME)
    model.freeze()
    linear_to_lora_layers(
        model, num_layers=4,
        config={"rank": LORA_RANK, "scale": 2.0, "dropout": 0.0,
                "keys": LORA_KEYS},
    )
    mx.eval(model.parameters())
    if patch_attention:
        ensure_attention_backend_for_per_sample_grads(
            model, mode="auto", warn=False,
        )
    return model, tokenizer


def _masked_loss(per_tok, mask_shifted):
    numer = (per_tok * mask_shifted).sum()
    denom = mx.maximum(mask_shifted.sum(), mx.array(1.0))
    return numer / denom


def per_sample_loss(model, x, y):
    targets = y[0]
    mask = y[1].astype(mx.float32)
    logits = model(x[None, :])
    per_tok = nn.losses.cross_entropy(
        logits[0, :-1, :], targets[1:], reduction="none",
    )
    return _masked_loss(per_tok, mask[1:])


# ---- eval ----------------------------------------------------------------

def evaluate_accuracy(model, tokenizer, items):
    """Forced-choice accuracy: compare logits for 'yes' vs 'no'."""
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"    token IDs: Yes={yes_id}, No={no_id}", flush=True)

    correct = 0
    pred_yes = 0
    for item in items:
        msgs = [{"role": "user", "content": f"{PROMPT}\n\n{item['text']}"}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        ids = mx.array(tokenizer.encode(prompt))[None, :]
        logits = model(ids)
        last = logits[0, -1, :]
        mx.eval(last)

        pred_pos = last[yes_id].item() > last[no_id].item()
        if pred_pos:
            pred_yes += 1
        if pred_pos == (item["label"] == 1):
            correct += 1

    print(f"    predictions: {pred_yes} yes / {len(items) - pred_yes} no", flush=True)
    return correct / len(items)


# ---- training ------------------------------------------------------------

def train_setting(setting_name, train_x, train_y, *, seed=SEED, epochs=EPOCHS):
    cfg = SETTINGS[setting_name]
    sigma = cfg["noise_multiplier"]
    clip = cfg["l2_norm_clip"]
    is_dp = sigma > 0
    N = int(train_x.shape[0])

    mx.random.seed(seed)
    np.random.seed(seed)
    model, tokenizer = build_model(patch_attention=is_dp)

    if is_dp:
        ps_fn = make_private_loss(
            model, per_sample_loss, configure_attention_backend=False,
        )
        optimizer = DPOptimizer(
            Adam(learning_rate=LR), l2_norm_clip=clip,
            noise_multiplier=sigma, target_delta=TARGET_DELTA,
            num_samples=N, compile=False,
        )
    else:
        def batch_loss(model, x, y):
            targets = y[:, 0, :]
            mask = y[:, 1, :].astype(mx.float32)
            logits = model(x)
            per_tok = nn.losses.cross_entropy(
                logits[:, :-1, :], targets[:, 1:], reduction="none",
            )
            return _masked_loss(per_tok, mask[:, 1:])

        loss_and_grad = nn.value_and_grad(model, batch_loss)
        optimizer = Adam(learning_rate=LR)

    steps_per_epoch = max(1, N // BATCH_SIZE)

    t0 = time.perf_counter()
    final_loss = float("nan")
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        epoch_losses = []
        for step in range(steps_per_epoch):
            idx = perm[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            if len(idx) < BATCH_SIZE:
                continue
            xb = train_x[mx.array(idx)]
            yb = train_y[mx.array(idx)]

            if is_dp:
                grads = ps_fn(xb, yb)
                mx.eval(grads)
                optimizer.step(model, grads)
                mx.eval(model.parameters())
                targets = yb[:, 0, :]
                mask = yb[:, 1, :].astype(mx.float32)
                logits = model(xb)
                per_tok = nn.losses.cross_entropy(
                    logits[:, :-1, :], targets[:, 1:], reduction="none",
                )
                loss = _masked_loss(per_tok, mask[:, 1:])
                mx.eval(loss)
            else:
                loss, grads = loss_and_grad(model, xb, yb)
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)

            epoch_losses.append(float(loss.item()))

            if step % 100 == 0:
                avg = float(np.mean(epoch_losses[-100:]))
                elapsed = time.perf_counter() - t0
                print(
                    f"    e{epoch+1} step {step}/{steps_per_epoch}"
                    f"  loss={avg:.4f}  [{elapsed:.0f}s]",
                    flush=True,
                )

        final_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"    epoch {epoch + 1}/{epochs}  loss={final_loss:.4f}", flush=True)

    elapsed = time.perf_counter() - t0
    epsilon = float(optimizer.epsilon) if is_dp else None

    return model, tokenizer, {
        "setting": setting_name,
        "epsilon": epsilon,
        "final_loss": round(final_loss, 4),
        "elapsed_s": round(elapsed, 1),
    }


# ---- MIA -----------------------------------------------------------------

def score_losses(model, all_x, all_y):
    losses = []
    for i in range(all_x.shape[0]):
        targets = all_y[i, 0, :]
        mask = all_y[i, 1, :].astype(mx.float32)
        logits = model(all_x[i : i + 1])
        per_tok = nn.losses.cross_entropy(
            logits[0, :-1, :], targets[1:], reduction="none",
        )
        loss = _masked_loss(per_tok, mask[1:])
        mx.eval(loss)
        losses.append(float(loss.item()))
    return losses


def compute_mia_metrics(member_losses, nonmember_losses):
    labels = np.array([1] * len(member_losses) + [0] * len(nonmember_losses))
    scores = np.array(member_losses + nonmember_losses)
    neg_scores = -scores

    sorted_idx = np.argsort(neg_scores)[::-1]
    sorted_labels = labels[sorted_idx]
    n_pos, n_neg = int(labels.sum()), len(labels) - int(labels.sum())

    tpr_list, fpr_list = [0.0], [0.0]
    tp = fp = 0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / max(n_pos, 1))
        fpr_list.append(fp / max(n_neg, 1))

    auc = sum(
        (fpr_list[i + 1] - fpr_list[i]) * (tpr_list[i + 1] + tpr_list[i]) / 2
        for i in range(len(fpr_list) - 1)
    )

    tpr_at_fpr_001 = 0.0
    for i in range(len(fpr_list)):
        if fpr_list[i] <= 0.01:
            tpr_at_fpr_001 = tpr_list[i]

    return {
        "roc_auc": round(float(auc), 4),
        "tpr@fpr=0.01": round(float(tpr_at_fpr_001), 4),
        "member_loss_mean": round(float(np.mean(member_losses)), 4),
        "nonmember_loss_mean": round(float(np.mean(nonmember_losses)), 4),
    }


# ---- main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--settings", nargs="+", default=["non_dp", "dp_mid", "dp_strong"])
    args = parser.parse_args()
    epochs, seed = args.epochs, args.seed

    print("IMDB Sentiment DP-LoRA", flush=True)
    print("=" * 60, flush=True)

    raw = load_imdb_reviews()
    print(f"Loaded {len(raw)} IMDB reviews", flush=True)

    model, tokenizer = build_model()

    filtered = filter_by_length(raw, tokenizer, SEQ_LEN)
    print(f"After length filter: {len(filtered)}/{len(raw)} fit SEQ_LEN={SEQ_LEN}", flush=True)

    train_items, test_items = balanced_split(filtered, N_TRAIN, N_TEST, seed)
    train_pos = sum(1 for x in train_items if x["label"] == 1)
    test_pos = sum(1 for x in test_items if x["label"] == 1)
    print(f"Train: {len(train_items)} ({train_pos} pos / {len(train_items) - train_pos} neg)", flush=True)
    print(f"Test:  {len(test_items)} ({test_pos} pos / {len(test_items) - test_pos} neg)", flush=True)

    train_x, train_y = format_and_tokenize(train_items, tokenizer, SEQ_LEN)
    test_x, test_y = format_and_tokenize(test_items, tokenizer, SEQ_LEN)
    mx.eval(train_x, train_y, test_x, test_y)
    print(f"Tokenized: train={train_x.shape}, test={test_x.shape}", flush=True)

    print("\nZero-shot baseline...", flush=True)
    baseline_acc = evaluate_accuracy(model, tokenizer, test_items[:200])
    print(f"  Zero-shot accuracy (200 examples): {baseline_acc:.1%}", flush=True)

    del model
    mx.clear_cache()

    results = []
    for setting in args.settings:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  {setting}", flush=True)
        print(f"{'=' * 60}", flush=True)

        model, tok, meta = train_setting(
            setting, train_x, train_y, seed=seed, epochs=epochs,
        )

        print("  Evaluating accuracy...", flush=True)
        acc = evaluate_accuracy(model, tok, test_items)
        meta["accuracy"] = round(acc, 4)
        print(f"  Accuracy: {acc:.1%}", flush=True)

        print("  Scoring MIA losses...", flush=True)
        train_losses = score_losses(model, train_x, train_y)
        test_losses = score_losses(model, test_x, test_y)
        mia = compute_mia_metrics(train_losses, test_losses)
        meta.update(mia)

        eps_str = f"ε={meta['epsilon']:.2f}" if meta["epsilon"] else "ε=∞"
        print(
            f"  {eps_str}  acc={acc:.1%}  MIA_AUC={mia['roc_auc']}  "
            f"TPR@1%={mia['tpr@fpr=0.01']}",
            flush=True,
        )

        results.append(meta)
        del model
        mx.clear_cache()

    print(f"\n{'=' * 60}", flush=True)
    print("  IMDB Sentiment Results", flush=True)
    print(f"{'=' * 60}", flush=True)
    hdr = (
        f"{'Setting':<12} {'ε':>8} {'Accuracy':>10} {'MIA AUC':>9}"
        f" {'TPR@1%':>8} {'Mem.Loss':>9} {'NM.Loss':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        eps = f"{r['epsilon']:.2f}" if r["epsilon"] else "∞"
        print(
            f"{r['setting']:<12} {eps:>8} {r['accuracy']:>9.1%}"
            f" {r['roc_auc']:>9.4f} {r['tpr@fpr=0.01']:>8.4f}"
            f" {r['member_loss_mean']:>9.4f} {r['nonmember_loss_mean']:>9.4f}"
        )

    os.makedirs(CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CACHE_DIR, f"imdb_results_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
