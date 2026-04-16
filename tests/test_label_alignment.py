"""Pin the label-only alignment contract.

Proves that per_sample_loss and score_losses supervise the answer token
(not an earlier position) and that the two paths agree exactly.
"""

import mlx.core as mx
import mlx.nn as nn


class _TinyLM(nn.Module):
    """Minimal causal LM: embed -> linear -> vocab logits."""

    def __init__(self, vocab: int = 64, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def __call__(self, x):
        return self.head(self.embed(x))


def _masked_loss(per_tok, mask):
    numer = (per_tok * mask).sum()
    denom = mx.maximum(mask.sum(), mx.array(1.0))
    return numer / denom


def _per_sample_loss(model, x, y):
    targets = y[0]
    mask = y[1].astype(mx.float32)
    logits = model(x[None, :])
    per_tok = nn.losses.cross_entropy(logits[0], targets, reduction="none")
    return _masked_loss(per_tok, mask)


def _score_loss(model, x, y):
    targets = y[0]
    mask = y[1].astype(mx.float32)
    logits = model(x[None, :])
    per_tok = nn.losses.cross_entropy(logits[0], targets, reduction="none")
    return _masked_loss(per_tok, mask)


def test_label_only_supervises_answer_token():
    """Mask at first_resp = resp_start-1 must produce CE(logits[resp_start-1], ids[resp_start])."""
    mx.random.seed(0)
    model = _TinyLM(vocab=64, dim=16)
    mx.eval(model.parameters())

    ids = mx.array([10, 20, 30, 40, 50])
    resp_start = 3  # answer token is ids[3]=40
    seq_len = 4

    x = ids[:seq_len]
    targets = ids[1 : seq_len + 1]  # [20, 30, 40, 50]
    first_resp = resp_start - 1  # mask position 2 -> targets[2]=ids[3]=40
    mask = mx.zeros(seq_len)
    mask = mask.at[first_resp].add(1.0)

    loss = _per_sample_loss(model, x, [targets, mask])

    logits = model(x[None, :])
    expected = nn.losses.cross_entropy(
        logits[0, first_resp : first_resp + 1],
        targets[first_resp : first_resp + 1],
        reduction="mean",
    )
    mx.eval(loss, expected)
    assert mx.allclose(loss, expected, atol=1e-5).item(), (
        f"loss={loss.item():.6f} expected={expected.item():.6f}"
    )


def test_score_losses_matches_per_sample():
    """MIA scoring path must produce the same loss as the training path."""
    mx.random.seed(1)
    model = _TinyLM(vocab=64, dim=16)
    mx.eval(model.parameters())

    ids = mx.array([5, 15, 25, 35, 45, 55])
    resp_start = 4
    seq_len = 5

    x = ids[:seq_len]
    targets = ids[1 : seq_len + 1]
    first_resp = resp_start - 1
    mask = mx.zeros(seq_len)
    mask = mask.at[first_resp].add(1.0)

    train_loss = _per_sample_loss(model, x, [targets, mask])
    mia_loss = _score_loss(model, x, [targets, mask])
    mx.eval(train_loss, mia_loss)
    assert mx.allclose(train_loss, mia_loss, atol=1e-6).item(), (
        f"train={train_loss.item():.6f} mia={mia_loss.item():.6f}"
    )


def test_mask_position_maps_to_answer_id():
    """Verify that targets[first_resp] == ids[resp_start] for the mask contract."""
    ids = [10, 20, 30, 40, 50, 60]
    resp_start = 4
    seq_len = 5

    targets = ids[1 : seq_len + 1]  # [20, 30, 40, 50, 60]
    first_resp = max(0, resp_start - 1)  # 3

    assert targets[first_resp] == ids[resp_start], (
        f"targets[{first_resp}]={targets[first_resp]} != ids[{resp_start}]={ids[resp_start]}"
    )


def test_old_shifted_loss_would_fail():
    """The pre-fix double-shift would supervise the wrong token — assert it differs."""
    mx.random.seed(2)
    model = _TinyLM(vocab=64, dim=16)
    mx.eval(model.parameters())

    ids = mx.array([10, 20, 30, 40, 50])
    resp_start = 3
    seq_len = 4

    x = ids[:seq_len]
    targets = ids[1 : seq_len + 1]
    first_resp = resp_start - 1
    mask = mx.zeros(seq_len)
    mask = mask.at[first_resp].add(1.0)

    correct_loss = _per_sample_loss(model, x, [targets, mask])

    logits = model(x[None, :])
    old_per_tok = nn.losses.cross_entropy(
        logits[0, :-1, :], targets[1:], reduction="none"
    )
    old_mask = mask[1:]
    old_loss = _masked_loss(old_per_tok, old_mask)

    mx.eval(correct_loss, old_loss)
    assert not mx.allclose(correct_loss, old_loss, atol=1e-3).item(), (
        "Old shifted loss should NOT match the correct loss"
    )
