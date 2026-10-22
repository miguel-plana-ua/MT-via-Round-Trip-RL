"""CPU sanity tests for the GRPO loss (run: python tests/test_grpo_loss.py).

Uses a tiny randomly initialised M2M100 (the NLLB architecture), so no download is needed.
Checks that
  * the logsumexp-based log-prob equals log_softmax + gather,
  * the chunked-backward mode gives the same loss/logs/gradients as the single-graph mode.
"""
import os
import sys

import torch
from transformers import M2M100Config, M2M100ForConditionalGeneration

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import _gather_log_probs_from_logits_logits, grpo_compute_loss_and_logs  # noqa: E402

PAD, EOS, LANG = 1, 2, 50


class FakeTokenizer:
    pad_token_id = PAD
    eos_token_id = EOS

    def batch_decode(self, ids, skip_special_tokens=True):
        return [" ".join(str(t) for t in row if t not in (0, PAD, EOS, LANG)) or "x" for row in ids.tolist()]


def make_model(seed=0):
    torch.manual_seed(seed)
    cfg = M2M100Config(
        vocab_size=100, d_model=32, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=64, decoder_ffn_dim=64,
        pad_token_id=PAD, eos_token_id=EOS, bos_token_id=0, decoder_start_token_id=EOS,
        dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, max_position_embeddings=64,
    )
    return M2M100ForConditionalGeneration(cfg).eval()


def make_batch(batch=4, cands=3, length=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    enc = {
        "input_ids": torch.randint(3, 100, (batch, 7), generator=g),
        "attention_mask": torch.ones(batch, 7, dtype=torch.long),
    }
    gen = torch.randint(3, 49, (batch, cands, length), generator=g)
    gen[:, :, 0] = EOS
    gen[:, :, 1] = LANG
    for b in range(batch):  # varied lengths: EOS then padding
        for c in range(cands):
            cut = 4 + (b * cands + c) % (length - 4)
            gen[b, c, cut] = EOS
            gen[b, c, cut + 1:] = PAD
    refs = [f"{3 + i} {7 + i} {9}" for i in range(batch)]
    return enc, gen, refs


def run(model, ref, enc, gen, refs, **kw):
    model.zero_grad()
    return grpo_compute_loss_and_logs(
        model, ref, FakeTokenizer(), enc, gen, refs,
        end_of_sentence_token_id=EOS, beta=0.04, clip_param=0.2, tgt_lang_id=LANG, **kw
    )


def main():
    # 1. logsumexp == log_softmax + gather
    logits = torch.randn(3, 5, 100)
    tgt = torch.randint(0, 100, (3, 5))
    old = torch.gather(logits.log_softmax(-1), -1, tgt.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(old, _gather_log_probs_from_logits_logits(logits, tgt), atol=1e-5)
    print("logprob gather OK")

    model, ref = make_model(0), make_model(0)
    # make ref differ from the policy so the KL term is non-trivial
    with torch.no_grad():
        for p in ref.parameters():
            p.add_(0.01 * torch.randn_like(p))
    enc, gen, refs = make_batch()

    loss_a, logs_a = run(model, ref, enc, gen, refs)
    loss_a.backward()
    grads_a = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    for chunk, scale in ((None, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (1, 0.5)):
        loss_b, logs_b = run(model, ref, enc, gen, refs, backward_scale=scale, chunk_groups=chunk)
        grads_b = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        assert torch.allclose(loss_a.detach(), loss_b, atol=1e-6), (chunk, loss_a, loss_b)
        for k in logs_a:
            assert torch.allclose(logs_a[k], logs_b[k], atol=1e-6), (k, chunk)
        for ga, gb in zip(grads_a, grads_b):
            assert torch.allclose(ga * scale, gb, atol=1e-6), chunk
        print(f"chunk_groups={chunk} backward_scale={scale}: loss/logs/grads match single-graph mode")
    print("kl =", float(logs_a["kl"]), "| loss =", float(loss_a))

    # reference refresh as done in main.py: in-place copy of the policy weights (tied embeddings included)
    ref.load_state_dict(model.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(ref.parameters(), model.parameters()))
    _, logs_c = run(model, ref, enc, gen, refs)
    assert float(logs_c["kl"]) < 1e-9, "KL must be 0 right after the reference refresh"
    print("reference refresh via load_state_dict OK (KL = 0 afterwards)")
    assert float(logs_a["kl"]) > 0


if __name__ == "__main__":
    main()
