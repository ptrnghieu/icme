"""Regression test for the Kaggle crash of 2026-11-20 (all OGM-GE runs FAILED rc=1 after ~0.2 h).

Cause: under fp16 AMP the GradScaler grows the loss scale every 2000 steps; the next step overflows, some
gradients become inf, OGM-GE's noise std = grad.std() becomes nan and torch's normal_(0, nan) raises.
This test (1) reproduces the exception with the original OGM-GE formula and (2) checks that
train.ogm_ge_modulate skips the step instead, leaving the gradients untouched so the GradScaler can skip it.
(3) It also runs a short CPU training loop with a GradScaler forced to overflow, if this torch build
supports fp16 GradScaler on CPU.
Run: python test_ogm_overflow.py
"""
import torch
import torch.nn as nn

from models import AVNet
from train import ogm_ge_modulate


def make_grads():
    torch.manual_seed(0)
    m = AVNet(6, "av")
    spec, imgs, y = torch.randn(4, 1, 257, 188), torch.randn(4, 3, 1, 64, 64), torch.tensor([0, 1, 2, 3])
    a, v, out = m(spec, imgs)
    nn.CrossEntropyLoss()(out, y).backward()
    out_a, out_v = m.decompose(a.detach(), v.detach())
    return m, out_a, out_v, y


def main():
    # (1) original formula crashes on an inf gradient
    m, out_a, out_v, y = make_grads()
    p = m.audio_net[0].weight
    p.grad[0, 0, 0, 0] = float("inf")
    try:
        p.grad = p.grad * 1.0 + torch.zeros_like(p.grad).normal_(0, p.grad.std().item() + 1e-8)
        raise SystemExit("expected the original OGM-GE noise step to raise on inf gradients")
    except RuntimeError as e:
        print("(1) original OGM-GE noise on inf grad raises:", e)

    # (2) guarded version skips and leaves gradients untouched
    m, out_a, out_v, y = make_grads()
    m.visual_net[0].weight.grad[0, 0, 0, 0] = float("inf")
    before = [q.grad.clone() for q in m.parameters()]
    ca, cv, ratio, applied = ogm_ge_modulate(m, out_a, out_v, y, 0.5)
    assert applied is False
    same = all(torch.equal(b, q.grad) for b, q in zip(before, m.parameters()))
    assert same, "gradients must be left untouched when the step is skipped"
    print("(2) guarded ogm_ge_modulate on inf grad: applied =", applied, "| gradients untouched =", same)

    # (3) short loop with an overflowing GradScaler (CPU fp16), if supported by this torch build
    try:
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 40, growth_interval=2)
        m = AVNet(6, "av")
        opt = torch.optim.SGD(m.parameters(), lr=1e-3)
        skipped, applied_n = 0, 0
        for step in range(4):
            spec, imgs = torch.randn(4, 1, 257, 188), torch.randn(4, 3, 1, 64, 64)
            y = torch.randint(0, 6, (4,))
            opt.zero_grad()
            with torch.autocast("cpu", dtype=torch.float16):
                a, v, out = m(spec, imgs)
                loss = nn.CrossEntropyLoss()(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            oa, ov = m.decompose(a.detach().float(), v.detach().float())
            *_, ok = ogm_ge_modulate(m, oa, ov, y, 0.5)
            applied_n += ok
            skipped += (not ok)
            scaler.step(opt)
            scaler.update()
        print(f"(3) CPU fp16 loop with forced overflow: 4 steps, modulation applied {applied_n}, skipped {skipped}, "
              f"final scale {scaler.get_scale():.3g} -> no crash")
    except (RuntimeError, TypeError, ValueError) as e:
        print("(3) skipped: CPU fp16 GradScaler not supported by this torch build:", e)


if __name__ == "__main__":
    main()
