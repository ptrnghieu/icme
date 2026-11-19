"""Check that train.ogm_ge_modulate reproduces the original OGM-GE update.

The reference block below is copied from GeWu-Lab/OGM-GE_CVPR2022 main.py (train_epoch, OGM_GE branch);
only the parameter-name test is adapted ('module.audio_net...' under DataParallel -> 'audio_net...').
Run: python test_ogm_equivalence.py
"""
import copy

import torch
import torch.nn as nn

from models import AVNet
from train import ogm_ge_modulate


def reference(model, out_a, out_v, label, alpha):
    softmax, relu, tanh = nn.Softmax(dim=1), nn.ReLU(inplace=True), nn.Tanh()
    score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
    score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])
    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v
    if ratio_v > 1:
        coeff_v = 1 - tanh(alpha * relu(ratio_v))
        coeff_a = 1
    else:
        coeff_a = 1 - tanh(alpha * relu(ratio_a))
        coeff_v = 1
    for name, parms in model.named_parameters():
        layer = ("module." + name).split(".")[1]
        if "audio" in layer and len(parms.grad.size()) == 4:
            parms.grad = parms.grad * coeff_a + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
        if "visual" in layer and len(parms.grad.size()) == 4:
            parms.grad = parms.grad * coeff_v + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
    return float(coeff_a), float(coeff_v)


def main():
    torch.manual_seed(0)
    m1 = AVNet(6, "av")
    spec, imgs, y = torch.randn(4, 1, 257, 188), torch.randn(4, 3, 1, 64, 64), torch.tensor([0, 1, 2, 3])
    a, v, out = m1(spec, imgs)
    nn.CrossEntropyLoss()(out, y).backward()
    m2 = copy.deepcopy(m1)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        p2.grad = p1.grad.clone()
    for alpha in (0.1, 0.8):
        mA, mB = copy.deepcopy(m1), copy.deepcopy(m2)
        for p1, p2, q1, q2 in zip(m1.parameters(), m2.parameters(), mA.parameters(), mB.parameters()):
            q1.grad, q2.grad = p1.grad.clone(), p2.grad.clone()
        out_a, out_v = m1.decompose(a.detach(), v.detach())
        torch.manual_seed(123)
        ca, cv, _ = ogm_ge_modulate(mA, out_a, out_v, y, alpha)
        torch.manual_seed(123)
        ra, rv = reference(mB, out_a, out_v, y, alpha)
        assert abs(ca - ra) < 1e-6 and abs(cv - rv) < 1e-6, (ca, ra, cv, rv)
        maxdiff = max((q1.grad - q2.grad).abs().max().item() for q1, q2 in zip(mA.parameters(), mB.parameters()))
        assert maxdiff < 1e-6, maxdiff
        print(f"alpha={alpha}: coeff_a={ca:.4f} coeff_v={cv:.4f} | max grad diff vs reference = {maxdiff:.2e}  OK")


if __name__ == "__main__":
    main()
