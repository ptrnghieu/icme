"""Audio-visual classifier equivalent to OGM-GE's AVClassifier with concat fusion.

Backbones: ResNet-18 trained from scratch (torchvision definition, no pretrained weights), audio input
with 1 channel. Visual frames are encoded independently and pooled with adaptive_avg_pool3d over
(time, h, w), as in OGM-GE. Fusion: a single linear layer on [a; v] (OGM-GE ConcatFusion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _backbone(in_ch):
    m = torchvision.models.resnet18(weights=None)
    if in_ch != 3:
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)


class AVNet(nn.Module):
    """mode: 'av' (concat fusion), 'audio' (audio only), 'visual' (visual only)."""

    def __init__(self, n_classes=6, mode="av"):
        super().__init__()
        self.mode = mode
        if mode in ("av", "audio"):
            self.audio_net = _backbone(1)
        if mode in ("av", "visual"):
            self.visual_net = _backbone(3)
        self.fc_out = nn.Linear(1024 if mode == "av" else 512, n_classes)

    def encode_audio(self, spec):
        return torch.flatten(F.adaptive_avg_pool2d(self.audio_net(spec), 1), 1)

    def encode_visual(self, imgs):
        b, c, t, h, w = imgs.shape
        x = imgs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        f = self.visual_net(x)
        _, cc, hh, ww = f.shape
        f = f.view(b, t, cc, hh, ww).permute(0, 2, 1, 3, 4)
        return torch.flatten(F.adaptive_avg_pool3d(f, 1), 1)

    def forward(self, spec, imgs):
        a = self.encode_audio(spec) if self.mode in ("av", "audio") else None
        v = self.encode_visual(imgs) if self.mode in ("av", "visual") else None
        if self.mode == "av":
            out = self.fc_out(torch.cat([a, v], 1))
        else:
            out = self.fc_out(a if a is not None else v)
        return a, v, out

    def decompose(self, a, v):
        """OGM-GE decomposition of the concat-fusion logits into audio and visual parts."""
        w, b = self.fc_out.weight, self.fc_out.bias
        d = a.shape[1]
        out_a = a @ w[:, :d].t() + b / 2
        out_v = v @ w[:, d:].t() + b / 2
        return out_a, out_v

    def coalition_logits(self, a, v):
        """Logits with a modality removed by zeroing its pooled feature (used for 2-player Shapley)."""
        za, zv = torch.zeros_like(a), torch.zeros_like(v)
        f = lambda x, y: self.fc_out(torch.cat([x, y], 1))  # noqa: E731
        return {"av": f(a, v), "a": f(a, zv), "v": f(za, v), "none": f(za, zv)}
