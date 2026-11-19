"""Shared constants and helpers for the CREMA-D modality-imbalance pilot."""
import csv
import os
import random

import numpy as np

# Class order follows OGM-GE (GeWu-Lab/OGM-GE_CVPR2022, dataset/CramedDataset.py).
CLASSES = ["NEU", "HAP", "SAD", "FEA", "DIS", "ANG"]
CLASS_DICT = {c: i for i, c in enumerate(CLASSES)}
# Letters used in CREMA-D finishedResponses.csv (respEmo column).
LETTER = {"NEU": "N", "HAP": "H", "SAD": "S", "FEA": "F", "DIS": "D", "ANG": "A"}
LETTERS = ["N", "H", "S", "F", "D", "A"]  # same order as CLASSES


def actor_of(clip: str) -> str:
    return clip.split("_")[0]


def emo_of(clip: str) -> str:
    return clip.split("_")[2]


def read_split(path):
    """Read an OGM-GE style split file: rows of `clip,LABEL` (no header)."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if not r or r[0] in ("clip", "name"):
                continue
            rows.append((r[0].strip(), r[1].strip()))
    return rows


def write_split(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for clip, lab in rows:
            w.writerow([clip, lab])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
