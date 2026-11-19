"""CREMA-D audio-visual dataset reading the files produced by prepare_cremad.py.

Transforms follow OGM-GE (train: RandomResizedCrop(224) + HorizontalFlip; eval: Resize(224, 224);
ImageNet normalisation; spectrogram not normalised, as in OGM-GE).
Deviation from OGM-GE (documented): OGM-GE computes a random frame index but then loads the first
`fps` files of an unsorted directory listing. Here training frames are sampled uniformly at random and
evaluation frames are evenly spaced, both deterministic given the RNG.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common import CLASS_DICT, read_split

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CremadAV(Dataset):
    def __init__(self, root, split_csv, train, n_frames=1, fps=1):
        self.root, self.train, self.n_frames = root, train, n_frames
        self.frame_dir = os.path.join(root, f"frames_fps{fps}")
        rows = read_split(split_csv)
        self.items = []
        for clip, lab in rows:
            sp = os.path.join(root, "spec", clip + ".npy")
            fd = os.path.join(self.frame_dir, clip)
            if os.path.exists(sp) and os.path.isdir(fd) and os.listdir(fd):
                self.items.append((clip, CLASS_DICT[lab]))
        if len(self.items) < len(rows):
            print(f"[data] {split_csv}: kept {len(self.items)}/{len(rows)} clips with spec+frames")
        if train:
            self.tf = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        else:
            self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                          transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def clips(self):
        return [c for c, _ in self.items]

    def __getitem__(self, i):
        clip, label = self.items[i]
        spec = torch.from_numpy(np.load(os.path.join(self.root, "spec", clip + ".npy")).astype(np.float32))
        spec = spec.unsqueeze(0)  # (1, 257, T)
        fd = os.path.join(self.frame_dir, clip)
        files = sorted(os.listdir(fd))
        n = len(files)
        if self.train:
            idx = torch.randperm(n)[: self.n_frames].sort().values.tolist()
        else:
            idx = np.linspace(0, n - 1, num=min(self.n_frames, n)).round().astype(int).tolist()
        while len(idx) < self.n_frames:  # clips shorter than n_frames seconds
            idx.append(idx[-1])
        imgs = torch.stack([self.tf(Image.open(os.path.join(fd, files[j])).convert("RGB")) for j in idx])
        imgs = imgs.permute(1, 0, 2, 3)  # (3, T, 224, 224) as in OGM-GE
        return spec, imgs, label, i
