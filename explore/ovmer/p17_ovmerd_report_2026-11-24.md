# P17, OV-MERD+ only (532/532 clips): Qwen2.5-Omni-7B zero-shot + official extractor (aggregate scores only)

Protocol: explore/ovmer/p17_score.py (pre-registered). The pooled primary verdict needs all 3 sets; the 'verdict' lines below cover OV-MERD+ alone and are NOT the registered decision. Empty extractions: 11.2% of lists (official prompt allows an empty list).

## Extraction: labels
## OV-MERD+ (n=532)
| method | EW-F L1 (cross-fitted) | parameter chosen (mode) | CONSTRUCT-D - method [95% CI] |
|---|---|---|---|
| CONSTRUCT-D | 69.07 | 1.25 |  |
| Agent-MER vote | 68.27 | 2 | +0.80 [-0.30, +1.97] |
| MultiMER rule | 68.72 | 0.4 | +0.35 [-0.65, +1.38] |
| MultiMER rule (no clip) | 69.04 | 0.3 | +0.03 [-0.94, +1.00] |
| GREEDY | 60.46 | - | +8.61 [+6.52, +10.89] |
| TOPK | 64.92 | - | +4.15 [+2.73, +5.62] |
| MBR | 68.05 | - | +1.02 [+0.01, +2.03] |
| CONSTRUCT | 68.91 | - | +0.16 [-0.54, +0.88] |
- CONSTRUCT - GREEDY = +8.45 [+6.34, +10.73]
- CONSTRUCT - TOPK = +3.99 [+2.72, +5.30]
- CONSTRUCT - MBR = +0.87 [+0.16, +1.61]
- sample-count curve (1 repetition): N=1: CONSTRUCT-D 55.56, Agent-MER vote 48.00, MultiMER rule 48.52, MBR 53.38, TOPK 48.00 | N=2: CONSTRUCT-D 61.80, Agent-MER vote 59.97, MultiMER rule 60.50, MBR 57.81, TOPK 56.37 | N=4: CONSTRUCT-D 66.44, Agent-MER vote 63.35, MultiMER rule 64.60, MBR 63.36, TOPK 60.52 | N=8: CONSTRUCT-D 67.90, Agent-MER vote 63.88, MultiMER rule 66.97, MBR 66.48, TOPK 64.74

## Pooled over sets (clip-weighted)
- CONSTRUCT-D - Agent-MER vote (pooled) = +0.80 [-0.30, +1.97]
- CONSTRUCT-D - MultiMER rule (pooled) = +0.35 [-0.65, +1.38]
- CONSTRUCT-D - MultiMER rule (no clip) (pooled) = +0.03 [-0.94, +1.00]
- CONSTRUCT-D - MBR (pooled) = +1.02 [+0.01, +2.03]
- CONSTRUCT-D - TOPK (pooled) = +4.15 [+2.73, +5.62]

**P17 verdict (primary): CONSTRUCT-D DOES NOT HOLD** (significantly worse than a calibrated rule on some set: False)

## Extraction: labels_seed1
## OV-MERD+ (n=532)
| method | EW-F L1 (cross-fitted) | parameter chosen (mode) | CONSTRUCT-D - method [95% CI] |
|---|---|---|---|
| CONSTRUCT-D | 69.24 | 1.25 |  |
| Agent-MER vote | 67.98 | 2 | +1.25 [+0.17, +2.41] |
| MultiMER rule | 68.95 | 0.35 | +0.29 [-0.70, +1.31] |
| MultiMER rule (no clip) | 69.15 | 0.25 | +0.09 [-0.88, +1.07] |
| GREEDY | 60.69 | - | +8.54 [+6.47, +10.89] |
| TOPK | 64.97 | - | +4.27 [+2.75, +5.77] |
| MBR | 67.93 | - | +1.31 [+0.24, +2.36] |
| CONSTRUCT | 69.01 | - | +0.22 [-0.53, +1.02] |
- CONSTRUCT - GREEDY = +8.32 [+6.22, +10.70]
- CONSTRUCT - TOPK = +4.04 [+2.62, +5.39]
- CONSTRUCT - MBR = +1.09 [+0.34, +1.87]

## Pooled over sets (clip-weighted)
- CONSTRUCT-D - Agent-MER vote (pooled) = +1.25 [+0.17, +2.41]
- CONSTRUCT-D - MultiMER rule (pooled) = +0.29 [-0.70, +1.31]
- CONSTRUCT-D - MultiMER rule (no clip) (pooled) = +0.09 [-0.88, +1.07]
- CONSTRUCT-D - MBR (pooled) = +1.31 [+0.24, +2.36]
- CONSTRUCT-D - TOPK (pooled) = +4.27 [+2.75, +5.77]

**P17 verdict (primary): CONSTRUCT-D DOES NOT HOLD** (significantly worse than a calibrated rule on some set: False)

## Extraction: labels_oldprompt
## OV-MERD+ (n=532)
| method | EW-F L1 (cross-fitted) | parameter chosen (mode) | CONSTRUCT-D - method [95% CI] |
|---|---|---|---|
| CONSTRUCT-D | 69.60 | 1.5 |  |
| Agent-MER vote | 68.24 | 2 | +1.36 [+0.22, +2.53] |
| MultiMER rule | 68.89 | 0.35 | +0.71 [-0.34, +1.85] |
| MultiMER rule (no clip) | 68.77 | 0.35 | +0.82 [-0.30, +1.97] |
| GREEDY | 60.46 | - | +9.14 [+7.02, +11.47] |
| TOPK | 64.89 | - | +4.71 [+3.12, +6.30] |
| MBR | 68.30 | - | +1.29 [+0.12, +2.44] |
| CONSTRUCT | 69.13 | - | +0.47 [-0.45, +1.43] |
- CONSTRUCT - GREEDY = +8.67 [+6.53, +11.03]
- CONSTRUCT - TOPK = +4.24 [+2.83, +5.65]
- CONSTRUCT - MBR = +0.82 [+0.12, +1.50]

## Pooled over sets (clip-weighted)
- CONSTRUCT-D - Agent-MER vote (pooled) = +1.36 [+0.22, +2.53]
- CONSTRUCT-D - MultiMER rule (pooled) = +0.71 [-0.34, +1.85]
- CONSTRUCT-D - MultiMER rule (no clip) (pooled) = +0.82 [-0.30, +1.97]
- CONSTRUCT-D - MBR (pooled) = +1.29 [+0.12, +2.44]
- CONSTRUCT-D - TOPK (pooled) = +4.71 [+3.12, +6.30]

**P17 verdict (primary): CONSTRUCT-D DOES NOT HOLD** (significantly worse than a calibrated rule on some set: False)

## Label-free comparison (paper-default rules, no OV-MERD+ label used)

```
labels (extraction seed 0): GREEDY 60.46 | CONSTRUCT 68.91 | MultiMER rule tau=0.29 68.92 (CONSTRUCT - it -0.01 [-1.28, +1.19]) | Agent-MER vote k=6/16 44.82, k=8/16 29.10
labels_seed1:               GREEDY 60.69 | CONSTRUCT 69.01 | MultiMER rule tau=0.29 69.20 (-0.19 [-1.48, +1.10])
labels_oldprompt:           GREEDY 60.46 | CONSTRUCT 69.13 | MultiMER rule tau=0.29 68.89 (+0.24 [-0.99, +1.46])
```
Agent-MER's own threshold (4 of 10 runs, about 40%) does not transfer to these samples: 16 diverse samples and 11% empty lists. The fair comparison for it is the cross-fitted k (k = 2: 68.27).

## Benchmark row (OV-MERD+, EW-F level 1; no OV-MERD+ label used for our rows)

| System | Training | EW-F |
|---|---|---|
| Qwen2.5-Omni-7B zero-shot, greedy (ours, official protocol) | none | 60.46 |
| **Qwen2.5-Omni-7B zero-shot + CONSTRUCT** | none | **68.91** (extraction seed 1: 69.01; older prompt: 69.13) |
| AffectGPT (published) | SFT 7B | 62.52 |
| AffectGPT-R1 / -RL (published; best of epochs 5–60 on OV-MERD+) | SFT + RL 7B | 68.39 |
| MER-R1 (published, final checkpoint) | SFT + RL, Qwen2.5-Omni (size not reported) | 70.68 |

## Headroom of sample-based decoding (exploratory; the oracles use the GT, so they are ceilings, not methods)

| Decoder | EW-F | Note |
|---|---|---|
| CONSTRUCT (label-free) | 68.91 | P 69.4 / R 68.5 |
| Oracle: best single sampled list per clip | 84.73 | uses GT |
| Oracle: best subset of the sampled labels per clip (greedy on the per-clip F against GT) | 91.48 | uses GT |

In 93.8% of the (clip, wheel) pairs, the pooled sampled labels contain at least one correct group.

**Reading:** the 16 samples already hold sets far above every published result. The bottleneck is the label-free selection signal (sample frequency), not the coverage of the samples.
