# P14' follow-up: CONSTRUCT vs the aggregation rules of Agent-MER and MultiMER on the same 16 samples (aggregate only)
NORM_MAP rules loaded: 211
Agent-MER k chosen on OV-MERD+: k = 3 of 16

## OV-MERD+ (n=532)
| aggregator | EW-F L1 | EW-F L2 | mean #labels | CONSTRUCT - aggregator [95% CI] |
|---|---|---|---|---|
| CONSTRUCT | 69.27 | 55.07 | 2.33 |  |
| CONSTRUCT + NORM_MAP | 69.18 | 54.28 | 2.29 | +0.08 [-0.49, +0.65] |
| Agent-MER vote k=3 (dev-tuned) | 68.15 | 53.66 | 3.92 | +1.12 [+0.06, +2.22] |
| Agent-MER vote k=3 (oracle on this set) | 68.15 | 53.66 | 3.92 | +1.12 [+0.06, +2.22] |
| MultiMER rule tau=0.29 (paper) | 68.09 | 52.46 | 4.00 | +1.18 [+0.06, +2.36] |
| MultiMER rule tau=0.4 (oracle on this set) | 68.47 | 53.05 | 3.54 | +0.80 [-0.25, +1.90] |
| MultiMER rule tau=0.4, no [3,6] clip (oracle) | 68.76 | 55.25 | 2.89 | +0.51 [-0.61, +1.64] |

## MER2025-FG (n=1000)
| aggregator | EW-F L1 | EW-F L2 | mean #labels | CONSTRUCT - aggregator [95% CI] |
|---|---|---|---|---|
| CONSTRUCT | 66.04 | 45.20 | 2.59 |  |
| CONSTRUCT + NORM_MAP | 66.17 | 45.23 | 2.48 | -0.13 [-0.64, +0.41] |
| Agent-MER vote k=3 (dev-tuned) | 64.63 | 45.91 | 3.84 | +1.41 [+0.49, +2.27] |
| Agent-MER vote k=2 (oracle on this set) | 67.88 | 47.24 | 6.98 | -1.84 [-2.84, -0.86] |
| MultiMER rule tau=0.29 (paper) | 67.77 | 47.53 | 4.24 | -1.73 [-2.65, -0.85] |
| MultiMER rule tau=0.2 (oracle on this set) | 68.41 | 47.57 | 4.86 | -2.37 [-3.33, -1.36] |
| MultiMER rule tau=0.2, no [3,6] clip (oracle) | 68.45 | 47.89 | 5.34 | -2.41 [-3.35, -1.44] |

## EmoS-MELD-fresh (n=400)
| aggregator | EW-F L1 | EW-F L2 | mean #labels | CONSTRUCT - aggregator [95% CI] |
|---|---|---|---|---|
| CONSTRUCT | 59.55 | 36.82 | 2.75 |  |
| CONSTRUCT + NORM_MAP | 58.82 | 36.19 | 2.64 | +0.73 [-0.02, +1.48] |
| Agent-MER vote k=3 (dev-tuned) | 57.93 | 34.03 | 4.36 | +1.62 [+0.07, +3.12] |
| Agent-MER vote k=3 (oracle on this set) | 57.93 | 34.03 | 4.36 | +1.62 [+0.07, +3.12] |
| MultiMER rule tau=0.29 (paper) | 57.44 | 33.77 | 4.34 | +2.11 [+0.64, +3.59] |
| MultiMER rule tau=0.45 (oracle on this set) | 57.71 | 34.18 | 3.55 | +1.84 [+0.40, +3.21] |
| MultiMER rule tau=0.45, no [3,6] clip (oracle) | 58.86 | 37.54 | 2.99 | +0.69 [-0.69, +2.19] |

## Diagnosis (exploratory, after the comparison)

**Ground-truth label count per clip vs label count in the model's samples:**

| Set | GT labels per clip, mean (median) | Labels per model sample, mean |
|---|---|---|
| OV-MERD+ | 3.84 (4) | 2.44 |
| MER2025-FG | 6.23 (6) | 2.46 |
| EmoS-MELD-fresh | 1.34 (1) | 2.72 |

**Corpus precision / recall (level 1):**

| Set | CONSTRUCT | MultiMER rule (τ = 0.29) | vote k = 3 |
|---|---|---|---|
| OV-MERD+ | P 69.58 / R 69.03 | P 63.11 / R 73.94 | P 65.96 / R 70.57 |
| MER2025-FG | P 73.08 / R 60.23 | P 67.00 / R 68.57 | P 69.35 / R 60.51 |
| EmoS-MELD-fresh | P 50.15 / R 73.57 | P 45.45 / R 78.36 | P 46.93 / R 76.01 |

**Forcing a minimum label count into CONSTRUCT does not close the gap** (MER2025-FG):

| Minimum labels | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| EW-F | 66.04 | 66.11 | 66.13 | 66.31 |

The forced labels are chosen by the same sample utility, so they add little recall.

**Reading:**
- CONSTRUCT reproduces the label density of the model's own samples, about 2.5 labels.
- The corpus EW-F optimum depends on the annotators' label density, which the samples do not reveal.
- MER2025-FG annotations are dense, so CONSTRUCT is precision-heavy there and loses to rules that emit at least 3 labels. EmoS annotations are sparse, so the same rules lose there.
