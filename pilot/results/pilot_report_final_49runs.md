Runs found: 49; used: 49; excluded (frame-dependent, dump_version < 2): 0

## G1 Protocol (test accuracy, %)
| protocol | method | n runs | val-selected | test-selected (OGM-GE style) |
|---|---|---|---|---|
| actor | audio_only | 4 | 51.24 ± 1.18 | 53.17 ± 1.21 |
| actor | joint | 4 | 50.43 ± 1.19 | 53.45 ± 0.11 |
| actor | ogm_ge(a=0.1) | 4 | 52.34 ± 1.84 | 53.42 ± 1.29 |
| actor | ogm_ge(a=0.3) | 4 | 53.29 ± 0.74 | 54.37 ± 0.91 |
| actor | ogm_ge(a=0.5) | 4 | 52.48 ± 3.63 | 55.23 ± 0.81 |
| actor | ogm_ge(a=0.8) | 4 | 55.17 ± 1.23 | 56.75 ± 0.80 |
| actor | visual_only | 4 | 42.13 ± 1.86 | 43.14 ± 2.20 |
| standard | audio_only | 3 | 51.61 ± 2.00 | 53.18 ± 2.65 |
| standard | joint | 3 | 51.84 ± 2.25 | 54.03 ± 0.19 |
| standard | ogm_ge(a=0.1) | 3 | 51.48 ± 0.95 | 52.28 ± 1.05 |
| standard | ogm_ge(a=0.3) | 3 | 54.30 ± 0.83 | 55.65 ± 0.38 |
| standard | ogm_ge(a=0.5) | 3 | 53.36 ± 2.62 | 55.78 ± 1.21 |
| standard | ogm_ge(a=0.8) | 3 | 58.33 ± 1.81 | 59.23 ± 1.17 |
| standard | visual_only | 3 | 47.49 ± 1.01 | 48.57 ± 0.78 |
- actor: OGM-GE alpha chosen on val = 0.8; gain over joint on test = +4.74 points
- standard: OGM-GE alpha chosen on val = 0.8; gain over joint on test = +6.50 points

Eval-frame check (same checkpoint; test accuracy %, centre frame vs first frame):
- joint: centre 51.03 vs first 50.92 (diff +0.12, n=7)
- ogm_ge(a=0.1): centre 51.97 vs first 51.84 (diff +0.13, n=7)
- ogm_ge(a=0.3): centre 53.73 vs first 52.51 (diff +1.22, n=7)
- ogm_ge(a=0.5): centre 52.86 vs first 50.89 (diff +1.97, n=7)
- ogm_ge(a=0.8): centre 56.53 vs first 53.19 (diff +3.33, n=7)
- visual_only: centre 44.43 vs first 26.35 (diff +18.07, n=7)

## G2 Q1: alignment of model modality reliance with humans (test clips)
| protocol | method | n | align label-free (primary) | align label-based | visual share (label-free) |
|---|---|---|---|---|---|
| actor | joint | 4 | -0.039 ± 0.011 | -0.021 | 0.269 |
| actor | ogm_ge(a=0.1) | 4 | -0.028 ± 0.018 | 0.038 | 0.260 |
| actor | ogm_ge(a=0.3) | 4 | 0.033 ± 0.008 | 0.140 | 0.349 |
| actor | ogm_ge(a=0.5) | 4 | 0.081 ± 0.037 | 0.181 | 0.475 |
| actor | ogm_ge(a=0.8) | 4 | 0.139 ± 0.024 | 0.250 | 0.520 |
| standard | joint | 3 | -0.055 ± 0.018 | -0.012 | 0.286 |
| standard | ogm_ge(a=0.1) | 3 | -0.062 ± 0.022 | 0.019 | 0.288 |
| standard | ogm_ge(a=0.3) | 3 | -0.025 ± 0.030 | 0.061 | 0.354 |
| standard | ogm_ge(a=0.5) | 3 | 0.037 ± 0.028 | 0.134 | 0.497 |
| standard | ogm_ge(a=0.8) | 3 | 0.182 ± 0.052 | 0.240 | 0.558 |

Targeting (OGM-GE minus joint, paired by split and seed; label-free):
- actor ogm_ge(a=0.1): targeting = 0.005 ± 0.032, delta visual share = -0.009 (n=4)
- actor ogm_ge(a=0.3): targeting = 0.084 ± 0.018, delta visual share = +0.081 (n=4)
- actor ogm_ge(a=0.5): targeting = 0.128 ± 0.035, delta visual share = +0.206 (n=4)
- actor ogm_ge(a=0.8): targeting = 0.165 ± 0.032, delta visual share = +0.251 (n=4)
- standard ogm_ge(a=0.1): targeting = -0.055 ± 0.030, delta visual share = +0.002 (n=3)
- standard ogm_ge(a=0.3): targeting = 0.050 ± 0.044, delta visual share = +0.068 (n=3)
- standard ogm_ge(a=0.5): targeting = 0.111 ± 0.030, delta visual share = +0.211 (n=3)
- standard ogm_ge(a=0.8): targeting = 0.204 ± 0.036, delta visual share = +0.272 (n=3)

## G3 Q2 / H: does human-aligned reliance predict accuracy on unseen actors?
- G3a [label-free, DECISION] (n=20 runs): partial rank corr(alignment on val actors, test acc | val acc) = 0.650, 95% bootstrap CI [0.387, 0.880]
- G3a [label-based, reference only] (n=20 runs): partial rank corr(alignment on val actors, test acc | val acc) = 0.626, 95% bootstrap CI [0.383, 0.849]
- G3b [descriptive] actor0: P(correct | label-free reliance agrees with humans) - P(correct | disagrees), stratified by human AV accuracy = +4.30 points, 95% CI [+1.54, +7.25] (clips=1436, clip-runs=14360; share agreeing = 0.45)
- G3b [descriptive] actor1: P(correct | label-free reliance agrees with humans) - P(correct | disagrees), stratified by human AV accuracy = +2.64 points, 95% CI [-0.19, +5.43] (clips=1360, clip-runs=13600; share agreeing = 0.49)

## Decision support (rule amended 2026-11-20, before any OGM-GE result existed; see README)
- GO (strong): G3a label-free estimate > 0 with 95% CI excluding 0 -> MET
- GO (weak, discuss): G2 uniform visual boost (|targeting| < 0.05 and visual share rises) -> not met; AND G1 OGM-GE gain under the actor protocol < half of the standard-split gain -> not met
- Outcome: GO (strong)
