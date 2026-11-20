Runs found: 25; used: 25; excluded (frame-dependent, dump_version < 2): 0

## G1 Protocol (test accuracy, %)
| protocol | method | n runs | val-selected | test-selected (OGM-GE style) |
|---|---|---|---|---|
| actor | audio_only | 2 | 50.45 ± 0.32 | 53.03 ± 0.59 |
| actor | joint | 2 | 51.51 ± 0.23 | 53.54 ± 0.06 |
| actor | ogm_ge(a=0.1) | 2 | 53.58 ± 0.09 | 53.97 ± 0.22 |
| actor | ogm_ge(a=0.3) | 2 | 53.69 ± 0.27 | 54.10 ± 0.07 |
| actor | ogm_ge(a=0.5) | 2 | 50.41 ± 4.17 | 55.47 ± 1.03 |
| actor | ogm_ge(a=0.8) | 2 | 56.38 ± 0.32 | 57.22 ± 0.90 |
| actor | visual_only | 2 | 42.86 ± 0.03 | 44.30 ± 0.92 |
| standard | audio_only | 2 | 51.01 ± 2.22 | 53.36 ± 3.23 |
| standard | joint | 1 | 53.49 ± 0.00 | 54.17 ± 0.00 |
| standard | ogm_ge(a=0.1) | 2 | 51.14 ± 1.01 | 51.68 ± 0.74 |
| standard | ogm_ge(a=0.3) | 2 | 53.76 ± 0.40 | 55.78 ± 0.40 |
| standard | ogm_ge(a=0.5) | 1 | 56.18 ± 0.00 | 57.26 ± 0.00 |
| standard | ogm_ge(a=0.8) | 1 | 56.99 ± 0.00 | 58.47 ± 0.00 |
| standard | visual_only | 2 | 47.85 ± 1.08 | 48.66 ± 0.94 |
- actor: OGM-GE alpha chosen on val = 0.8; gain over joint on test = +4.86 points
- standard: OGM-GE alpha chosen on val = 0.8; gain over joint on test = +3.49 points

Eval-frame check (same checkpoint; test accuracy %, centre frame vs first frame):
- joint: centre 52.17 vs first 52.11 (diff +0.07, n=3)
- ogm_ge(a=0.1): centre 52.36 vs first 52.21 (diff +0.15, n=4)
- ogm_ge(a=0.3): centre 53.73 vs first 52.51 (diff +1.22, n=4)
- ogm_ge(a=0.5): centre 52.33 vs first 49.65 (diff +2.69, n=3)
- ogm_ge(a=0.8): centre 56.58 vs first 53.35 (diff +3.23, n=3)
- visual_only: centre 45.36 vs first 26.61 (diff +18.74, n=4)

## G2 Q1: alignment of model modality reliance with humans (test clips)
| protocol | method | n | align label-free (primary) | align label-based | visual share (label-free) |
|---|---|---|---|---|---|
| actor | joint | 2 | -0.034 ± 0.014 | 0.047 | 0.201 |
| actor | ogm_ge(a=0.1) | 2 | -0.015 ± 0.015 | 0.054 | 0.268 |
| actor | ogm_ge(a=0.3) | 2 | 0.029 ± 0.001 | 0.120 | 0.348 |
| actor | ogm_ge(a=0.5) | 2 | 0.045 ± 0.004 | 0.166 | 0.575 |
| actor | ogm_ge(a=0.8) | 2 | 0.131 ± 0.002 | 0.227 | 0.420 |
| standard | joint | 1 | -0.032 ± 0.000 | 0.040 | 0.258 |
| standard | ogm_ge(a=0.1) | 2 | -0.073 ± 0.019 | 0.016 | 0.290 |
| standard | ogm_ge(a=0.3) | 2 | -0.011 ± 0.027 | 0.107 | 0.293 |
| standard | ogm_ge(a=0.5) | 1 | 0.046 ± 0.000 | 0.129 | 0.472 |
| standard | ogm_ge(a=0.8) | 1 | 0.244 ± 0.000 | 0.257 | 0.538 |

Targeting (OGM-GE minus joint, paired by split and seed; label-free):
- actor ogm_ge(a=0.1): targeting = 0.035 ± 0.000, delta visual share = +0.027 (n=1)
- actor ogm_ge(a=0.5): targeting = 0.080 ± 0.000, delta visual share = +0.352 (n=1)
- actor ogm_ge(a=0.8): targeting = 0.152 ± 0.003, delta visual share = +0.219 (n=2)
- standard ogm_ge(a=0.1): targeting = -0.037 ± 0.000, delta visual share = +0.072 (n=1)
- standard ogm_ge(a=0.8): targeting = 0.225 ± 0.000, delta visual share = +0.280 (n=1)

## G3 Q2 / H: does human-aligned reliance predict accuracy on unseen actors?
- G3a [label-free, DECISION] (n=10 runs): partial rank corr(alignment on val actors, test acc | val acc) = 0.267, 95% bootstrap CI [-0.374, 0.806]
- G3a [label-based, reference only] (n=10 runs): partial rank corr(alignment on val actors, test acc | val acc) = 0.446, 95% bootstrap CI [-0.311, 0.909]
- G3b [descriptive] actor0: P(correct | label-free reliance agrees with humans) - P(correct | disagrees), stratified by human AV accuracy = +4.22 points, 95% CI [+1.38, +7.09] (clips=1436, clip-runs=8616; share agreeing = 0.46)
- G3b [descriptive] actor1: P(correct | label-free reliance agrees with humans) - P(correct | disagrees), stratified by human AV accuracy = +2.83 points, 95% CI [-0.57, +6.17] (clips=1360, clip-runs=5440; share agreeing = 0.47)

## Decision support (rule amended 2026-11-20, before any OGM-GE result existed; see README)
- GO (strong): G3a label-free estimate > 0 with 95% CI excluding 0 -> not met
- GO (weak, discuss): G2 uniform visual boost (|targeting| < 0.05 and visual share rises) -> not met; AND G1 OGM-GE gain under the actor protocol < half of the standard-split gain -> not met
- Outcome: NO-GO
