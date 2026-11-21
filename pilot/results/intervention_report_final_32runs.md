Runs: 32 | per condition: {'anti': 8, 'human': 8, 'shuffled': 8, 'uniform': 8}

## Means per condition (over all runs)
| condition | n | acc last-10 (%) | acc val-selected (%) | alignment | visual share |
|---|---|---|---|---|---|
| human | 8 | 56.66 ± 1.11 | 56.61 | 0.203 ± 0.041 | 0.109 |
| shuffled | 8 | 56.48 ± 1.03 | 56.19 | 0.164 ± 0.016 | 0.072 |
| anti | 8 | 53.85 ± 1.19 | 53.96 | 0.019 ± 0.047 | 0.050 |
| uniform | 8 | 57.26 ± 1.44 | 56.94 | 0.161 ± 0.028 | 0.082 |

## Paired contrasts over (fold, seed) pairs
- human - shuffled (n pairs = 8): Delta acc last-10 = +0.18 [-1.56, +1.92] (pairs > 0: 4/8); Delta acc val-selected = +0.42; Delta align = +0.038 [+0.002, +0.074]; Delta visual share = +0.038
- human - anti (n pairs = 8): Delta acc last-10 = +2.81 [+1.50, +4.12] (pairs > 0: 8/8); Delta acc val-selected = +2.65; Delta align = +0.184 [+0.141, +0.227]; Delta visual share = +0.059
- human - uniform (n pairs = 8): Delta acc last-10 = -0.60 [-1.81, +0.61] (pairs > 0: 2/8); Delta acc val-selected = -0.33; Delta align = +0.042 [-0.009, +0.093]; Delta visual share = +0.028

- Reference: two-level bootstrap of Delta acc (val-selected checkpoints; pairs then clips): +0.42 [-1.11, +1.97]

## Decision (rule fixed in INTERVENTION.md before any run)
- Manipulation check (Delta align > 0, CI excludes 0): PASSED
- Effect (Delta acc last-10 > 0, CI excludes 0): NO
- Outcome: H NOT SUPPORTED
