# LoRA-Aware Cache Policy Sweep — Results

Total runs in results: **198** (198 ok, 0 timed out, 0 failed).

## Scenario: `adapter_thrashing`

_Round-robin across all adapters: every request hits a different adapter than the previous, so blocks rarely repeat. Tests how well the policy avoids holding dead-adapter blocks._

| Rank | Policy | OutTok/s | TotTok/s | HitRate | Evicted | P95 (s) |
|-----:|:-------|---------:|---------:|--------:|--------:|--------:|
| 1 | `lru` |   23.8 | 3050.2 | 49.86% |   678 |  4.30 |
| 2 | `lru_k` |   23.5 | 3003.8 | 49.86% |   678 |  4.33 |
| 3 | `lora_costaware:s3fifo` |   23.4 | 2997.8 | 49.53% |   678 |  4.27 |
| 4 | `lora_tight:lru_k` |   23.4 | 2992.2 | 33.00% |  1631 |  4.28 |
| 5 | `lora_correlated:lru` |   23.4 | 2989.6 | 49.70% |   684 |  4.15 |
| 6 | `lora_loose:s3fifo` |   23.3 | 2981.4 | 49.86% |   678 |  3.74 |
| 7 | `lora_twolevel` |   23.2 | 2971.0 | 19.45% |  1877 |  4.26 |
| 8 | `lora_position:s3fifo` |   23.2 | 2964.1 | 49.86% |   678 |  4.36 |
| 9 | `s3fifo` |   23.1 | 2957.4 | 52.15% |   678 |  4.27 |
| 10 | `lora_position:sieve` |   23.1 | 2955.3 | 38.44% |  1658 |  3.78 |
| 11 | `lora_prefixtree:lru` |   23.0 | 2945.9 | 49.70% |   678 |  4.16 |
| 12 | `lora_tight:tinylfu` |   23.0 | 2942.3 | 33.00% |  1631 |  4.20 |
| 13 | `lora_correlated:tinylfu` |   23.0 | 2940.0 | 31.29% |  1476 |  4.21 |
| 14 | `lora_loose:tinylfu` |   22.9 | 2935.7 | 32.42% |  1293 |  4.28 |
| 15 | `lora_adabudget:sieve` |   22.9 | 2935.4 | 36.32% |  1476 |  4.54 |
| 16 | `lora_adabudget:s3fifo` |   22.9 | 2934.0 | 49.86% |   678 |  4.31 |
| 17 | `lora_costaware:lru_k` |   22.9 | 2935.1 | 49.86% |   678 |  4.26 |
| 18 | `lora_tight:lru` |   22.8 | 2914.6 | 33.00% |  1631 |  3.73 |
| 19 | `lora_costaware:lru` |   22.7 | 2909.4 | 49.52% |   678 |  3.97 |
| 20 | `lora_tight:sieve` |   22.7 | 2908.0 | 33.00% |  1631 |  4.32 |
| 21 | `lora_prefixtree:tinylfu` |   22.7 | 2906.8 | 29.99% |  1287 |  4.19 |
| 22 | `lora_loose:lru_k` |   22.7 | 2904.6 | 49.86% |   678 |  4.41 |
| 23 | `lora_budget:sieve` |   22.7 | 2903.5 | 43.96% |   889 |  4.31 |
| 24 | `lora_hysteresis:s3fifo` |   22.6 | 2895.3 | 23.96% |  1820 |  4.50 |
| 25 | `lora_adabudget:lru_k` |   22.6 | 2893.3 | 48.87% |   678 |  4.30 |
| 26 | `lora_prefixtree:lru_k` |   22.6 | 2892.1 | 49.86% |   678 |  4.42 |
| 27 | `lora_soft:lru` |   22.6 | 2891.6 | 44.31% |   873 |  4.37 |
| 28 | `lora_correlated:sieve` |   22.6 | 2890.5 | 36.70% |  1435 |  4.32 |
| 29 | `lora_position:lru_k` |   22.6 | 2891.7 | 49.86% |   678 |  4.18 |
| 30 | `lora_budget:s3fifo` |   22.6 | 2888.3 | 49.86% |   678 |  4.59 |
| 31 | `lora_position:lru` |   22.6 | 2885.2 | 49.86% |   678 |  4.67 |
| 32 | `lora_soft:s3fifo` |   22.5 | 2883.9 | 49.70% |   678 |  4.24 |
| 33 | `lora_prefixtree:s3fifo` |   22.5 | 2877.7 | 49.86% |   678 |  4.56 |
| 34 | `tinylfu` |   22.5 | 2875.1 | 29.11% |  1614 |  4.23 |
| 35 | `lora_correlated:s3fifo` |   22.4 | 2869.7 | 50.50% |   678 |  4.25 |
| 36 | `lora_ghost:sieve` |   22.4 | 2868.4 | 33.00% |  1631 |  3.77 |
| 37 | `lora_adabudget:lru` |   22.4 | 2867.0 | 49.17% |   678 |  4.81 |
| 38 | `lora_hysteresis:sieve` |   22.4 | 2862.8 | 23.96% |  1820 |  4.18 |
| 39 | `lora_loose:sieve` |   22.3 | 2856.2 | 37.65% |  1476 |  3.88 |
| 40 | `lora_ghost:tinylfu` |   22.3 | 2854.4 | 32.57% |  1631 |  4.55 |
| 41 | `lora_soft:lru_k` |   22.3 | 2850.6 | 49.52% |   678 |  4.41 |
| 42 | `lora_freqweighted:lru` |   22.2 | 2844.8 | 18.23% |  2187 |  4.51 |
| 43 | `lora_budget:lru` |   22.1 | 2831.4 | 49.86% |   678 |  4.32 |
| 44 | `lora_hysteresis:lru` |   22.1 | 2829.1 | 26.12% |  1972 |  4.39 |
| 45 | `lora_costaware:tinylfu` |   22.1 | 2825.8 | 27.84% |  1529 |  4.34 |
| 46 | `lora_ghost:s3fifo` |   22.1 | 2824.6 | 33.00% |  1631 |  4.33 |
| 47 | `lora_costaware:sieve` |   22.1 | 2822.5 | 36.31% |  1658 |  4.67 |
| 48 | `lora_freqweighted:tinylfu` |   22.0 | 2818.6 | 22.16% |  2224 |  4.25 |
| 49 | `sieve` |   22.0 | 2815.2 | 36.92% |  1658 |  4.15 |
| 50 | `lora_soft:sieve` |   21.9 | 2808.3 | 35.13% |  1393 |  4.27 |
| 51 | `lora_freqweighted:sieve` |   21.9 | 2806.8 | 18.23% |  2187 |  4.30 |
| 52 | `lora_budget:tinylfu` |   21.8 | 2793.4 | 29.27% |  1007 |  4.19 |
| 53 | `lora_adabudget:tinylfu` |   21.8 | 2788.7 | 31.25% |  1295 |  4.65 |
| 54 | `lora_freqweighted:s3fifo` |   21.8 | 2785.2 | 18.23% |  2187 |  4.28 |
| 55 | `lora_hysteresis:tinylfu` |   21.7 | 2778.6 | 23.96% |  1820 |  4.14 |
| 56 | `lora_tight:s3fifo` |   21.7 | 2777.7 | 33.53% |  1631 |  4.76 |
| 57 | `lora_soft:tinylfu` |   21.5 | 2749.7 | 41.83% |   986 |  4.24 |
| 58 | `lora_ghost:lru_k` |   21.5 | 2746.9 | 33.00% |  1631 |  4.54 |
| 59 | `lora_freqweighted:lru_k` |   21.2 | 2709.7 | 18.23% |  2187 |  4.49 |
| 60 | `lora_correlated:lru_k` |   21.2 | 2709.4 | 49.86% |   678 |  4.35 |
| 61 | `lora_hysteresis:lru_k` |   21.1 | 2699.1 | 23.96% |  1820 |  4.63 |
| 62 | `lora_position:tinylfu` |   21.0 | 2688.1 | 29.37% |  1340 |  4.54 |
| 63 | `lora_ghost:lru` |   20.9 | 2674.6 | 33.00% |  1631 |  4.41 |
| 64 | `lora_prefixtree:sieve` |   20.9 | 2669.5 | 36.92% |  1658 |  4.43 |
| 65 | `lora_loose:lru` |   20.9 | 2666.7 | 49.86% |   678 |  5.12 |
| 66 | `lora_budget:lru_k` |   20.6 | 2640.3 | 49.86% |   678 |  4.51 |

### LoRA coupling uplift over bare base — `adapter_thrashing`

Positive `Δ_thr` = LoRA decorator beats the bare inner base on output throughput. `n/a` means base run was missing/failed.

| Coupling \ Base | lru | lru_k | s3fifo | sieve | tinylfu |
|---|---|---|---|---|---|
| `lora_adabudget` | -1.4 (-6.0%) | -0.9 (-3.7%) | -0.2 (-0.8%) | +0.9 (+4.3%) | -0.7 (-3.0%) |
| `lora_budget` | -1.7 (-7.2%) | -2.8 (-12.1%) | -0.5 (-2.3%) | +0.7 (+3.1%) | -0.6 (-2.8%) |
| `lora_correlated` | -0.5 (-2.0%) | -2.3 (-9.8%) | -0.7 (-3.0%) | +0.6 (+2.7%) | +0.5 (+2.3%) |
| `lora_costaware` | -1.1 (-4.6%) | -0.5 (-2.3%) | +0.3 (+1.3%) | +0.0 (+0.2%) | -0.4 (-1.7%) |
| `lora_freqweighted` | -1.6 (-6.7%) | -2.3 (-9.8%) | -1.4 (-5.8%) | -0.1 (-0.3%) | -0.4 (-2.0%) |
| `lora_ghost` | -2.9 (-12.3%) | -2.0 (-8.6%) | -1.0 (-4.5%) | +0.4 (+1.9%) | -0.2 (-0.7%) |
| `lora_hysteresis` | -1.7 (-7.2%) | -2.4 (-10.1%) | -0.5 (-2.1%) | +0.4 (+1.7%) | -0.8 (-3.3%) |
| `lora_loose` | -3.0 (-12.5%) | -0.8 (-3.3%) | +0.2 (+0.8%) | +0.3 (+1.5%) | +0.5 (+2.1%) |
| `lora_position` | -1.3 (-5.4%) | -0.9 (-3.7%) | +0.1 (+0.2%) | +1.1 (+5.0%) | -1.5 (-6.5%) |
| `lora_prefixtree` | -0.8 (-3.4%) | -0.9 (-3.7%) | -0.6 (-2.7%) | -1.1 (-5.2%) | +0.2 (+1.1%) |
| `lora_soft` | -1.2 (-5.2%) | -1.2 (-5.1%) | -0.6 (-2.5%) | -0.1 (-0.3%) | -1.0 (-4.4%) |
| `lora_tight` | -1.1 (-4.4%) | -0.1 (-0.4%) | -1.4 (-6.1%) | +0.7 (+3.3%) | +0.5 (+2.4%) |

## Scenario: `adapter_locality`

_Bursts of consecutive requests on the same adapter. Tests whether the policy preserves grouped per-adapter blocks while pruning older adapter groups._

| Rank | Policy | OutTok/s | TotTok/s | HitRate | Evicted | P95 (s) |
|-----:|:-------|---------:|---------:|--------:|--------:|--------:|
| 1 | `lora_ghost:sieve` |   90.6 | 11585.0 |  0.00% |   309 |  1.42 |
| 2 | `s3fifo` |   89.2 | 11401.7 |  0.00% |   276 |  1.43 |
| 3 | `lora_hysteresis:s3fifo` |   89.0 | 11387.0 |  0.00% |   309 |  0.97 |
| 4 | `lora_budget:sieve` |   88.7 | 11346.2 |  0.00% |   276 |  1.38 |
| 5 | `lora_tight:lru_k` |   88.6 | 11332.8 |  0.00% |   309 |  0.95 |
| 6 | `lora_costaware:lru` |   88.6 | 11331.4 |  0.00% |   276 |  1.40 |
| 7 | `lora_adabudget:sieve` |   88.4 | 11307.0 |  0.00% |   276 |  1.39 |
| 8 | `lora_correlated:lru` |   88.1 | 11267.9 |  0.00% |   276 |  0.99 |
| 9 | `lora_correlated:sieve` |   87.4 | 11176.9 |  0.00% |   276 |  1.40 |
| 10 | `lora_tight:lru` |   87.4 | 11172.2 |  0.00% |   309 |  1.42 |
| 11 | `lora_position:sieve` |   87.3 | 11167.2 |  0.00% |   276 |  1.52 |
| 12 | `lora_budget:lru` |   86.6 | 11079.6 |  0.00% |   276 |  0.98 |
| 13 | `lora_prefixtree:sieve` |   86.5 | 11064.1 |  0.00% |   276 |  1.38 |
| 14 | `lora_prefixtree:s3fifo` |   85.9 | 10981.4 |  0.00% |   276 |  1.55 |
| 15 | `lora_hysteresis:tinylfu` |   85.8 | 10977.2 |  0.00% |   309 |  1.43 |
| 16 | `lora_soft:tinylfu` |   85.4 | 10916.8 |  0.00% |   276 |  1.41 |
| 17 | `lora_adabudget:lru` |   85.3 | 10909.5 |  0.00% |   276 |  1.18 |
| 18 | `lora_loose:s3fifo` |   84.9 | 10861.2 |  0.00% |   276 |  1.39 |
| 19 | `sieve` |   84.6 | 10824.5 |  0.00% |   276 |  1.51 |
| 20 | `lora_costaware:sieve` |   84.5 | 10800.0 |  0.00% |   276 |  1.40 |
| 21 | `lora_ghost:lru` |   84.1 | 10755.2 |  0.00% |   309 |  1.08 |
| 22 | `lora_loose:sieve` |   83.5 | 10682.6 |  0.00% |   276 |  1.62 |
| 23 | `lora_loose:lru` |   82.1 | 10499.7 |  0.00% |   276 |  1.59 |
| 24 | `lora_soft:lru_k` |   82.1 | 10495.8 |  0.00% |   276 |  1.63 |
| 25 | `tinylfu` |   66.5 | 8498.8 |  0.00% |   276 |  1.41 |
| 26 | `lora_ghost:lru_k` |   66.3 | 8479.7 |  0.00% |   309 |  1.43 |
| 27 | `lora_adabudget:lru_k` |   66.2 | 8460.6 |  0.00% |   276 |  1.40 |
| 28 | `lora_budget:lru_k` |   65.8 | 8413.2 |  0.00% |   276 |  1.43 |
| 29 | `lora_budget:tinylfu` |   65.7 | 8403.9 |  0.00% |   276 |  1.49 |
| 30 | `lora_correlated:s3fifo` |   65.7 | 8399.2 |  0.00% |   276 |  1.45 |
| 31 | `lora_freqweighted:sieve` |   65.4 | 8362.8 |  0.00% |   309 |  1.42 |
| 32 | `lora_tight:tinylfu` |   65.4 | 8360.4 |  0.00% |   309 |  1.50 |
| 33 | `lora_tight:s3fifo` |   65.3 | 8354.9 |  0.00% |   309 |  1.49 |
| 34 | `lru_k` |   65.3 | 8347.1 |  0.00% |   276 |  1.44 |
| 35 | `lora_costaware:tinylfu` |   65.2 | 8344.4 |  0.00% |   276 |  1.46 |
| 36 | `lora_soft:lru` |   65.2 | 8343.8 |  0.00% |   276 |  1.44 |
| 37 | `lora_hysteresis:lru_k` |   65.2 | 8341.3 |  0.00% |   309 |  1.44 |
| 38 | `lora_position:lru_k` |   65.2 | 8338.3 |  0.00% |   276 |  1.55 |
| 39 | `lora_hysteresis:sieve` |   65.0 | 8319.2 |  0.00% |   309 |  1.43 |
| 40 | `lora_soft:sieve` |   65.0 | 8317.0 |  0.00% |   276 |  1.50 |
| 41 | `lora_adabudget:s3fifo` |   65.0 | 8305.9 |  0.00% |   276 |  1.41 |
| 42 | `lora_ghost:s3fifo` |   64.9 | 8303.7 |  0.00% |   309 |  1.44 |
| 43 | `lora_hysteresis:lru` |   64.9 | 8300.6 |  0.00% |   309 |  1.59 |
| 44 | `lora_costaware:s3fifo` |   64.8 | 8291.5 |  0.00% |   276 |  1.51 |
| 45 | `lora_position:s3fifo` |   64.8 | 8282.9 |  0.00% |   276 |  1.43 |
| 46 | `lora_tight:sieve` |   64.7 | 8276.4 |  0.00% |   309 |  1.47 |
| 47 | `lora_position:tinylfu` |   64.7 | 8272.7 |  0.00% |   276 |  1.46 |
| 48 | `lora_adabudget:tinylfu` |   64.7 | 8270.0 |  0.00% |   276 |  1.44 |
| 49 | `lora_correlated:lru_k` |   64.6 | 8263.0 |  0.00% |   276 |  1.45 |
| 50 | `lora_ghost:tinylfu` |   64.3 | 8228.1 |  0.00% |   309 |  1.53 |
| 51 | `lora_position:lru` |   64.2 | 8212.1 |  0.00% |   276 |  1.54 |
| 52 | `lora_prefixtree:lru` |   64.2 | 8208.8 |  0.00% |   276 |  1.47 |
| 53 | `lora_freqweighted:lru` |   63.8 | 8160.2 |  0.00% |   309 |  2.29 |
| 54 | `lora_freqweighted:lru_k` |   63.8 | 8155.2 |  0.00% |   309 |  1.71 |
| 55 | `lora_twolevel` |   63.3 | 8094.0 |  0.00% |   276 |  1.48 |
| 56 | `lora_prefixtree:tinylfu` |   63.2 | 8077.8 |  0.00% |   276 |  1.41 |
| 57 | `lora_freqweighted:tinylfu` |   63.1 | 8067.1 |  0.00% |   309 |  1.46 |
| 58 | `lora_correlated:tinylfu` |   63.1 | 8066.4 |  0.00% |   276 |  1.48 |
| 59 | `lora_freqweighted:s3fifo` |   63.0 | 8051.9 |  0.00% |   309 |  1.62 |
| 60 | `lora_prefixtree:lru_k` |   62.7 | 8023.6 |  0.00% |   276 |  1.45 |
| 61 | `lru` |   62.7 | 8019.7 |  0.00% |   276 |  1.52 |
| 62 | `lora_loose:lru_k` |   62.6 | 8006.8 |  0.00% |   276 |  1.67 |
| 63 | `lora_budget:s3fifo` |   62.5 | 7998.7 |  0.00% |   276 |  1.47 |
| 64 | `lora_loose:tinylfu` |   62.5 | 7997.4 |  0.00% |   276 |  1.55 |
| 65 | `lora_costaware:lru_k` |   60.6 | 7751.6 |  0.00% |   276 |  1.68 |
| 66 | `lora_soft:s3fifo` |   60.3 | 7716.6 |  0.00% |   276 |  1.57 |

### LoRA coupling uplift over bare base — `adapter_locality`

Positive `Δ_thr` = LoRA decorator beats the bare inner base on output throughput. `n/a` means base run was missing/failed.

| Coupling \ Base | lru | lru_k | s3fifo | sieve | tinylfu |
|---|---|---|---|---|---|
| `lora_adabudget` | +22.6 (+36.0%) | +0.9 (+1.4%) | -24.2 (-27.1%) | +3.8 (+4.5%) | -1.8 (-2.7%) |
| `lora_budget` | +23.9 (+38.2%) | +0.5 (+0.8%) | -26.6 (-29.8%) | +4.1 (+4.8%) | -0.8 (-1.1%) |
| `lora_correlated` | +25.4 (+40.5%) | -0.7 (-1.0%) | -23.5 (-26.3%) | +2.8 (+3.3%) | -3.4 (-5.1%) |
| `lora_costaware` | +25.9 (+41.3%) | -4.7 (-7.1%) | -24.3 (-27.3%) | -0.2 (-0.2%) | -1.2 (-1.8%) |
| `lora_freqweighted` | +1.1 (+1.8%) | -1.5 (-2.3%) | -26.2 (-29.4%) | -19.2 (-22.7%) | -3.4 (-5.1%) |
| `lora_ghost` | +21.4 (+34.1%) | +1.0 (+1.6%) | -24.2 (-27.2%) | +6.0 (+7.0%) | -2.1 (-3.2%) |
| `lora_hysteresis` | +2.2 (+3.5%) | -0.0 (-0.1%) | -0.1 (-0.1%) | -19.6 (-23.1%) | +19.4 (+29.1%) |
| `lora_loose` | +19.4 (+30.9%) | -2.7 (-4.1%) | -4.2 (-4.7%) | -1.1 (-1.3%) | -3.9 (-5.9%) |
| `lora_position` | +1.5 (+2.4%) | -0.1 (-0.1%) | -24.4 (-27.3%) | +2.7 (+3.2%) | -1.8 (-2.7%) |
| `lora_prefixtree` | +1.5 (+2.4%) | -2.5 (-3.9%) | -3.3 (-3.7%) | +1.9 (+2.2%) | -3.3 (-5.0%) |
| `lora_soft` | +2.5 (+4.0%) | +16.8 (+25.7%) | -28.8 (-32.3%) | -19.6 (-23.2%) | +18.9 (+28.4%) |
| `lora_tight` | +24.6 (+39.3%) | +23.4 (+35.8%) | -23.8 (-26.7%) | -19.9 (-23.5%) | -1.1 (-1.6%) |

## Scenario: `mixed_popularity`

_Zipfian adapter mix: a few hot adapters dominate. Tests interaction of adapter popularity with block recency._

| Rank | Policy | OutTok/s | TotTok/s | HitRate | Evicted | P95 (s) |
|-----:|:-------|---------:|---------:|--------:|--------:|--------:|
| 1 | `lru` |   33.5 | 4289.9 |  5.43% |   575 |  3.48 |
| 2 | `lru_k` |   33.5 | 4286.7 |  5.43% |   575 |  3.47 |
| 3 | `tinylfu` |   33.0 | 4220.1 |  7.23% |   530 |  3.73 |
| 4 | `lora_costaware:s3fifo` |   32.9 | 4211.4 |  5.43% |   575 |  3.54 |
| 5 | `lora_loose:tinylfu` |   32.8 | 4198.5 |  7.23% |   554 |  3.58 |
| 6 | `lora_tight:tinylfu` |   32.8 | 4197.2 |  2.72% |   659 |  3.56 |
| 7 | `lora_twolevel` |   32.8 | 4193.5 |  2.72% |   610 |  3.56 |
| 8 | `lora_position:tinylfu` |   32.7 | 4183.1 |  7.23% |   530 |  3.51 |
| 9 | `lora_tight:s3fifo` |   32.7 | 4178.4 |  2.72% |   659 |  3.45 |
| 10 | `lora_hysteresis:lru_k` |   32.6 | 4172.2 |  2.72% |   659 |  3.12 |
| 11 | `lora_freqweighted:sieve` |   32.6 | 4167.8 |  2.72% |   659 |  3.38 |
| 12 | `lora_soft:s3fifo` |   32.6 | 4163.9 |  2.72% |   625 |  3.49 |
| 13 | `lora_correlated:sieve` |   32.4 | 4145.3 |  5.43% |   625 |  2.95 |
| 14 | `lora_freqweighted:lru_k` |   32.4 | 4142.7 |  2.72% |   659 |  3.37 |
| 15 | `lora_tight:sieve` |   32.4 | 4141.6 |  2.72% |   659 |  3.45 |
| 16 | `lora_loose:lru_k` |   32.4 | 4139.7 |  5.43% |   575 |  3.71 |
| 17 | `lora_soft:lru` |   32.3 | 4132.6 |  4.24% |   597 |  3.64 |
| 18 | `lora_prefixtree:tinylfu` |   32.3 | 4130.0 |  7.23% |   530 |  3.41 |
| 19 | `lora_ghost:s3fifo` |   32.2 | 4120.6 |  2.72% |   659 |  3.42 |
| 20 | `lora_costaware:tinylfu` |   32.1 | 4110.2 |  7.23% |   530 |  3.73 |
| 21 | `lora_budget:tinylfu` |   32.1 | 4110.0 |  8.15% |   525 |  3.39 |
| 22 | `lora_position:lru` |   32.1 | 4106.1 |  5.43% |   575 |  3.52 |
| 23 | `lora_ghost:tinylfu` |   32.0 | 4098.1 |  2.72% |   659 |  3.55 |
| 24 | `lora_soft:sieve` |   32.0 | 4093.0 |  5.43% |   625 |  3.49 |
| 25 | `lora_freqweighted:lru` |   32.0 | 4090.6 |  2.72% |   659 |  3.01 |
| 26 | `lora_prefixtree:s3fifo` |   31.9 | 4086.4 |  5.43% |   575 |  2.95 |
| 27 | `lora_adabudget:sieve` |   31.9 | 4074.3 |  5.43% |   625 |  2.93 |
| 28 | `lora_budget:s3fifo` |   31.8 | 4069.3 |  8.15% |   525 |  3.40 |
| 29 | `lora_budget:lru_k` |   31.8 | 4070.1 |  8.15% |   525 |  3.53 |
| 30 | `lora_adabudget:lru_k` |   31.8 | 4069.0 |  5.43% |   575 |  3.39 |
| 31 | `lora_adabudget:tinylfu` |   31.6 | 4043.1 |  8.15% |   539 |  3.46 |
| 32 | `lora_correlated:s3fifo` |   31.6 | 4040.1 |  2.72% |   614 |  3.48 |
| 33 | `lora_prefixtree:lru_k` |   31.6 | 4035.7 |  5.43% |   575 |  3.45 |
| 34 | `lora_position:s3fifo` |   31.4 | 4014.2 |  5.43% |   575 |  3.41 |
| 35 | `lora_ghost:lru_k` |   31.3 | 4005.5 |  2.72% |   659 |  3.48 |
| 36 | `lora_costaware:lru_k` |   31.1 | 3983.5 |  5.43% |   575 |  3.58 |
| 37 | `lora_adabudget:s3fifo` |   31.1 | 3978.3 |  2.72% |   621 |  3.47 |
| 38 | `lora_budget:sieve` |   31.1 | 3977.3 |  8.15% |   575 |  2.93 |
| 39 | `lora_freqweighted:tinylfu` |   31.0 | 3969.6 |  2.72% |   659 |  3.64 |
| 40 | `lora_correlated:tinylfu` |   31.0 | 3967.4 |  7.77% |   530 |  3.47 |
| 41 | `lora_costaware:sieve` |   31.0 | 3961.0 |  5.43% |   625 |  3.19 |
| 42 | `lora_hysteresis:sieve` |   30.9 | 3948.4 |  2.72% |   659 |  3.50 |
| 43 | `lora_correlated:lru_k` |   30.9 | 3947.9 |  5.43% |   575 |  3.55 |
| 44 | `lora_freqweighted:s3fifo` |   30.9 | 3946.9 |  2.72% |   659 |  3.97 |
| 45 | `lora_prefixtree:lru` |   30.6 | 3913.0 |  5.43% |   575 |  3.77 |
| 46 | `lora_position:lru_k` |   30.4 | 3882.2 |  5.43% |   575 |  3.90 |
| 47 | `lora_hysteresis:lru` |   29.9 | 3820.8 |  2.72% |   659 |  3.75 |
| 48 | `lora_tight:lru` |   29.6 | 3786.3 |  2.72% |   659 |  3.40 |
| 49 | `s3fifo` |   29.6 | 3780.8 |  5.43% |   575 |  3.55 |
| 50 | `sieve` |   29.5 | 3770.8 |  5.43% |   625 |  3.41 |
| 51 | `lora_prefixtree:sieve` |   29.1 | 3723.2 |  5.43% |   625 |  3.46 |
| 52 | `lora_loose:s3fifo` |   29.1 | 3716.1 |  2.72% |   595 |  3.67 |
| 53 | `lora_soft:tinylfu` |   29.1 | 3715.4 |  8.15% |   525 |  3.64 |
| 54 | `lora_budget:lru` |   28.8 | 3681.9 |  8.15% |   525 |  3.51 |
| 55 | `lora_ghost:lru` |   28.7 | 3675.5 |  2.72% |   659 |  3.52 |
| 56 | `lora_tight:lru_k` |   28.7 | 3672.9 |  2.72% |   659 |  3.80 |
| 57 | `lora_adabudget:lru` |   28.7 | 3666.6 |  5.43% |   575 |  3.43 |
| 58 | `lora_hysteresis:tinylfu` |   28.6 | 3652.8 |  0.00% |   709 |  3.42 |
| 59 | `lora_position:sieve` |   28.6 | 3650.9 |  5.43% |   625 |  3.75 |
| 60 | `lora_correlated:lru` |   28.2 | 3613.5 |  5.43% |   575 |  3.50 |
| 61 | `lora_hysteresis:s3fifo` |   27.9 | 3564.3 |  2.72% |   659 |  3.53 |
| 62 | `lora_soft:lru_k` |   27.8 | 3551.4 |  2.72% |   597 |  3.56 |
| 63 | `lora_loose:sieve` |   27.3 | 3488.6 |  5.43% |   625 |  3.58 |
| 64 | `lora_loose:lru` |   27.3 | 3486.8 |  5.43% |   575 |  3.69 |
| 65 | `lora_costaware:lru` |   27.0 | 3454.2 |  5.43% |   575 |  3.53 |
| 66 | `lora_ghost:sieve` |   26.5 | 3393.0 |  2.72% |   659 |  3.58 |

### LoRA coupling uplift over bare base — `mixed_popularity`

Positive `Δ_thr` = LoRA decorator beats the bare inner base on output throughput. `n/a` means base run was missing/failed.

| Coupling \ Base | lru | lru_k | s3fifo | sieve | tinylfu |
|---|---|---|---|---|---|
| `lora_adabudget` | -4.9 (-14.5%) | -1.7 (-5.1%) | +1.6 (+5.2%) | +2.4 (+8.1%) | -1.4 (-4.2%) |
| `lora_budget` | -4.8 (-14.2%) | -1.7 (-5.1%) | +2.3 (+7.6%) | +1.6 (+5.5%) | -0.9 (-2.6%) |
| `lora_correlated` | -5.3 (-15.8%) | -2.7 (-7.9%) | +2.0 (+6.9%) | +2.9 (+9.9%) | -2.0 (-6.0%) |
| `lora_costaware` | -6.5 (-19.5%) | -2.4 (-7.1%) | +3.4 (+11.4%) | +1.5 (+5.1%) | -0.9 (-2.6%) |
| `lora_freqweighted` | -1.6 (-4.7%) | -1.1 (-3.4%) | +1.3 (+4.4%) | +3.1 (+10.5%) | -2.0 (-5.9%) |
| `lora_ghost` | -4.8 (-14.3%) | -2.2 (-6.6%) | +2.7 (+9.0%) | -2.9 (-10.0%) | -1.0 (-2.9%) |
| `lora_hysteresis` | -3.7 (-10.9%) | -0.9 (-2.7%) | -1.7 (-5.7%) | +1.4 (+4.7%) | -4.4 (-13.5%) |
| `lora_loose` | -6.3 (-18.7%) | -1.2 (-3.4%) | -0.5 (-1.7%) | -2.2 (-7.5%) | -0.2 (-0.5%) |
| `lora_position` | -1.4 (-4.3%) | -3.2 (-9.5%) | +1.8 (+6.2%) | -0.9 (-3.2%) | -0.3 (-0.9%) |
| `lora_prefixtree` | -2.9 (-8.8%) | -2.0 (-5.9%) | +2.4 (+8.1%) | -0.4 (-1.3%) | -0.7 (-2.2%) |
| `lora_soft` | -1.2 (-3.7%) | -5.8 (-17.2%) | +3.0 (+10.1%) | +2.5 (+8.5%) | -3.9 (-12.0%) |
| `lora_tight` | -3.9 (-11.7%) | -4.8 (-14.3%) | +3.1 (+10.5%) | +2.9 (+9.8%) | -0.2 (-0.5%) |

## Overall ranking (avg rank by output throughput)

| Rank | Policy | Avg Rank | adapter_thrashing rank | adapter_locality rank | mixed_popularity rank |
|-----:|:-------|---------:|:---:|:---:|:---:|
| 1 | `lru_k` | 12.7 | 2 | 34 | 2 |
| 2 | `lora_adabudget:sieve` | 16.3 | 15 | 7 | 27 |
| 3 | `lora_tight:tinylfu` | 16.7 | 12 | 32 | 6 |
| 4 | `lora_correlated:sieve` | 16.7 | 28 | 9 | 13 |
| 5 | `lora_costaware:s3fifo` | 17.0 | 3 | 44 | 4 |
| 6 | `s3fifo` | 20.0 | 9 | 2 | 49 |
| 7 | `tinylfu` | 20.7 | 34 | 25 | 3 |
| 8 | `lru` | 21.0 | 1 | 61 | 1 |
| 9 | `lora_tight:lru_k` | 21.7 | 4 | 5 | 56 |
| 10 | `lora_budget:sieve` | 21.7 | 23 | 4 | 38 |
| 11 | `lora_twolevel` | 23.0 | 7 | 55 | 7 |
| 12 | `lora_correlated:lru` | 24.3 | 5 | 8 | 60 |
| 13 | `lora_prefixtree:s3fifo` | 24.3 | 33 | 14 | 26 |
| 14 | `lora_loose:s3fifo` | 25.3 | 6 | 18 | 52 |
| 15 | `lora_tight:lru` | 25.3 | 18 | 10 | 48 |
| 16 | `lora_position:sieve` | 26.7 | 10 | 11 | 59 |
| 17 | `lora_soft:lru` | 26.7 | 27 | 36 | 17 |
| 18 | `lora_tight:sieve` | 27.0 | 20 | 46 | 15 |
| 19 | `lora_adabudget:lru_k` | 27.3 | 25 | 27 | 30 |
| 20 | `lora_loose:tinylfu` | 27.7 | 14 | 64 | 5 |
| 21 | `lora_position:s3fifo` | 29.0 | 8 | 45 | 34 |
| 22 | `lora_hysteresis:s3fifo` | 29.3 | 24 | 3 | 61 |
| 23 | `lora_costaware:lru` | 30.0 | 19 | 6 | 65 |
| 24 | `lora_freqweighted:sieve` | 31.0 | 51 | 31 | 11 |
| 25 | `lora_adabudget:s3fifo` | 31.3 | 16 | 41 | 37 |
| 26 | `lora_prefixtree:tinylfu` | 31.7 | 21 | 56 | 18 |
| 27 | `lora_correlated:s3fifo` | 32.3 | 35 | 30 | 32 |
| 28 | `lora_tight:s3fifo` | 32.7 | 56 | 33 | 9 |
| 29 | `lora_loose:lru_k` | 33.3 | 22 | 62 | 16 |
| 30 | `lora_costaware:tinylfu` | 33.3 | 45 | 35 | 20 |
| 31 | `lora_budget:tinylfu` | 34.0 | 52 | 29 | 21 |
| 32 | `lora_ghost:sieve` | 34.3 | 36 | 1 | 66 |
| 33 | `lora_position:lru` | 34.7 | 31 | 51 | 22 |
| 34 | `lora_ghost:s3fifo` | 35.7 | 46 | 42 | 19 |
| 35 | `lora_prefixtree:lru` | 36.0 | 11 | 52 | 45 |
| 36 | `lora_costaware:sieve` | 36.0 | 47 | 20 | 41 |
| 37 | `lora_hysteresis:lru_k` | 36.0 | 61 | 37 | 10 |
| 38 | `lora_budget:lru` | 36.3 | 43 | 12 | 54 |
| 39 | `lora_soft:s3fifo` | 36.7 | 32 | 66 | 12 |
| 40 | `lora_correlated:tinylfu` | 37.0 | 13 | 58 | 40 |
| 41 | `lora_adabudget:lru` | 37.0 | 37 | 17 | 57 |
| 42 | `lora_position:lru_k` | 37.7 | 29 | 38 | 46 |
| 43 | `lora_ghost:tinylfu` | 37.7 | 40 | 50 | 23 |
| 44 | `lora_soft:sieve` | 38.0 | 50 | 40 | 24 |
| 45 | `lora_position:tinylfu` | 39.0 | 62 | 47 | 8 |
| 46 | `lora_costaware:lru_k` | 39.3 | 17 | 65 | 36 |
| 47 | `sieve` | 39.3 | 49 | 19 | 50 |
| 48 | `lora_prefixtree:lru_k` | 39.7 | 26 | 60 | 33 |
| 49 | `lora_hysteresis:sieve` | 39.7 | 38 | 39 | 42 |
| 50 | `lora_ghost:lru_k` | 39.7 | 58 | 26 | 35 |
| 51 | `lora_freqweighted:lru` | 40.0 | 42 | 53 | 25 |
| 52 | `lora_budget:s3fifo` | 40.3 | 30 | 63 | 28 |
| 53 | `lora_budget:lru_k` | 41.0 | 66 | 28 | 29 |
| 54 | `lora_loose:sieve` | 41.3 | 39 | 22 | 63 |
| 55 | `lora_soft:tinylfu` | 42.0 | 57 | 16 | 53 |
| 56 | `lora_soft:lru_k` | 42.3 | 41 | 24 | 62 |
| 57 | `lora_freqweighted:lru_k` | 42.3 | 59 | 54 | 14 |
| 58 | `lora_hysteresis:tinylfu` | 42.7 | 55 | 15 | 58 |
| 59 | `lora_prefixtree:sieve` | 42.7 | 64 | 13 | 51 |
| 60 | `lora_adabudget:tinylfu` | 44.0 | 53 | 48 | 31 |
| 61 | `lora_hysteresis:lru` | 44.7 | 44 | 43 | 47 |
| 62 | `lora_ghost:lru` | 46.3 | 63 | 21 | 55 |
| 63 | `lora_freqweighted:tinylfu` | 48.0 | 48 | 57 | 39 |
| 64 | `lora_correlated:lru_k` | 50.7 | 60 | 49 | 43 |
| 65 | `lora_loose:lru` | 50.7 | 65 | 23 | 64 |
| 66 | `lora_freqweighted:s3fifo` | 52.3 | 54 | 59 | 44 |

## Recommendations

- **adapter_thrashing**: best is `lru` (out_thr=23.8 tok/s, hit_rate=49.86%).
- **adapter_locality**: best is `lora_ghost:sieve` (out_thr=90.6 tok/s, hit_rate=0.00%).
- **mixed_popularity**: best is `lru` (out_thr=33.5 tok/s, hit_rate=5.43%).


[wrote /tmp/lora_sweep_logs/REPORT.md]
