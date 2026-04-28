# LoRA-Aware Cache Policy Sweep — Results

Total runs in results: **198** (198 ok, 0 timed out, 0 failed).

## Scenario: `adapter_thrashing`

_Round-robin across all adapters: every request hits a different adapter than the previous, so blocks rarely repeat. Tests how well the policy avoids holding dead-adapter blocks._

| Rank | Policy | OutTok/s | TotTok/s | HitRate | Evicted | P95 (s) |
|-----:|:-------|---------:|---------:|--------:|--------:|--------:|
| 1 | `s3fifo` |   23.1 | 2957.4 | 52.15% |   678 |  4.27 |
| 2 | `lora_correlated:s3fifo` |   22.4 | 2869.7 | 50.50% |   678 |  4.25 |
| 3 | `lru` |   23.8 | 3050.2 | 49.86% |   678 |  4.30 |
| 4 | `lru_k` |   23.5 | 3003.8 | 49.86% |   678 |  4.33 |
| 5 | `lora_loose:s3fifo` |   23.3 | 2981.4 | 49.86% |   678 |  3.74 |
| 6 | `lora_position:s3fifo` |   23.2 | 2964.1 | 49.86% |   678 |  4.36 |
| 7 | `lora_adabudget:s3fifo` |   22.9 | 2934.0 | 49.86% |   678 |  4.31 |
| 8 | `lora_costaware:lru_k` |   22.9 | 2935.1 | 49.86% |   678 |  4.26 |
| 9 | `lora_loose:lru_k` |   22.7 | 2904.6 | 49.86% |   678 |  4.41 |
| 10 | `lora_prefixtree:lru_k` |   22.6 | 2892.1 | 49.86% |   678 |  4.42 |
| 11 | `lora_position:lru_k` |   22.6 | 2891.7 | 49.86% |   678 |  4.18 |
| 12 | `lora_budget:s3fifo` |   22.6 | 2888.3 | 49.86% |   678 |  4.59 |
| 13 | `lora_position:lru` |   22.6 | 2885.2 | 49.86% |   678 |  4.67 |
| 14 | `lora_prefixtree:s3fifo` |   22.5 | 2877.7 | 49.86% |   678 |  4.56 |
| 15 | `lora_budget:lru` |   22.1 | 2831.4 | 49.86% |   678 |  4.32 |
| 16 | `lora_correlated:lru_k` |   21.2 | 2709.4 | 49.86% |   678 |  4.35 |
| 17 | `lora_loose:lru` |   20.9 | 2666.7 | 49.86% |   678 |  5.12 |
| 18 | `lora_budget:lru_k` |   20.6 | 2640.3 | 49.86% |   678 |  4.51 |
| 19 | `lora_correlated:lru` |   23.4 | 2989.6 | 49.70% |   684 |  4.15 |
| 20 | `lora_prefixtree:lru` |   23.0 | 2945.9 | 49.70% |   678 |  4.16 |
| 21 | `lora_soft:s3fifo` |   22.5 | 2883.9 | 49.70% |   678 |  4.24 |
| 22 | `lora_costaware:s3fifo` |   23.4 | 2997.8 | 49.53% |   678 |  4.27 |
| 23 | `lora_costaware:lru` |   22.7 | 2909.4 | 49.52% |   678 |  3.97 |
| 24 | `lora_soft:lru_k` |   22.3 | 2850.6 | 49.52% |   678 |  4.41 |
| 25 | `lora_adabudget:lru` |   22.4 | 2867.0 | 49.17% |   678 |  4.81 |
| 26 | `lora_adabudget:lru_k` |   22.6 | 2893.3 | 48.87% |   678 |  4.30 |
| 27 | `lora_soft:lru` |   22.6 | 2891.6 | 44.31% |   873 |  4.37 |
| 28 | `lora_budget:sieve` |   22.7 | 2903.5 | 43.96% |   889 |  4.31 |
| 29 | `lora_soft:tinylfu` |   21.5 | 2749.7 | 41.83% |   986 |  4.24 |
| 30 | `lora_position:sieve` |   23.1 | 2955.3 | 38.44% |  1658 |  3.78 |
| 31 | `lora_loose:sieve` |   22.3 | 2856.2 | 37.65% |  1476 |  3.88 |
| 32 | `sieve` |   22.0 | 2815.2 | 36.92% |  1658 |  4.15 |
| 33 | `lora_prefixtree:sieve` |   20.9 | 2669.5 | 36.92% |  1658 |  4.43 |
| 34 | `lora_correlated:sieve` |   22.6 | 2890.5 | 36.70% |  1435 |  4.32 |
| 35 | `lora_adabudget:sieve` |   22.9 | 2935.4 | 36.32% |  1476 |  4.54 |
| 36 | `lora_costaware:sieve` |   22.1 | 2822.5 | 36.31% |  1658 |  4.67 |
| 37 | `lora_soft:sieve` |   21.9 | 2808.3 | 35.13% |  1393 |  4.27 |
| 38 | `lora_tight:s3fifo` |   21.7 | 2777.7 | 33.53% |  1631 |  4.76 |
| 39 | `lora_tight:lru_k` |   23.4 | 2992.2 | 33.00% |  1631 |  4.28 |
| 40 | `lora_tight:tinylfu` |   23.0 | 2942.3 | 33.00% |  1631 |  4.20 |
| 41 | `lora_tight:lru` |   22.8 | 2914.6 | 33.00% |  1631 |  3.73 |
| 42 | `lora_tight:sieve` |   22.7 | 2908.0 | 33.00% |  1631 |  4.32 |
| 43 | `lora_ghost:sieve` |   22.4 | 2868.4 | 33.00% |  1631 |  3.77 |
| 44 | `lora_ghost:s3fifo` |   22.1 | 2824.6 | 33.00% |  1631 |  4.33 |
| 45 | `lora_ghost:lru_k` |   21.5 | 2746.9 | 33.00% |  1631 |  4.54 |
| 46 | `lora_ghost:lru` |   20.9 | 2674.6 | 33.00% |  1631 |  4.41 |
| 47 | `lora_ghost:tinylfu` |   22.3 | 2854.4 | 32.57% |  1631 |  4.55 |
| 48 | `lora_loose:tinylfu` |   22.9 | 2935.7 | 32.42% |  1293 |  4.28 |
| 49 | `lora_correlated:tinylfu` |   23.0 | 2940.0 | 31.29% |  1476 |  4.21 |
| 50 | `lora_adabudget:tinylfu` |   21.8 | 2788.7 | 31.25% |  1295 |  4.65 |
| 51 | `lora_prefixtree:tinylfu` |   22.7 | 2906.8 | 29.99% |  1287 |  4.19 |
| 52 | `lora_position:tinylfu` |   21.0 | 2688.1 | 29.37% |  1340 |  4.54 |
| 53 | `lora_budget:tinylfu` |   21.8 | 2793.4 | 29.27% |  1007 |  4.19 |
| 54 | `tinylfu` |   22.5 | 2875.1 | 29.11% |  1614 |  4.23 |
| 55 | `lora_costaware:tinylfu` |   22.1 | 2825.8 | 27.84% |  1529 |  4.34 |
| 56 | `lora_hysteresis:lru` |   22.1 | 2829.1 | 26.12% |  1972 |  4.39 |
| 57 | `lora_hysteresis:s3fifo` |   22.6 | 2895.3 | 23.96% |  1820 |  4.50 |
| 58 | `lora_hysteresis:sieve` |   22.4 | 2862.8 | 23.96% |  1820 |  4.18 |
| 59 | `lora_hysteresis:tinylfu` |   21.7 | 2778.6 | 23.96% |  1820 |  4.14 |
| 60 | `lora_hysteresis:lru_k` |   21.1 | 2699.1 | 23.96% |  1820 |  4.63 |
| 61 | `lora_freqweighted:tinylfu` |   22.0 | 2818.6 | 22.16% |  2224 |  4.25 |
| 62 | `lora_twolevel` |   23.2 | 2971.0 | 19.45% |  1877 |  4.26 |
| 63 | `lora_freqweighted:lru` |   22.2 | 2844.8 | 18.23% |  2187 |  4.51 |
| 64 | `lora_freqweighted:sieve` |   21.9 | 2806.8 | 18.23% |  2187 |  4.30 |
| 65 | `lora_freqweighted:s3fifo` |   21.8 | 2785.2 | 18.23% |  2187 |  4.28 |
| 66 | `lora_freqweighted:lru_k` |   21.2 | 2709.7 | 18.23% |  2187 |  4.49 |

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
| 1 | `lora_budget:tinylfu` |   32.1 | 4110.0 |  8.15% |   525 |  3.39 |
| 2 | `lora_budget:s3fifo` |   31.8 | 4069.3 |  8.15% |   525 |  3.40 |
| 3 | `lora_budget:lru_k` |   31.8 | 4070.1 |  8.15% |   525 |  3.53 |
| 4 | `lora_adabudget:tinylfu` |   31.6 | 4043.1 |  8.15% |   539 |  3.46 |
| 5 | `lora_budget:sieve` |   31.1 | 3977.3 |  8.15% |   575 |  2.93 |
| 6 | `lora_soft:tinylfu` |   29.1 | 3715.4 |  8.15% |   525 |  3.64 |
| 7 | `lora_budget:lru` |   28.8 | 3681.9 |  8.15% |   525 |  3.51 |
| 8 | `lora_correlated:tinylfu` |   31.0 | 3967.4 |  7.77% |   530 |  3.47 |
| 9 | `tinylfu` |   33.0 | 4220.1 |  7.23% |   530 |  3.73 |
| 10 | `lora_loose:tinylfu` |   32.8 | 4198.5 |  7.23% |   554 |  3.58 |
| 11 | `lora_position:tinylfu` |   32.7 | 4183.1 |  7.23% |   530 |  3.51 |
| 12 | `lora_prefixtree:tinylfu` |   32.3 | 4130.0 |  7.23% |   530 |  3.41 |
| 13 | `lora_costaware:tinylfu` |   32.1 | 4110.2 |  7.23% |   530 |  3.73 |
| 14 | `lru` |   33.5 | 4289.9 |  5.43% |   575 |  3.48 |
| 15 | `lru_k` |   33.5 | 4286.7 |  5.43% |   575 |  3.47 |
| 16 | `lora_costaware:s3fifo` |   32.9 | 4211.4 |  5.43% |   575 |  3.54 |
| 17 | `lora_correlated:sieve` |   32.4 | 4145.3 |  5.43% |   625 |  2.95 |
| 18 | `lora_loose:lru_k` |   32.4 | 4139.7 |  5.43% |   575 |  3.71 |
| 19 | `lora_position:lru` |   32.1 | 4106.1 |  5.43% |   575 |  3.52 |
| 20 | `lora_soft:sieve` |   32.0 | 4093.0 |  5.43% |   625 |  3.49 |
| 21 | `lora_prefixtree:s3fifo` |   31.9 | 4086.4 |  5.43% |   575 |  2.95 |
| 22 | `lora_adabudget:sieve` |   31.9 | 4074.3 |  5.43% |   625 |  2.93 |
| 23 | `lora_adabudget:lru_k` |   31.8 | 4069.0 |  5.43% |   575 |  3.39 |
| 24 | `lora_prefixtree:lru_k` |   31.6 | 4035.7 |  5.43% |   575 |  3.45 |
| 25 | `lora_position:s3fifo` |   31.4 | 4014.2 |  5.43% |   575 |  3.41 |
| 26 | `lora_costaware:lru_k` |   31.1 | 3983.5 |  5.43% |   575 |  3.58 |
| 27 | `lora_costaware:sieve` |   31.0 | 3961.0 |  5.43% |   625 |  3.19 |
| 28 | `lora_correlated:lru_k` |   30.9 | 3947.9 |  5.43% |   575 |  3.55 |
| 29 | `lora_prefixtree:lru` |   30.6 | 3913.0 |  5.43% |   575 |  3.77 |
| 30 | `lora_position:lru_k` |   30.4 | 3882.2 |  5.43% |   575 |  3.90 |
| 31 | `s3fifo` |   29.6 | 3780.8 |  5.43% |   575 |  3.55 |
| 32 | `sieve` |   29.5 | 3770.8 |  5.43% |   625 |  3.41 |
| 33 | `lora_prefixtree:sieve` |   29.1 | 3723.2 |  5.43% |   625 |  3.46 |
| 34 | `lora_adabudget:lru` |   28.7 | 3666.6 |  5.43% |   575 |  3.43 |
| 35 | `lora_position:sieve` |   28.6 | 3650.9 |  5.43% |   625 |  3.75 |
| 36 | `lora_correlated:lru` |   28.2 | 3613.5 |  5.43% |   575 |  3.50 |
| 37 | `lora_loose:sieve` |   27.3 | 3488.6 |  5.43% |   625 |  3.58 |
| 38 | `lora_loose:lru` |   27.3 | 3486.8 |  5.43% |   575 |  3.69 |
| 39 | `lora_costaware:lru` |   27.0 | 3454.2 |  5.43% |   575 |  3.53 |
| 40 | `lora_soft:lru` |   32.3 | 4132.6 |  4.24% |   597 |  3.64 |
| 41 | `lora_tight:tinylfu` |   32.8 | 4197.2 |  2.72% |   659 |  3.56 |
| 42 | `lora_twolevel` |   32.8 | 4193.5 |  2.72% |   610 |  3.56 |
| 43 | `lora_tight:s3fifo` |   32.7 | 4178.4 |  2.72% |   659 |  3.45 |
| 44 | `lora_hysteresis:lru_k` |   32.6 | 4172.2 |  2.72% |   659 |  3.12 |
| 45 | `lora_freqweighted:sieve` |   32.6 | 4167.8 |  2.72% |   659 |  3.38 |
| 46 | `lora_soft:s3fifo` |   32.6 | 4163.9 |  2.72% |   625 |  3.49 |
| 47 | `lora_freqweighted:lru_k` |   32.4 | 4142.7 |  2.72% |   659 |  3.37 |
| 48 | `lora_tight:sieve` |   32.4 | 4141.6 |  2.72% |   659 |  3.45 |
| 49 | `lora_ghost:s3fifo` |   32.2 | 4120.6 |  2.72% |   659 |  3.42 |
| 50 | `lora_ghost:tinylfu` |   32.0 | 4098.1 |  2.72% |   659 |  3.55 |
| 51 | `lora_freqweighted:lru` |   32.0 | 4090.6 |  2.72% |   659 |  3.01 |
| 52 | `lora_correlated:s3fifo` |   31.6 | 4040.1 |  2.72% |   614 |  3.48 |
| 53 | `lora_ghost:lru_k` |   31.3 | 4005.5 |  2.72% |   659 |  3.48 |
| 54 | `lora_adabudget:s3fifo` |   31.1 | 3978.3 |  2.72% |   621 |  3.47 |
| 55 | `lora_freqweighted:tinylfu` |   31.0 | 3969.6 |  2.72% |   659 |  3.64 |
| 56 | `lora_hysteresis:sieve` |   30.9 | 3948.4 |  2.72% |   659 |  3.50 |
| 57 | `lora_freqweighted:s3fifo` |   30.9 | 3946.9 |  2.72% |   659 |  3.97 |
| 58 | `lora_hysteresis:lru` |   29.9 | 3820.8 |  2.72% |   659 |  3.75 |
| 59 | `lora_tight:lru` |   29.6 | 3786.3 |  2.72% |   659 |  3.40 |
| 60 | `lora_loose:s3fifo` |   29.1 | 3716.1 |  2.72% |   595 |  3.67 |
| 61 | `lora_ghost:lru` |   28.7 | 3675.5 |  2.72% |   659 |  3.52 |
| 62 | `lora_tight:lru_k` |   28.7 | 3672.9 |  2.72% |   659 |  3.80 |
| 63 | `lora_hysteresis:s3fifo` |   27.9 | 3564.3 |  2.72% |   659 |  3.53 |
| 64 | `lora_soft:lru_k` |   27.8 | 3551.4 |  2.72% |   597 |  3.56 |
| 65 | `lora_ghost:sieve` |   26.5 | 3393.0 |  2.72% |   659 |  3.58 |
| 66 | `lora_hysteresis:tinylfu` |   28.6 | 3652.8 |  0.00% |   709 |  3.42 |

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
| 1 | `s3fifo` | 11.3 | 1 | 2 | 31 |
| 2 | `lora_budget:lru` | 11.3 | 15 | 12 | 7 |
| 3 | `lora_budget:sieve` | 12.3 | 28 | 4 | 5 |
| 4 | `lora_prefixtree:s3fifo` | 16.3 | 14 | 14 | 21 |
| 5 | `lora_budget:lru_k` | 16.3 | 18 | 28 | 3 |
| 6 | `lora_soft:tinylfu` | 17.0 | 29 | 16 | 6 |
| 7 | `lru_k` | 17.7 | 4 | 34 | 15 |
| 8 | `lora_correlated:sieve` | 20.0 | 34 | 9 | 17 |
| 9 | `lora_correlated:lru` | 21.0 | 19 | 8 | 36 |
| 10 | `lora_adabudget:sieve` | 21.3 | 35 | 7 | 22 |
| 11 | `lora_costaware:lru` | 22.7 | 23 | 6 | 39 |
| 12 | `lora_position:s3fifo` | 25.3 | 6 | 45 | 25 |
| 13 | `lora_adabudget:lru` | 25.3 | 25 | 17 | 34 |
| 14 | `lora_adabudget:lru_k` | 25.3 | 26 | 27 | 23 |
| 15 | `lora_position:sieve` | 25.3 | 30 | 11 | 35 |
| 16 | `lora_budget:s3fifo` | 25.7 | 12 | 63 | 2 |
| 17 | `lru` | 26.0 | 3 | 61 | 14 |
| 18 | `lora_loose:lru` | 26.0 | 17 | 23 | 38 |
| 19 | `lora_position:lru_k` | 26.3 | 11 | 38 | 30 |
| 20 | `lora_prefixtree:sieve` | 26.3 | 33 | 13 | 33 |
| 21 | `lora_costaware:s3fifo` | 27.3 | 22 | 44 | 16 |
| 22 | `lora_loose:s3fifo` | 27.7 | 5 | 18 | 60 |
| 23 | `lora_position:lru` | 27.7 | 13 | 51 | 19 |
| 24 | `sieve` | 27.7 | 32 | 19 | 32 |
| 25 | `lora_costaware:sieve` | 27.7 | 36 | 20 | 27 |
| 26 | `lora_budget:tinylfu` | 27.7 | 53 | 29 | 1 |
| 27 | `lora_correlated:s3fifo` | 28.0 | 2 | 30 | 52 |
| 28 | `tinylfu` | 29.3 | 54 | 25 | 9 |
| 29 | `lora_loose:lru_k` | 29.7 | 9 | 62 | 18 |
| 30 | `lora_loose:sieve` | 30.0 | 31 | 22 | 37 |
| 31 | `lora_correlated:lru_k` | 31.0 | 16 | 49 | 28 |
| 32 | `lora_prefixtree:lru_k` | 31.3 | 10 | 60 | 24 |
| 33 | `lora_soft:sieve` | 32.3 | 37 | 40 | 20 |
| 34 | `lora_costaware:lru_k` | 33.0 | 8 | 65 | 26 |
| 35 | `lora_prefixtree:lru` | 33.7 | 20 | 52 | 29 |
| 36 | `lora_adabudget:s3fifo` | 34.0 | 7 | 41 | 54 |
| 37 | `lora_adabudget:tinylfu` | 34.0 | 50 | 48 | 4 |
| 38 | `lora_soft:lru` | 34.3 | 27 | 36 | 40 |
| 39 | `lora_costaware:tinylfu` | 34.3 | 55 | 35 | 13 |
| 40 | `lora_tight:lru_k` | 35.3 | 39 | 5 | 62 |
| 41 | `lora_ghost:sieve` | 36.3 | 43 | 1 | 65 |
| 42 | `lora_tight:lru` | 36.7 | 41 | 10 | 59 |
| 43 | `lora_position:tinylfu` | 36.7 | 52 | 47 | 11 |
| 44 | `lora_soft:lru_k` | 37.3 | 24 | 24 | 64 |
| 45 | `lora_tight:tinylfu` | 37.7 | 40 | 32 | 41 |
| 46 | `lora_tight:s3fifo` | 38.0 | 38 | 33 | 43 |
| 47 | `lora_correlated:tinylfu` | 38.3 | 49 | 58 | 8 |
| 48 | `lora_prefixtree:tinylfu` | 39.7 | 51 | 56 | 12 |
| 49 | `lora_loose:tinylfu` | 40.7 | 48 | 64 | 10 |
| 50 | `lora_hysteresis:s3fifo` | 41.0 | 57 | 3 | 63 |
| 51 | `lora_ghost:lru_k` | 41.3 | 45 | 26 | 53 |
| 52 | `lora_ghost:lru` | 42.7 | 46 | 21 | 61 |
| 53 | `lora_soft:s3fifo` | 44.3 | 21 | 66 | 46 |
| 54 | `lora_ghost:s3fifo` | 45.0 | 44 | 42 | 49 |
| 55 | `lora_tight:sieve` | 45.3 | 42 | 46 | 48 |
| 56 | `lora_hysteresis:tinylfu` | 46.7 | 59 | 15 | 66 |
| 57 | `lora_freqweighted:sieve` | 46.7 | 64 | 31 | 45 |
| 58 | `lora_hysteresis:lru_k` | 47.0 | 60 | 37 | 44 |
| 59 | `lora_ghost:tinylfu` | 49.0 | 47 | 50 | 50 |
| 60 | `lora_hysteresis:sieve` | 51.0 | 58 | 39 | 56 |
| 61 | `lora_hysteresis:lru` | 52.3 | 56 | 43 | 58 |
| 62 | `lora_twolevel` | 53.0 | 62 | 55 | 42 |
| 63 | `lora_freqweighted:lru` | 55.7 | 63 | 53 | 51 |
| 64 | `lora_freqweighted:lru_k` | 55.7 | 66 | 54 | 47 |
| 65 | `lora_freqweighted:tinylfu` | 57.7 | 61 | 57 | 55 |
| 66 | `lora_freqweighted:s3fifo` | 60.3 | 65 | 59 | 57 |

## Recommendations

- **adapter_thrashing**: best is `s3fifo` (out_thr=23.1 tok/s, hit_rate=52.15%).
- **adapter_locality**: best is `lora_ghost:sieve` (out_thr=90.6 tok/s, hit_rate=0.00%).
- **mixed_popularity**: best is `lora_budget:tinylfu` (out_thr=32.1 tok/s, hit_rate=8.15%).


[wrote /tmp/lora_sweep_logs/REPORT_byhit.md]
