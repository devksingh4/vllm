# LoRA-Aware Cache Policy TTFT Sweep — Results

Total runs: **198** (198 ok, 0 timed out, 0 failed).

Per-request TTFT was measured via `LLMEngine.add_request` + `engine.step()` loop, recording `now()` when each request first emitted a token.

## Scenario: `adapter_thrashing`

_Round-robin across 16 LoRA adapters (max_loras=2). Highest CPU-offload reuse pressure._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_loose:tinylfu` | 1.234s | 2.761s | 3.481s | 1.868s | 32.39% |
| 2 | `lru` | 1.246s | 2.717s | 3.468s | 1.863s | 49.52% |
| 3 | `lora_hysteresis:s3fifo` | 1.247s | 2.732s | 3.539s | 1.888s | 23.96% |
| 4 | `lora_ghost:lru_k` | 1.252s | 3.022s | 3.751s | 1.979s | 33.00% |
| 5 | `lora_adabudget:sieve` | 1.262s | 2.755s | 3.529s | 1.912s | 37.85% |
| 6 | `lora_soft:tinylfu` | 1.268s | 2.835s | 3.556s | 1.915s | 38.70% |
| 7 | `lora_hysteresis:tinylfu` | 1.274s | 2.686s | 3.452s | 1.945s | 23.96% |
| 8 | `lora_prefixtree:lru` | 1.278s | 2.785s | 3.727s | 1.956s | 49.34% |
| 9 | `lora_adabudget:lru_k` | 1.283s | 3.113s | 3.858s | 1.973s | 49.86% |
| 10 | `lora_prefixtree:sieve` | 1.287s | 2.776s | 3.534s | 1.923s | 36.92% |
| 11 | `lora_loose:lru` | 1.290s | 2.687s | 3.508s | 1.910s | 49.52% |
| 12 | `lora_tight:s3fifo` | 1.293s | 2.758s | 3.487s | 1.921s | 33.00% |
| 13 | `lora_adabudget:lru` | 1.293s | 2.793s | 3.536s | 1.961s | 49.86% |
| 14 | `lora_budget:lru_k` | 1.298s | 2.273s | 3.143s | 1.922s | 49.86% |
| 15 | `lora_soft:lru` | 1.305s | 2.850s | 3.727s | 1.999s | 42.85% |
| 16 | `lora_ghost:sieve` | 1.308s | 2.806s | 3.572s | 1.942s | 33.00% |
| 17 | `lora_tight:lru` | 1.315s | 2.958s | 3.731s | 1.992s | 33.00% |
| 18 | `lora_correlated:sieve` | 1.315s | 2.679s | 3.434s | 1.990s | 37.02% |
| 19 | `lora_ghost:s3fifo` | 1.323s | 2.569s | 3.175s | 2.078s | 33.00% |
| 20 | `lora_tight:tinylfu` | 1.329s | 2.936s | 3.661s | 2.027s | 33.00% |
| 21 | `lora_costaware:tinylfu` | 1.337s | 2.958s | 3.701s | 2.005s | 32.92% |
| 22 | `lora_correlated:lru_k` | 1.342s | 2.777s | 3.545s | 1.976s | 49.70% |
| 23 | `lora_soft:s3fifo` | 1.361s | 2.913s | 3.665s | 1.955s | 49.86% |
| 24 | `lora_adabudget:s3fifo` | 1.369s | 2.490s | 3.272s | 1.896s | 49.63% |
| 25 | `lora_freqweighted:lru_k` | 1.370s | 2.887s | 3.635s | 2.065s | 18.23% |
| 26 | `lora_soft:sieve` | 1.374s | 2.391s | 3.599s | 1.926s | 35.35% |
| 27 | `lora_hysteresis:sieve` | 1.381s | 2.794s | 3.510s | 1.968s | 23.96% |
| 28 | `lru_k` | 1.383s | 2.788s | 3.515s | 1.909s | 49.86% |
| 29 | `s3fifo` | 1.386s | 2.950s | 3.759s | 1.880s | 48.34% |
| 30 | `lora_loose:lru_k` | 1.398s | 2.770s | 3.551s | 1.913s | 49.86% |
| 31 | `lora_budget:lru` | 1.402s | 2.710s | 3.579s | 1.923s | 49.86% |
| 32 | `lora_correlated:lru` | 1.405s | 2.323s | 3.163s | 1.933s | 49.41% |
| 33 | `lora_budget:s3fifo` | 1.407s | 2.741s | 3.514s | 1.925s | 49.86% |
| 34 | `lora_position:lru_k` | 1.417s | 2.296s | 3.009s | 1.937s | 49.70% |
| 35 | `lora_costaware:lru` | 1.419s | 2.959s | 3.787s | 1.950s | 49.86% |
| 36 | `lora_budget:sieve` | 1.420s | 2.252s | 2.969s | 1.958s | 42.73% |
| 37 | `lora_tight:lru_k` | 1.423s | 2.358s | 2.961s | 1.983s | 36.09% |
| 38 | `lora_correlated:s3fifo` | 1.425s | 2.774s | 3.508s | 1.940s | 53.06% |
| 39 | `lora_prefixtree:s3fifo` | 1.437s | 2.465s | 3.520s | 1.995s | 49.86% |
| 40 | `lora_position:sieve` | 1.440s | 2.423s | 3.663s | 1.982s | 39.01% |
| 41 | `lora_prefixtree:lru_k` | 1.443s | 2.785s | 3.536s | 1.934s | 48.34% |
| 42 | `lora_costaware:lru_k` | 1.444s | 2.339s | 3.662s | 1.958s | 49.86% |
| 43 | `lora_position:lru` | 1.444s | 2.866s | 3.687s | 1.937s | 49.00% |
| 44 | `lora_loose:s3fifo` | 1.450s | 2.340s | 3.620s | 1.973s | 49.70% |
| 45 | `lora_loose:sieve` | 1.459s | 2.723s | 3.498s | 1.972s | 37.85% |
| 46 | `lora_freqweighted:s3fifo` | 1.462s | 2.525s | 3.070s | 2.018s | 18.23% |
| 47 | `lora_position:tinylfu` | 1.475s | 2.961s | 3.756s | 2.073s | 31.17% |
| 48 | `lora_position:s3fifo` | 1.479s | 2.897s | 3.725s | 2.008s | 49.70% |
| 49 | `sieve` | 1.481s | 2.996s | 3.754s | 2.044s | 36.92% |
| 50 | `lora_costaware:s3fifo` | 1.488s | 2.901s | 3.777s | 2.048s | 49.63% |
| 51 | `lora_correlated:tinylfu` | 1.489s | 2.563s | 3.502s | 2.029s | 31.22% |
| 52 | `lora_soft:lru_k` | 1.491s | 2.518s | 3.721s | 2.066s | 49.86% |
| 53 | `lora_freqweighted:sieve` | 1.492s | 2.675s | 3.403s | 2.009s | 18.23% |
| 54 | `tinylfu` | 1.493s | 2.499s | 3.182s | 2.021s | 26.46% |
| 55 | `lora_costaware:sieve` | 1.493s | 2.808s | 3.710s | 2.026s | 36.92% |
| 56 | `lora_ghost:lru` | 1.497s | 2.829s | 3.625s | 1.942s | 33.00% |
| 57 | `lora_adabudget:tinylfu` | 1.500s | 3.116s | 3.885s | 2.017s | 26.78% |
| 58 | `lora_prefixtree:tinylfu` | 1.504s | 2.818s | 3.583s | 2.010s | 34.20% |
| 59 | `lora_twolevel` | 1.506s | 2.857s | 3.603s | 2.064s | 19.45% |
| 60 | `lora_budget:tinylfu` | 1.515s | 2.908s | 3.702s | 2.091s | 31.11% |
| 61 | `lora_freqweighted:tinylfu` | 1.528s | 2.699s | 3.485s | 2.085s | 18.23% |
| 62 | `lora_hysteresis:lru_k` | 1.559s | 2.738s | 3.490s | 2.077s | 23.96% |
| 63 | `lora_hysteresis:lru` | 1.573s | 2.511s | 3.481s | 2.098s | 23.96% |
| 64 | `lora_tight:sieve` | 1.578s | 2.842s | 3.620s | 2.155s | 33.00% |
| 65 | `lora_ghost:tinylfu` | 1.581s | 2.744s | 3.576s | 2.073s | 33.00% |
| 66 | `lora_freqweighted:lru` | 1.605s | 2.957s | 3.783s | 2.163s | 18.23% |

## Scenario: `adapter_locality`

_Bursts of consecutive requests on the same adapter (burst_len=8). Mostly serviced by GPU prefix cache._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_freqweighted:s3fifo` | 0.475s | 0.911s | 0.922s | 0.888s |  0.00% |
| 2 | `lora_loose:sieve` | 0.476s | 0.878s | 0.892s | 0.897s |  0.00% |
| 3 | `lora_loose:lru_k` | 0.476s | 0.924s | 0.938s | 0.901s |  0.00% |
| 4 | `lora_hysteresis:s3fifo` | 0.477s | 0.538s | 0.543s | 0.896s |  0.00% |
| 5 | `lora_position:sieve` | 0.477s | 0.967s | 0.978s | 0.913s |  0.00% |
| 6 | `lora_tight:lru` | 0.478s | 0.890s | 0.902s | 0.890s |  0.00% |
| 7 | `lora_tight:tinylfu` | 0.478s | 0.899s | 0.923s | 0.920s |  0.00% |
| 8 | `lora_hysteresis:sieve` | 0.478s | 0.884s | 0.896s | 0.937s |  0.00% |
| 9 | `lora_soft:lru` | 0.478s | 0.524s | 0.535s | 0.893s |  0.00% |
| 10 | `lora_budget:tinylfu` | 0.478s | 1.086s | 1.098s | 0.899s |  0.00% |
| 11 | `lora_position:s3fifo` | 0.478s | 0.997s | 1.007s | 0.923s |  0.00% |
| 12 | `lora_soft:tinylfu` | 0.479s | 0.558s | 0.570s | 0.913s |  0.00% |
| 13 | `lora_freqweighted:sieve` | 0.479s | 0.956s | 0.968s | 0.904s |  0.00% |
| 14 | `lora_budget:lru` | 0.479s | 0.906s | 0.917s | 0.910s |  0.00% |
| 15 | `lora_costaware:s3fifo` | 0.479s | 0.908s | 0.918s | 0.910s |  0.00% |
| 16 | `lora_ghost:lru` | 0.479s | 0.998s | 1.008s | 0.919s |  0.00% |
| 17 | `lora_position:lru_k` | 0.479s | 1.003s | 1.023s | 0.909s |  0.00% |
| 18 | `lora_tight:s3fifo` | 0.480s | 0.915s | 0.926s | 0.896s |  0.00% |
| 19 | `lora_soft:sieve` | 0.480s | 0.896s | 0.907s | 0.898s |  0.00% |
| 20 | `lora_correlated:lru_k` | 0.480s | 1.118s | 1.128s | 0.902s |  0.00% |
| 21 | `lora_costaware:lru` | 0.480s | 0.977s | 0.990s | 0.905s |  0.00% |
| 22 | `lora_ghost:s3fifo` | 0.480s | 0.922s | 0.933s | 0.947s |  0.00% |
| 23 | `lora_loose:tinylfu` | 0.481s | 0.887s | 0.898s | 0.941s |  0.00% |
| 24 | `lora_correlated:s3fifo` | 0.481s | 0.928s | 0.939s | 0.905s |  0.00% |
| 25 | `lora_budget:s3fifo` | 0.481s | 1.034s | 1.045s | 0.906s |  0.00% |
| 26 | `lora_costaware:lru_k` | 0.481s | 0.952s | 0.963s | 0.915s |  0.00% |
| 27 | `lora_ghost:tinylfu` | 0.481s | 0.881s | 0.892s | 0.908s |  0.00% |
| 28 | `lora_prefixtree:lru` | 0.481s | 1.023s | 1.034s | 0.905s |  0.00% |
| 29 | `lora_prefixtree:lru_k` | 0.481s | 0.515s | 0.519s | 0.918s |  0.00% |
| 30 | `lora_tight:sieve` | 0.482s | 0.965s | 0.976s | 0.944s |  0.00% |
| 31 | `lora_loose:s3fifo` | 0.482s | 1.032s | 1.043s | 0.894s |  0.00% |
| 32 | `lora_hysteresis:lru_k` | 0.482s | 0.873s | 0.884s | 0.953s |  0.00% |
| 33 | `lora_adabudget:lru_k` | 0.482s | 0.887s | 0.897s | 0.941s |  0.00% |
| 34 | `lora_costaware:sieve` | 0.482s | 0.899s | 0.910s | 0.907s |  0.00% |
| 35 | `lora_correlated:sieve` | 0.483s | 0.928s | 0.939s | 0.908s |  0.00% |
| 36 | `lora_tight:lru_k` | 0.484s | 1.020s | 1.032s | 0.922s |  0.00% |
| 37 | `lora_loose:lru` | 0.484s | 0.891s | 0.901s | 0.895s |  0.00% |
| 38 | `lora_freqweighted:tinylfu` | 0.484s | 0.886s | 0.896s | 0.913s |  0.00% |
| 39 | `lora_adabudget:sieve` | 0.484s | 0.887s | 0.898s | 0.905s |  0.00% |
| 40 | `lora_adabudget:s3fifo` | 0.484s | 0.935s | 0.949s | 0.942s |  0.00% |
| 41 | `lora_prefixtree:s3fifo` | 0.484s | 0.909s | 0.921s | 0.915s |  0.00% |
| 42 | `lora_freqweighted:lru` | 0.485s | 0.925s | 0.938s | 0.911s |  0.00% |
| 43 | `lora_correlated:lru` | 0.485s | 0.898s | 0.908s | 0.915s |  0.00% |
| 44 | `lora_budget:lru_k` | 0.485s | 0.909s | 0.923s | 0.920s |  0.00% |
| 45 | `lora_position:tinylfu` | 0.485s | 0.884s | 0.896s | 0.913s |  0.00% |
| 46 | `lora_prefixtree:tinylfu` | 0.485s | 1.062s | 1.074s | 0.931s |  0.00% |
| 47 | `lora_adabudget:lru` | 0.486s | 0.924s | 0.935s | 0.953s |  0.00% |
| 48 | `lora_position:lru` | 0.486s | 0.929s | 0.942s | 0.914s |  0.00% |
| 49 | `lru_k` | 0.487s | 0.918s | 0.928s | 0.935s |  0.00% |
| 50 | `sieve` | 0.488s | 0.957s | 0.969s | 0.914s |  0.00% |
| 51 | `lora_adabudget:tinylfu` | 0.488s | 0.929s | 0.941s | 0.913s |  0.00% |
| 52 | `lora_ghost:lru_k` | 0.488s | 1.067s | 1.078s | 0.938s |  0.00% |
| 53 | `lru` | 0.489s | 0.955s | 0.966s | 0.909s |  0.00% |
| 54 | `lora_hysteresis:tinylfu` | 0.489s | 0.902s | 0.914s | 0.936s |  0.00% |
| 55 | `tinylfu` | 0.490s | 0.987s | 0.999s | 0.926s |  0.00% |
| 56 | `lora_correlated:tinylfu` | 0.491s | 0.925s | 0.937s | 0.917s |  0.00% |
| 57 | `lora_prefixtree:sieve` | 0.491s | 0.937s | 0.947s | 0.945s |  0.00% |
| 58 | `lora_hysteresis:lru` | 0.493s | 1.075s | 1.086s | 0.921s |  0.00% |
| 59 | `lora_costaware:tinylfu` | 0.493s | 0.953s | 0.967s | 0.922s |  0.00% |
| 60 | `lora_twolevel` | 0.497s | 1.112s | 1.125s | 0.939s |  0.00% |
| 61 | `lora_budget:sieve` | 0.499s | 0.988s | 1.000s | 0.924s |  0.00% |
| 62 | `lora_ghost:sieve` | 0.499s | 0.941s | 0.957s | 0.945s |  0.00% |
| 63 | `lora_soft:s3fifo` | 0.512s | 0.986s | 0.999s | 0.928s |  0.00% |
| 64 | `s3fifo` | 0.513s | 1.039s | 1.066s | 0.935s |  0.00% |
| 65 | `lora_soft:lru_k` | 0.514s | 0.994s | 1.007s | 0.943s |  0.00% |
| 66 | `lora_freqweighted:lru_k` | 0.614s | 1.022s | 1.033s | 1.046s |  0.00% |

## Scenario: `mixed_popularity`

_Zipfian (alpha=1.2) over 16 adapters — a few hot adapters dominate._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_freqweighted:lru` | 0.371s | 1.944s | 2.845s | 0.940s |  2.72% |
| 2 | `lora_tight:lru_k` | 0.378s | 1.609s | 1.918s | 0.928s |  2.72% |
| 3 | `lora_adabudget:tinylfu` | 0.381s | 1.915s | 2.958s | 0.967s |  4.51% |
| 4 | `lora_soft:sieve` | 0.382s | 2.052s | 2.980s | 0.991s |  5.43% |
| 5 | `lora_loose:tinylfu` | 0.383s | 2.002s | 3.020s | 1.011s |  7.50% |
| 6 | `lora_adabudget:lru` | 0.384s | 1.948s | 2.876s | 1.038s |  5.43% |
| 7 | `lora_ghost:lru_k` | 0.386s | 2.320s | 3.245s | 0.952s |  2.72% |
| 8 | `lora_twolevel` | 0.389s | 2.169s | 3.102s | 0.987s |  2.72% |
| 9 | `lora_adabudget:sieve` | 0.390s | 2.011s | 2.938s | 0.999s |  5.43% |
| 10 | `lora_hysteresis:lru` | 0.391s | 2.208s | 3.138s | 1.004s |  2.72% |
| 11 | `lora_freqweighted:s3fifo` | 0.391s | 1.710s | 1.961s | 0.966s |  2.72% |
| 12 | `tinylfu` | 0.392s | 1.996s | 2.884s | 0.963s |  7.23% |
| 13 | `lora_freqweighted:tinylfu` | 0.392s | 1.927s | 2.838s | 1.005s |  2.72% |
| 14 | `lora_soft:tinylfu` | 0.393s | 2.151s | 3.108s | 0.979s |  8.15% |
| 15 | `lora_budget:sieve` | 0.393s | 1.634s | 1.918s | 0.942s |  8.15% |
| 16 | `lora_budget:tinylfu` | 0.393s | 2.230s | 3.161s | 0.952s |  8.15% |
| 17 | `lru` | 0.394s | 1.997s | 2.910s | 1.004s |  5.43% |
| 18 | `lora_loose:lru_k` | 0.394s | 2.058s | 2.960s | 1.015s |  5.43% |
| 19 | `lora_correlated:sieve` | 0.394s | 2.009s | 2.932s | 0.960s |  5.43% |
| 20 | `lora_hysteresis:sieve` | 0.395s | 1.959s | 2.857s | 0.979s |  0.00% |
| 21 | `lora_correlated:lru_k` | 0.395s | 2.073s | 3.017s | 1.021s |  5.43% |
| 22 | `lora_adabudget:lru_k` | 0.395s | 1.976s | 2.870s | 0.997s |  5.43% |
| 23 | `lora_tight:tinylfu` | 0.397s | 2.166s | 3.102s | 0.975s |  2.72% |
| 24 | `lora_costaware:sieve` | 0.397s | 2.207s | 3.126s | 0.979s |  5.43% |
| 25 | `lora_position:lru_k` | 0.397s | 1.624s | 2.002s | 0.974s |  5.43% |
| 26 | `lora_prefixtree:tinylfu` | 0.398s | 2.230s | 3.139s | 1.002s |  7.23% |
| 27 | `lora_soft:lru` | 0.399s | 2.203s | 3.137s | 1.059s |  4.24% |
| 28 | `lora_adabudget:s3fifo` | 0.399s | 1.967s | 2.912s | 0.990s |  2.72% |
| 29 | `lru_k` | 0.401s | 1.942s | 2.912s | 0.988s |  5.43% |
| 30 | `lora_prefixtree:sieve` | 0.402s | 2.014s | 2.948s | 0.986s |  5.43% |
| 31 | `lora_hysteresis:s3fifo` | 0.403s | 1.969s | 2.868s | 1.038s |  0.00% |
| 32 | `lora_budget:s3fifo` | 0.404s | 2.041s | 3.031s | 1.077s |  8.15% |
| 33 | `lora_freqweighted:lru_k` | 0.405s | 2.302s | 3.216s | 0.980s |  2.72% |
| 34 | `lora_budget:lru_k` | 0.406s | 2.045s | 2.972s | 1.066s |  8.15% |
| 35 | `lora_loose:lru` | 0.408s | 1.928s | 2.859s | 0.931s |  5.43% |
| 36 | `lora_costaware:lru` | 0.408s | 1.990s | 3.001s | 1.050s |  5.43% |
| 37 | `lora_costaware:lru_k` | 0.408s | 2.368s | 3.262s | 1.007s |  5.43% |
| 38 | `lora_hysteresis:lru_k` | 0.409s | 1.952s | 2.848s | 0.984s |  2.72% |
| 39 | `lora_soft:lru_k` | 0.409s | 2.072s | 3.019s | 0.963s |  2.72% |
| 40 | `lora_ghost:sieve` | 0.411s | 2.008s | 2.971s | 0.994s |  2.72% |
| 41 | `lora_ghost:tinylfu` | 0.411s | 2.086s | 3.237s | 1.095s |  2.72% |
| 42 | `lora_prefixtree:lru` | 0.411s | 2.013s | 2.905s | 0.967s |  5.43% |
| 43 | `lora_position:s3fifo` | 0.414s | 2.187s | 3.190s | 1.024s |  5.43% |
| 44 | `lora_tight:lru` | 0.418s | 2.139s | 3.091s | 1.006s |  2.72% |
| 45 | `lora_tight:s3fifo` | 0.418s | 2.173s | 3.096s | 0.998s |  2.72% |
| 46 | `lora_soft:s3fifo` | 0.421s | 2.111s | 3.019s | 1.012s |  2.72% |
| 47 | `lora_position:tinylfu` | 0.422s | 2.053s | 2.995s | 1.008s |  8.15% |
| 48 | `lora_tight:sieve` | 0.423s | 1.918s | 2.816s | 0.995s |  2.72% |
| 49 | `lora_loose:s3fifo` | 0.424s | 1.973s | 2.877s | 0.987s |  2.72% |
| 50 | `lora_correlated:tinylfu` | 0.425s | 1.725s | 2.086s | 1.049s |  7.55% |
| 51 | `lora_prefixtree:s3fifo` | 0.429s | 1.850s | 2.046s | 1.024s |  5.43% |
| 52 | `lora_costaware:tinylfu` | 0.430s | 1.929s | 2.843s | 1.087s |  7.23% |
| 53 | `sieve` | 0.431s | 2.047s | 2.987s | 1.015s |  5.43% |
| 54 | `lora_loose:sieve` | 0.433s | 1.995s | 2.973s | 0.974s |  5.43% |
| 55 | `lora_correlated:lru` | 0.433s | 1.719s | 2.010s | 0.998s |  5.43% |
| 56 | `lora_position:lru` | 0.433s | 1.931s | 2.849s | 0.990s |  5.43% |
| 57 | `lora_correlated:s3fifo` | 0.437s | 2.100s | 3.169s | 1.122s |  2.72% |
| 58 | `lora_hysteresis:tinylfu` | 0.438s | 2.101s | 3.037s | 1.079s |  0.00% |
| 59 | `lora_position:sieve` | 0.438s | 2.036s | 3.116s | 0.981s |  5.43% |
| 60 | `lora_prefixtree:lru_k` | 0.444s | 2.025s | 2.973s | 1.093s |  5.43% |
| 61 | `lora_ghost:s3fifo` | 0.451s | 1.929s | 2.849s | 1.043s |  2.72% |
| 62 | `lora_budget:lru` | 0.456s | 2.067s | 2.986s | 1.019s |  8.15% |
| 63 | `s3fifo` | 0.462s | 2.072s | 3.031s | 1.007s |  5.43% |
| 64 | `lora_freqweighted:sieve` | 0.470s | 2.348s | 3.282s | 1.066s |  2.72% |
| 65 | `lora_ghost:lru` | 0.485s | 2.063s | 3.189s | 1.143s |  2.72% |
| 66 | `lora_costaware:s3fifo` | 0.558s | 2.035s | 2.916s | 0.971s |  5.43% |

## Overall ranking — avg TTFT-P50 rank

Average rank across the **two non-degenerate scenarios** (`adapter_thrashing` + `mixed_popularity`). Lower is better. `adapter_locality` is excluded because every policy's hit rate is 0% there.

| Rank | Policy | Avg Rank | thrashing | mixed |
|-----:|:-------|---------:|:---------:|:-----:|
| 1 | `lora_loose:tinylfu` | 3.0 | 1 | 5 |
| 2 | `lora_ghost:lru_k` | 5.5 | 4 | 7 |
| 3 | `lora_adabudget:sieve` | 7.0 | 5 | 9 |
| 4 | `lru` | 9.5 | 2 | 17 |
| 5 | `lora_adabudget:lru` | 9.5 | 13 | 6 |
| 6 | `lora_soft:tinylfu` | 10.0 | 6 | 14 |
| 7 | `lora_soft:sieve` | 15.0 | 26 | 4 |
| 8 | `lora_adabudget:lru_k` | 15.5 | 9 | 22 |
| 9 | `lora_hysteresis:s3fifo` | 17.0 | 3 | 31 |
| 10 | `lora_correlated:sieve` | 18.5 | 18 | 19 |
| 11 | `lora_tight:lru_k` | 19.5 | 37 | 2 |
| 12 | `lora_prefixtree:sieve` | 20.0 | 10 | 30 |
| 13 | `lora_soft:lru` | 21.0 | 15 | 27 |
| 14 | `lora_tight:tinylfu` | 21.5 | 20 | 23 |
| 15 | `lora_correlated:lru_k` | 21.5 | 22 | 21 |
| 16 | `lora_loose:lru` | 23.0 | 11 | 35 |
| 17 | `lora_hysteresis:sieve` | 23.5 | 27 | 20 |
| 18 | `lora_budget:lru_k` | 24.0 | 14 | 34 |
| 19 | `lora_loose:lru_k` | 24.0 | 30 | 18 |
| 20 | `lora_prefixtree:lru` | 25.0 | 8 | 42 |
| 21 | `lora_budget:sieve` | 25.5 | 36 | 15 |
| 22 | `lora_adabudget:s3fifo` | 26.0 | 24 | 28 |
| 23 | `lora_ghost:sieve` | 28.0 | 16 | 40 |
| 24 | `lora_tight:s3fifo` | 28.5 | 12 | 45 |
| 25 | `lru_k` | 28.5 | 28 | 29 |
| 26 | `lora_freqweighted:s3fifo` | 28.5 | 46 | 11 |
| 27 | `lora_freqweighted:lru_k` | 29.0 | 25 | 33 |
| 28 | `lora_position:lru_k` | 29.5 | 34 | 25 |
| 29 | `lora_adabudget:tinylfu` | 30.0 | 57 | 3 |
| 30 | `lora_tight:lru` | 30.5 | 17 | 44 |

## Recommendations

- **adapter_thrashing**: best TTFT P50 = `lora_loose:tinylfu` (1.234s, hit_rate=32.39%).
- **adapter_locality**: best TTFT P50 = `lora_freqweighted:s3fifo` (0.475s, hit_rate=0.00%).
- **mixed_popularity**: best TTFT P50 = `lora_freqweighted:lru` (0.371s, hit_rate=2.72%).
