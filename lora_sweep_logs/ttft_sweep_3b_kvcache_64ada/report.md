# LoRA-Aware Cache Policy TTFT Sweep — Results

Total runs: **198** (198 ok, 0 timed out, 0 failed).

Per-request TTFT was measured via `LLMEngine.add_request` + `engine.step()` loop, recording `now()` when each request first emitted a token.

## Scenario: `adapter_thrashing`

_Round-robin across 16 LoRA adapters (max_loras=2). Highest CPU-offload reuse pressure._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_budget:lru_k` | 1.589s | 8.363s | 9.929s | 2.044s | 66.37% |
| 2 | `lora_prefixtree:lru` | 1.596s | 8.329s | 9.918s | 2.032s | 66.37% |
| 3 | `lru` | 1.604s | 8.218s | 9.859s | 2.066s | 66.37% |
| 4 | `lora_costaware:lru_k` | 1.617s | 8.167s | 9.741s | 2.016s | 66.37% |
| 5 | `lora_budget:s3fifo` | 1.627s | 8.132s | 9.763s | 2.060s | 66.03% |
| 6 | `lora_prefixtree:lru_k` | 1.640s | 8.625s | 10.058s | 2.073s | 66.37% |
| 7 | `lora_position:lru_k` | 1.641s | 8.150s | 9.732s | 2.112s | 65.68% |
| 8 | `lru_k` | 1.644s | 8.369s | 9.993s | 2.123s | 66.37% |
| 9 | `lora_position:lru` | 1.676s | 8.146s | 9.778s | 2.105s | 66.37% |
| 10 | `lora_budget:lru` | 1.680s | 8.124s | 9.786s | 2.185s | 66.37% |
| 11 | `lora_soft:lru_k` | 1.682s | 8.242s | 9.861s | 2.039s | 66.37% |
| 12 | `lora_costaware:lru` | 1.758s | 8.206s | 9.772s | 2.198s | 66.37% |
| 13 | `lora_soft:tinylfu` | 1.923s | 8.427s | 10.097s | 2.463s | 62.91% |
| 14 | `lora_budget:sieve` | 1.969s | 8.571s | 10.203s | 2.457s | 63.61% |
| 15 | `lora_loose:lru` | 2.002s | 8.180s | 9.736s | 2.566s | 65.13% |
| 16 | `lora_costaware:s3fifo` | 2.048s | 8.047s | 9.581s | 2.539s | 61.88% |
| 17 | `lora_prefixtree:sieve` | 2.158s | 9.633s | 9.745s | 2.522s | 53.24% |
| 18 | `lora_loose:lru_k` | 2.166s | 8.573s | 10.398s | 2.562s | 62.57% |
| 19 | `lora_position:s3fifo` | 2.179s | 8.232s | 9.856s | 2.615s | 59.11% |
| 20 | `s3fifo` | 2.180s | 8.283s | 9.899s | 2.687s | 58.09% |
| 21 | `lora_position:sieve` | 2.216s | 8.448s | 9.653s | 2.715s | 52.54% |
| 22 | `lora_prefixtree:s3fifo` | 2.217s | 8.207s | 9.796s | 2.728s | 62.22% |
| 23 | `lora_costaware:sieve` | 2.239s | 8.763s | 9.859s | 2.783s | 50.12% |
| 24 | `sieve` | 2.250s | 8.413s | 9.677s | 2.672s | 53.24% |
| 25 | `lora_loose:s3fifo` | 2.363s | 8.492s | 10.026s | 3.117s | 48.40% |
| 26 | `lora_correlated:lru_k` | 2.374s | 8.616s | 9.618s | 3.045s | 47.71% |
| 27 | `lora_adabudget:lru` | 2.392s | 8.246s | 9.815s | 2.964s | 58.17% |
| 28 | `lora_correlated:lru` | 2.534s | 8.369s | 10.025s | 3.119s | 57.45% |
| 29 | `lora_correlated:sieve` | 2.593s | 9.177s | 9.605s | 3.227s | 44.59% |
| 30 | `lora_adabudget:s3fifo` | 2.751s | 8.826s | 10.495s | 3.584s | 39.41% |
| 31 | `lora_correlated:s3fifo` | 2.840s | 8.436s | 9.931s | 3.578s | 36.54% |
| 32 | `lora_hysteresis:lru` | 2.992s | 8.577s | 9.669s | 3.867s | 36.64% |
| 33 | `lora_tight:lru_k` | 2.998s | 8.490s | 9.659s | 3.906s | 36.64% |
| 34 | `lora_adabudget:lru_k` | 3.017s | 8.557s | 10.211s | 3.419s | 48.31% |
| 35 | `lora_ghost:tinylfu` | 3.055s | 8.677s | 9.630s | 3.867s | 35.61% |
| 36 | `tinylfu` | 3.083s | 8.330s | 9.914s | 3.994s | 36.00% |
| 37 | `lora_tight:s3fifo` | 3.105s | 8.411s | 9.613s | 3.925s | 36.64% |
| 38 | `lora_correlated:tinylfu` | 3.131s | 8.427s | 9.954s | 3.847s | 39.45% |
| 39 | `lora_ghost:lru` | 3.143s | 8.573s | 9.698s | 3.910s | 36.64% |
| 40 | `lora_soft:lru` | 3.149s | 8.433s | 9.801s | 3.731s | 37.16% |
| 41 | `lora_budget:tinylfu` | 3.211s | 8.429s | 9.753s | 4.310s | 32.86% |
| 42 | `lora_soft:sieve` | 3.218s | 8.897s | 9.749s | 3.860s | 37.28% |
| 43 | `lora_tight:tinylfu` | 3.249s | 8.187s | 9.745s | 3.924s | 35.61% |
| 44 | `lora_costaware:tinylfu` | 3.251s | 8.501s | 9.801s | 3.985s | 37.12% |
| 45 | `lora_loose:tinylfu` | 3.261s | 8.329s | 9.716s | 3.780s | 32.60% |
| 46 | `lora_ghost:s3fifo` | 3.265s | 8.760s | 9.899s | 3.979s | 35.95% |
| 47 | `lora_adabudget:sieve` | 3.298s | 9.611s | 9.930s | 3.866s | 31.11% |
| 48 | `lora_position:tinylfu` | 3.417s | 8.497s | 9.649s | 4.567s | 32.68% |
| 49 | `lora_prefixtree:tinylfu` | 3.425s | 8.327s | 9.667s | 4.241s | 36.93% |
| 50 | `lora_adabudget:tinylfu` | 3.458s | 8.068s | 9.675s | 4.083s | 35.30% |
| 51 | `lora_ghost:lru_k` | 3.462s | 8.601s | 9.680s | 4.227s | 33.53% |
| 52 | `lora_tight:sieve` | 3.472s | 9.063s | 9.737s | 4.537s | 33.88% |
| 53 | `lora_soft:s3fifo` | 3.498s | 8.485s | 9.949s | 4.386s | 32.49% |
| 54 | `lora_ghost:sieve` | 3.528s | 8.544s | 10.055s | 4.561s | 31.11% |
| 55 | `lora_loose:sieve` | 3.619s | 9.674s | 9.871s | 4.844s | 28.00% |
| 56 | `lora_tight:lru` | 3.651s | 8.550s | 10.239s | 4.967s | 29.73% |
| 57 | `lora_hysteresis:lru_k` | 3.655s | 8.519s | 9.878s | 4.998s | 30.77% |
| 58 | `lora_hysteresis:sieve` | 3.747s | 8.607s | 9.714s | 4.900s | 30.77% |
| 59 | `lora_hysteresis:s3fifo` | 3.901s | 8.622s | 9.667s | 4.897s | 33.19% |
| 60 | `lora_hysteresis:tinylfu` | 4.140s | 8.364s | 9.578s | 5.004s | 29.73% |
| 61 | `lora_freqweighted:s3fifo` | 4.389s | 8.699s | 9.990s | 5.140s | 19.01% |
| 62 | `lora_freqweighted:lru` | 4.497s | 8.653s | 9.564s | 5.166s | 17.28% |
| 63 | `lora_freqweighted:tinylfu` | 4.584s | 9.672s | 9.793s | 5.524s | 15.90% |
| 64 | `lora_freqweighted:sieve` | 4.671s | 9.618s | 9.778s | 5.590s | 15.56% |
| 65 | `lora_freqweighted:lru_k` | 4.681s | 9.591s | 9.758s | 5.393s | 18.67% |
| 66 | `lora_twolevel` | 4.844s | 9.077s | 9.885s | 5.738s | 13.14% |

## Scenario: `adapter_locality`

_Bursts of consecutive requests on the same adapter (burst_len=8). Mostly serviced by GPU prefix cache._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_tight:s3fifo` | 1.552s | 2.221s | 2.245s | 1.966s |  0.00% |
| 2 | `lora_correlated:s3fifo` | 1.579s | 2.286s | 2.311s | 2.235s |  0.00% |
| 3 | `lora_prefixtree:sieve` | 1.583s | 2.123s | 2.160s | 1.894s |  0.00% |
| 4 | `lora_position:lru_k` | 1.587s | 2.246s | 2.271s | 2.041s |  0.00% |
| 5 | `lora_soft:s3fifo` | 1.593s | 2.049s | 2.056s | 1.964s |  0.00% |
| 6 | `lora_adabudget:lru_k` | 1.597s | 2.104s | 2.127s | 1.901s |  0.00% |
| 7 | `lora_position:tinylfu` | 1.605s | 2.274s | 2.301s | 2.022s |  0.00% |
| 8 | `lora_freqweighted:lru_k` | 1.606s | 2.368s | 2.395s | 1.861s |  0.00% |
| 9 | `lora_freqweighted:lru` | 1.607s | 2.151s | 2.168s | 1.955s |  0.00% |
| 10 | `lora_adabudget:lru` | 1.607s | 2.196s | 2.227s | 1.924s |  0.00% |
| 11 | `lora_hysteresis:s3fifo` | 1.608s | 2.366s | 2.396s | 1.993s |  0.00% |
| 12 | `lora_freqweighted:s3fifo` | 1.608s | 2.196s | 2.223s | 2.233s |  0.00% |
| 13 | `lora_correlated:lru` | 1.608s | 2.307s | 2.340s | 1.933s |  0.00% |
| 14 | `lora_adabudget:s3fifo` | 1.613s | 2.213s | 2.240s | 1.904s |  0.00% |
| 15 | `lora_budget:lru_k` | 1.614s | 2.229s | 2.260s | 2.230s |  0.00% |
| 16 | `lora_loose:s3fifo` | 1.616s | 2.240s | 2.264s | 2.228s |  0.00% |
| 17 | `lora_correlated:tinylfu` | 1.616s | 2.207s | 2.233s | 1.897s |  0.00% |
| 18 | `lru` | 1.617s | 2.212s | 2.237s | 2.224s |  0.00% |
| 19 | `lora_adabudget:tinylfu` | 1.618s | 2.252s | 2.278s | 2.249s |  0.00% |
| 20 | `lora_hysteresis:sieve` | 1.621s | 2.112s | 2.127s | 2.015s |  0.00% |
| 21 | `lora_adabudget:sieve` | 1.623s | 2.309s | 2.339s | 2.253s |  0.00% |
| 22 | `lora_tight:tinylfu` | 1.624s | 2.229s | 2.254s | 2.226s |  0.00% |
| 23 | `lora_soft:sieve` | 1.625s | 2.283s | 2.311s | 2.222s |  0.00% |
| 24 | `lru_k` | 1.628s | 2.393s | 2.420s | 1.863s |  0.00% |
| 25 | `lora_costaware:sieve` | 1.629s | 2.203s | 2.230s | 1.959s |  0.00% |
| 26 | `lora_costaware:s3fifo` | 1.630s | 2.237s | 2.263s | 1.848s |  0.00% |
| 27 | `lora_position:sieve` | 1.631s | 2.405s | 2.429s | 2.246s |  0.00% |
| 28 | `lora_soft:lru_k` | 1.635s | 2.198s | 2.228s | 1.969s |  0.00% |
| 29 | `lora_position:s3fifo` | 1.637s | 2.378s | 2.403s | 1.894s |  0.00% |
| 30 | `lora_prefixtree:s3fifo` | 1.637s | 2.215s | 2.242s | 2.242s |  0.00% |
| 31 | `lora_loose:lru_k` | 1.640s | 2.248s | 2.275s | 1.941s |  0.00% |
| 32 | `lora_loose:lru` | 1.641s | 2.205s | 2.230s | 2.239s |  0.00% |
| 33 | `lora_freqweighted:sieve` | 1.641s | 2.204s | 2.229s | 2.237s |  0.00% |
| 34 | `lora_costaware:tinylfu` | 1.642s | 2.224s | 2.253s | 2.253s |  0.00% |
| 35 | `lora_soft:tinylfu` | 1.643s | 2.216s | 2.242s | 2.233s |  0.00% |
| 36 | `lora_budget:s3fifo` | 1.643s | 2.322s | 2.348s | 1.955s |  0.00% |
| 37 | `lora_tight:lru_k` | 1.644s | 2.256s | 2.282s | 2.239s |  0.00% |
| 38 | `lora_soft:lru` | 1.644s | 2.222s | 2.247s | 1.976s |  0.00% |
| 39 | `lora_hysteresis:lru` | 1.647s | 2.273s | 2.297s | 2.288s |  0.00% |
| 40 | `lora_prefixtree:tinylfu` | 1.649s | 2.281s | 2.307s | 2.244s |  0.00% |
| 41 | `lora_correlated:lru_k` | 1.651s | 2.225s | 2.249s | 2.015s |  0.00% |
| 42 | `lora_tight:sieve` | 1.655s | 2.361s | 2.387s | 2.140s |  0.00% |
| 43 | `lora_correlated:sieve` | 1.655s | 2.246s | 2.279s | 2.136s |  0.00% |
| 44 | `lora_ghost:lru` | 1.658s | 2.264s | 2.289s | 2.234s |  0.00% |
| 45 | `lora_twolevel` | 1.662s | 2.337s | 2.362s | 2.255s |  0.00% |
| 46 | `lora_ghost:s3fifo` | 1.667s | 2.271s | 2.298s | 2.226s |  0.00% |
| 47 | `lora_prefixtree:lru_k` | 1.675s | 2.223s | 2.248s | 2.232s |  0.00% |
| 48 | `lora_costaware:lru` | 1.676s | 2.199s | 2.224s | 2.221s |  0.00% |
| 49 | `tinylfu` | 1.687s | 2.275s | 2.302s | 2.276s |  0.00% |
| 50 | `lora_position:lru` | 1.690s | 2.197s | 2.225s | 2.247s |  0.00% |
| 51 | `sieve` | 1.702s | 2.315s | 2.343s | 2.272s |  0.00% |
| 52 | `lora_budget:sieve` | 1.702s | 2.201s | 2.226s | 2.252s |  0.00% |
| 53 | `lora_ghost:lru_k` | 1.707s | 2.416s | 2.450s | 2.398s |  0.00% |
| 54 | `lora_costaware:lru_k` | 1.711s | 2.290s | 2.317s | 1.917s |  0.00% |
| 55 | `lora_budget:lru` | 1.717s | 2.272s | 2.297s | 2.249s |  0.00% |
| 56 | `lora_ghost:sieve` | 1.720s | 2.293s | 2.318s | 2.246s |  0.00% |
| 57 | `lora_ghost:tinylfu` | 1.724s | 2.348s | 2.378s | 2.241s |  0.00% |
| 58 | `lora_budget:tinylfu` | 1.725s | 2.303s | 2.316s | 2.312s |  0.00% |
| 59 | `lora_hysteresis:lru_k` | 1.730s | 2.234s | 2.261s | 2.265s |  0.00% |
| 60 | `lora_hysteresis:tinylfu` | 1.731s | 2.242s | 2.268s | 2.235s |  0.00% |
| 61 | `lora_prefixtree:lru` | 1.740s | 2.436s | 2.461s | 2.238s |  0.00% |
| 62 | `s3fifo` | 1.745s | 2.301s | 2.326s | 2.249s |  0.00% |
| 63 | `lora_loose:sieve` | 1.764s | 2.320s | 2.345s | 2.308s |  0.00% |
| 64 | `lora_freqweighted:tinylfu` | 1.767s | 2.364s | 2.388s | 2.160s |  0.00% |
| 65 | `lora_loose:tinylfu` | 1.788s | 2.439s | 2.477s | 2.329s |  0.00% |
| 66 | `lora_tight:lru` | 1.820s | 2.327s | 2.362s | 2.106s |  0.00% |

## Scenario: `mixed_popularity`

_Zipfian (alpha=1.2) over 16 adapters — a few hot adapters dominate._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_position:tinylfu` | 1.269s | 3.557s | 6.214s | 1.865s |  1.31% |
| 2 | `lora_soft:tinylfu` | 1.270s | 3.603s | 6.319s | 1.811s |  1.31% |
| 3 | `lora_loose:lru` | 1.289s | 3.274s | 6.140s | 1.939s |  1.31% |
| 4 | `lora_adabudget:sieve` | 1.297s | 3.464s | 5.978s | 1.840s |  1.31% |
| 5 | `lora_loose:sieve` | 1.300s | 3.597s | 6.258s | 2.140s |  1.31% |
| 6 | `lora_budget:s3fifo` | 1.300s | 3.623s | 6.131s | 1.827s |  1.31% |
| 7 | `sieve` | 1.308s | 3.710s | 6.253s | 1.893s |  1.31% |
| 8 | `s3fifo` | 1.308s | 3.611s | 6.078s | 1.897s |  1.31% |
| 9 | `lora_freqweighted:tinylfu` | 1.308s | 3.884s | 6.406s | 1.781s |  1.31% |
| 10 | `lora_ghost:sieve` | 1.308s | 3.572s | 6.091s | 1.999s |  1.31% |
| 11 | `lora_hysteresis:sieve` | 1.312s | 3.012s | 6.409s | 1.950s |  1.31% |
| 12 | `lora_correlated:tinylfu` | 1.313s | 3.687s | 6.342s | 1.858s |  1.31% |
| 13 | `tinylfu` | 1.318s | 3.545s | 6.158s | 1.990s |  1.31% |
| 14 | `lora_budget:lru` | 1.323s | 3.568s | 6.559s | 1.952s |  1.31% |
| 15 | `lora_position:lru` | 1.324s | 3.538s | 6.015s | 1.974s |  1.31% |
| 16 | `lora_adabudget:tinylfu` | 1.325s | 2.935s | 6.004s | 1.802s |  1.31% |
| 17 | `lora_loose:tinylfu` | 1.327s | 3.537s | 5.135s | 2.007s |  1.31% |
| 18 | `lora_loose:s3fifo` | 1.332s | 3.468s | 5.979s | 1.725s |  1.31% |
| 19 | `lora_soft:lru_k` | 1.333s | 3.642s | 6.224s | 2.020s |  1.31% |
| 20 | `lora_tight:lru` | 1.340s | 3.485s | 5.926s | 1.707s |  1.31% |
| 21 | `lru_k` | 1.344s | 3.203s | 6.453s | 1.991s |  1.31% |
| 22 | `lora_freqweighted:lru_k` | 1.344s | 3.598s | 6.278s | 1.982s |  1.31% |
| 23 | `lora_correlated:lru_k` | 1.344s | 3.487s | 5.973s | 1.980s |  1.31% |
| 24 | `lora_hysteresis:tinylfu` | 1.345s | 3.521s | 6.000s | 1.810s |  1.31% |
| 25 | `lora_twolevel` | 1.348s | 3.705s | 6.470s | 1.954s |  1.31% |
| 26 | `lora_budget:sieve` | 1.349s | 3.720s | 6.266s | 1.892s |  1.31% |
| 27 | `lora_loose:lru_k` | 1.351s | 3.514s | 5.997s | 1.909s |  1.31% |
| 28 | `lora_hysteresis:s3fifo` | 1.353s | 3.565s | 6.139s | 1.921s |  1.31% |
| 29 | `lora_costaware:sieve` | 1.355s | 3.604s | 6.192s | 1.898s |  1.31% |
| 30 | `lora_position:s3fifo` | 1.357s | 3.193s | 6.254s | 1.944s |  1.31% |
| 31 | `lora_prefixtree:tinylfu` | 1.357s | 3.466s | 5.373s | 2.189s |  1.31% |
| 32 | `lora_ghost:s3fifo` | 1.362s | 3.474s | 5.933s | 1.800s |  1.31% |
| 33 | `lora_position:sieve` | 1.363s | 3.568s | 6.395s | 1.975s |  1.31% |
| 34 | `lora_freqweighted:s3fifo` | 1.364s | 3.661s | 6.223s | 1.976s |  1.31% |
| 35 | `lora_correlated:sieve` | 1.364s | 3.554s | 6.902s | 1.955s |  1.31% |
| 36 | `lora_hysteresis:lru` | 1.369s | 3.303s | 4.871s | 1.938s |  1.31% |
| 37 | `lora_correlated:lru` | 1.372s | 3.511s | 6.146s | 1.781s |  1.31% |
| 38 | `lora_prefixtree:sieve` | 1.376s | 3.128s | 6.155s | 1.961s |  1.31% |
| 39 | `lora_ghost:lru_k` | 1.380s | 3.780s | 6.199s | 1.896s |  1.31% |
| 40 | `lora_adabudget:s3fifo` | 1.383s | 3.668s | 6.357s | 1.775s |  1.31% |
| 41 | `lora_soft:s3fifo` | 1.390s | 3.090s | 6.027s | 1.960s |  1.31% |
| 42 | `lora_position:lru_k` | 1.390s | 3.105s | 6.279s | 1.994s |  1.31% |
| 43 | `lora_freqweighted:lru` | 1.394s | 3.622s | 6.365s | 1.944s |  1.31% |
| 44 | `lora_costaware:s3fifo` | 1.395s | 3.017s | 6.225s | 1.906s |  1.31% |
| 45 | `lru` | 1.400s | 3.881s | 6.405s | 2.072s |  1.31% |
| 46 | `lora_correlated:s3fifo` | 1.404s | 3.536s | 6.311s | 2.010s |  1.31% |
| 47 | `lora_costaware:lru` | 1.407s | 3.814s | 6.539s | 1.915s |  1.31% |
| 48 | `lora_tight:lru_k` | 1.417s | 3.539s | 5.354s | 1.921s |  1.31% |
| 49 | `lora_soft:lru` | 1.419s | 3.725s | 6.353s | 1.896s |  1.31% |
| 50 | `lora_freqweighted:sieve` | 1.422s | 3.918s | 6.454s | 2.136s |  1.31% |
| 51 | `lora_ghost:tinylfu` | 1.423s | 3.847s | 6.723s | 2.045s |  1.31% |
| 52 | `lora_tight:tinylfu` | 1.430s | 3.641s | 6.297s | 1.844s |  1.31% |
| 53 | `lora_tight:s3fifo` | 1.434s | 3.182s | 6.339s | 2.014s |  1.31% |
| 54 | `lora_adabudget:lru_k` | 1.434s | 3.219s | 6.359s | 1.998s |  1.31% |
| 55 | `lora_prefixtree:lru` | 1.442s | 3.172s | 5.382s | 2.035s |  1.31% |
| 56 | `lora_budget:lru_k` | 1.445s | 3.600s | 6.313s | 1.935s |  1.31% |
| 57 | `lora_costaware:tinylfu` | 1.450s | 3.354s | 6.191s | 2.080s |  1.31% |
| 58 | `lora_costaware:lru_k` | 1.457s | 3.794s | 6.397s | 1.852s |  1.31% |
| 59 | `lora_tight:sieve` | 1.459s | 3.513s | 5.992s | 2.157s |  1.31% |
| 60 | `lora_hysteresis:lru_k` | 1.471s | 3.475s | 4.880s | 1.929s |  1.31% |
| 61 | `lora_budget:tinylfu` | 1.480s | 3.516s | 5.965s | 1.878s |  1.31% |
| 62 | `lora_prefixtree:lru_k` | 1.489s | 3.893s | 6.453s | 1.857s |  1.31% |
| 63 | `lora_ghost:lru` | 1.491s | 3.137s | 5.196s | 2.062s |  1.31% |
| 64 | `lora_soft:sieve` | 1.510s | 3.372s | 5.490s | 2.085s |  1.31% |
| 65 | `lora_adabudget:lru` | 1.512s | 3.184s | 5.374s | 1.983s |  1.31% |
| 66 | `lora_prefixtree:s3fifo` | 1.550s | 3.491s | 6.570s | 2.052s |  1.31% |

## Overall ranking — avg TTFT-P50 rank

Average rank across the **two non-degenerate scenarios** (`adapter_thrashing` + `mixed_popularity`). Lower is better. `adapter_locality` is excluded because every policy's hit rate is 0% there.

| Rank | Policy | Avg Rank | thrashing | mixed |
|-----:|:-------|---------:|:---------:|:-----:|
| 1 | `lora_budget:s3fifo` | 5.5 | 5 | 6 |
| 2 | `lora_soft:tinylfu` | 7.5 | 13 | 2 |
| 3 | `lora_loose:lru` | 9.0 | 15 | 3 |
| 4 | `lora_position:lru` | 12.0 | 9 | 15 |
| 5 | `lora_budget:lru` | 12.0 | 10 | 14 |
| 6 | `s3fifo` | 14.0 | 20 | 8 |
| 7 | `lru_k` | 14.5 | 8 | 21 |
| 8 | `lora_soft:lru_k` | 15.0 | 11 | 19 |
| 9 | `sieve` | 15.5 | 24 | 7 |
| 10 | `lora_budget:sieve` | 20.0 | 14 | 26 |
| 11 | `lora_loose:s3fifo` | 21.5 | 25 | 18 |
| 12 | `lora_loose:lru_k` | 22.5 | 18 | 27 |
| 13 | `lru` | 24.0 | 3 | 45 |
| 14 | `lora_position:lru_k` | 24.5 | 7 | 42 |
| 15 | `lora_position:s3fifo` | 24.5 | 19 | 30 |
| 16 | `lora_correlated:lru_k` | 24.5 | 26 | 23 |
| 17 | `tinylfu` | 24.5 | 36 | 13 |
| 18 | `lora_position:tinylfu` | 24.5 | 48 | 1 |
| 19 | `lora_correlated:tinylfu` | 25.0 | 38 | 12 |
| 20 | `lora_adabudget:sieve` | 25.5 | 47 | 4 |
| 21 | `lora_costaware:sieve` | 26.0 | 23 | 29 |
| 22 | `lora_position:sieve` | 27.0 | 21 | 33 |
| 23 | `lora_prefixtree:sieve` | 27.5 | 17 | 38 |
| 24 | `lora_budget:lru_k` | 28.5 | 1 | 56 |
| 25 | `lora_prefixtree:lru` | 28.5 | 2 | 55 |
| 26 | `lora_costaware:lru` | 29.5 | 12 | 47 |
| 27 | `lora_costaware:s3fifo` | 30.0 | 16 | 44 |
| 28 | `lora_loose:sieve` | 30.0 | 55 | 5 |
| 29 | `lora_costaware:lru_k` | 31.0 | 4 | 58 |
| 30 | `lora_loose:tinylfu` | 31.0 | 45 | 17 |

## Recommendations

- **adapter_thrashing**: best TTFT P50 = `lora_budget:lru_k` (1.589s, hit_rate=66.37%).
- **adapter_locality**: best TTFT P50 = `lora_tight:s3fifo` (1.552s, hit_rate=0.00%).
- **mixed_popularity**: best TTFT P50 = `lora_position:tinylfu` (1.269s, hit_rate=1.31%).