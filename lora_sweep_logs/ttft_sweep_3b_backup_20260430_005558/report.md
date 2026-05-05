# LoRA-Aware Cache Policy TTFT Sweep — Results

Total runs: **198** (198 ok, 0 timed out, 0 failed).

Per-request TTFT was measured via `LLMEngine.add_request` + `engine.step()` loop, recording `now()` when each request first emitted a token.

## Scenario: `adapter_thrashing`

_Round-robin across 16 LoRA adapters (max_loras=2). Highest CPU-offload reuse pressure._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_tight:tinylfu` | 1.220s | 7.384s | 10.247s | 1.857s |  0.00% |
| 2 | `lora_prefixtree:lru_k` | 1.303s | 6.982s | 9.930s | 1.969s |  0.00% |
| 3 | `lora_soft:tinylfu` | 1.308s | 7.357s | 9.903s | 1.876s |  0.00% |
| 4 | `lora_prefixtree:sieve` | 1.326s | 7.054s | 9.629s | 1.929s |  0.00% |
| 5 | `lora_costaware:lru_k` | 1.328s | 7.173s | 10.065s | 1.854s |  0.00% |
| 6 | `lora_freqweighted:s3fifo` | 1.332s | 7.494s | 9.919s | 1.899s |  0.00% |
| 7 | `lora_adabudget:lru_k` | 1.335s | 7.273s | 9.861s | 1.916s |  0.00% |
| 8 | `lora_prefixtree:tinylfu` | 1.339s | 7.852s | 10.425s | 1.874s |  0.00% |
| 9 | `lora_correlated:sieve` | 1.350s | 7.528s | 9.571s | 1.942s |  0.00% |
| 10 | `lora_ghost:sieve` | 1.351s | 7.432s | 9.838s | 1.930s |  0.00% |
| 11 | `lora_prefixtree:s3fifo` | 1.352s | 7.043s | 9.739s | 1.895s |  0.00% |
| 12 | `lora_freqweighted:lru` | 1.353s | 7.309s | 9.971s | 1.868s |  0.00% |
| 13 | `lora_twolevel` | 1.354s | 7.516s | 10.102s | 1.885s |  0.00% |
| 14 | `lora_costaware:lru` | 1.366s | 7.185s | 9.675s | 1.915s |  0.00% |
| 15 | `lora_soft:lru_k` | 1.372s | 6.888s | 9.877s | 1.921s |  0.00% |
| 16 | `lora_position:sieve` | 1.373s | 7.331s | 9.903s | 1.919s |  0.00% |
| 17 | `lora_budget:sieve` | 1.379s | 7.293s | 9.856s | 1.907s |  0.00% |
| 18 | `lora_freqweighted:sieve` | 1.385s | 7.102s | 9.510s | 1.912s |  0.00% |
| 19 | `lora_hysteresis:lru_k` | 1.395s | 7.370s | 9.761s | 1.940s |  0.00% |
| 20 | `lora_soft:sieve` | 1.396s | 7.276s | 9.740s | 1.945s |  0.00% |
| 21 | `lora_tight:lru_k` | 1.402s | 7.471s | 9.934s | 1.954s |  0.00% |
| 22 | `lora_ghost:lru_k` | 1.404s | 7.140s | 9.531s | 1.880s |  0.00% |
| 23 | `s3fifo` | 1.405s | 7.103s | 9.712s | 1.938s |  0.00% |
| 24 | `lru` | 1.424s | 7.858s | 11.190s | 1.897s |  0.00% |
| 25 | `lora_soft:s3fifo` | 1.427s | 7.223s | 9.745s | 1.961s |  0.00% |
| 26 | `lru_k` | 1.429s | 7.435s | 9.851s | 1.957s |  0.00% |
| 27 | `lora_hysteresis:sieve` | 1.436s | 7.512s | 9.956s | 1.919s |  0.00% |
| 28 | `lora_freqweighted:tinylfu` | 1.439s | 7.043s | 9.462s | 1.925s |  0.00% |
| 29 | `lora_budget:s3fifo` | 1.441s | 7.576s | 10.235s | 1.913s |  0.00% |
| 30 | `lora_budget:tinylfu` | 1.448s | 7.097s | 9.591s | 1.913s |  0.00% |
| 31 | `lora_hysteresis:lru` | 1.453s | 7.112s | 9.602s | 1.946s |  0.00% |
| 32 | `lora_budget:lru` | 1.453s | 7.477s | 9.925s | 2.021s |  0.00% |
| 33 | `lora_loose:lru` | 1.457s | 7.162s | 9.891s | 1.989s |  0.00% |
| 34 | `lora_costaware:sieve` | 1.458s | 7.742s | 10.147s | 1.888s |  0.00% |
| 35 | `lora_position:lru_k` | 1.459s | 7.293s | 9.884s | 1.942s |  0.00% |
| 36 | `lora_costaware:s3fifo` | 1.481s | 7.819s | 9.491s | 1.994s |  0.00% |
| 37 | `sieve` | 1.486s | 7.151s | 9.766s | 1.912s |  0.00% |
| 38 | `lora_correlated:lru` | 1.491s | 7.271s | 9.968s | 1.888s |  0.00% |
| 39 | `lora_position:lru` | 1.498s | 6.980s | 9.719s | 2.080s |  0.00% |
| 40 | `lora_loose:s3fifo` | 1.501s | 7.364s | 9.866s | 1.902s |  0.00% |
| 41 | `tinylfu` | 1.504s | 7.411s | 9.937s | 1.930s |  0.00% |
| 42 | `lora_correlated:s3fifo` | 1.518s | 7.288s | 9.704s | 1.986s |  0.00% |
| 43 | `lora_adabudget:tinylfu` | 1.538s | 7.134s | 9.705s | 1.941s |  0.00% |
| 44 | `lora_adabudget:s3fifo` | 1.540s | 6.992s | 9.805s | 1.920s |  0.00% |
| 45 | `lora_soft:lru` | 1.547s | 6.884s | 9.780s | 1.986s |  0.00% |
| 46 | `lora_loose:tinylfu` | 1.548s | 7.190s | 9.990s | 1.904s |  0.00% |
| 47 | `lora_correlated:tinylfu` | 1.555s | 7.118s | 9.514s | 1.890s |  0.00% |
| 48 | `lora_loose:lru_k` | 1.557s | 7.286s | 9.721s | 1.974s |  0.00% |
| 49 | `lora_budget:lru_k` | 1.560s | 7.565s | 10.123s | 1.933s |  0.00% |
| 50 | `lora_tight:s3fifo` | 1.571s | 7.168s | 9.610s | 1.926s |  0.00% |
| 51 | `lora_position:s3fifo` | 1.574s | 7.625s | 9.958s | 2.041s |  0.00% |
| 52 | `lora_freqweighted:lru_k` | 1.577s | 7.023s | 10.199s | 1.954s |  0.00% |
| 53 | `lora_position:tinylfu` | 1.581s | 7.418s | 9.838s | 1.941s |  0.00% |
| 54 | `lora_ghost:s3fifo` | 1.586s | 7.125s | 9.523s | 1.967s |  0.00% |
| 55 | `lora_tight:sieve` | 1.587s | 7.473s | 9.923s | 1.985s |  0.00% |
| 56 | `lora_loose:sieve` | 1.590s | 7.249s | 9.768s | 2.036s |  0.00% |
| 57 | `lora_hysteresis:s3fifo` | 1.591s | 7.370s | 9.600s | 2.044s |  0.00% |
| 58 | `lora_tight:lru` | 1.599s | 7.072s | 9.636s | 1.949s |  0.00% |
| 59 | `lora_adabudget:lru` | 1.601s | 7.173s | 9.566s | 1.934s |  0.00% |
| 60 | `lora_costaware:tinylfu` | 1.604s | 7.462s | 9.822s | 2.081s |  0.00% |
| 61 | `lora_prefixtree:lru` | 1.617s | 7.186s | 9.896s | 1.982s |  0.00% |
| 62 | `lora_ghost:tinylfu` | 1.620s | 6.883s | 9.880s | 2.003s |  0.00% |
| 63 | `lora_hysteresis:tinylfu` | 1.628s | 7.353s | 9.733s | 1.988s |  0.00% |
| 64 | `lora_adabudget:sieve` | 1.632s | 7.508s | 9.903s | 2.075s |  0.00% |
| 65 | `lora_ghost:lru` | 1.721s | 6.945s | 9.931s | 2.190s |  0.00% |
| 66 | `lora_correlated:lru_k` | 1.807s | 7.153s | 9.665s | 2.299s |  0.00% |

## Scenario: `adapter_locality`

_Bursts of consecutive requests on the same adapter (burst_len=8). Mostly serviced by GPU prefix cache._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_prefixtree:lru` | 1.586s | 2.419s | 2.439s | 1.924s |  0.00% |
| 2 | `lora_soft:lru_k` | 1.592s | 2.228s | 2.247s | 1.893s |  0.00% |
| 3 | `s3fifo` | 1.595s | 2.221s | 2.241s | 1.919s |  0.00% |
| 4 | `lora_position:sieve` | 1.597s | 2.292s | 2.312s | 1.964s |  0.00% |
| 5 | `lora_hysteresis:sieve` | 1.601s | 2.201s | 2.227s | 2.034s |  0.00% |
| 6 | `lora_costaware:lru` | 1.601s | 2.180s | 2.201s | 2.052s |  0.00% |
| 7 | `lora_loose:lru_k` | 1.604s | 2.319s | 2.341s | 1.923s |  0.00% |
| 8 | `lora_budget:sieve` | 1.604s | 2.046s | 2.056s | 1.840s |  0.00% |
| 9 | `tinylfu` | 1.605s | 2.185s | 2.211s | 1.885s |  0.00% |
| 10 | `lru_k` | 1.606s | 2.068s | 2.080s | 1.835s |  0.00% |
| 11 | `lora_prefixtree:tinylfu` | 1.606s | 2.207s | 2.227s | 1.898s |  0.00% |
| 12 | `lora_soft:lru` | 1.608s | 2.187s | 2.213s | 1.853s |  0.00% |
| 13 | `lora_adabudget:tinylfu` | 1.609s | 2.212s | 2.233s | 1.888s |  0.00% |
| 14 | `lora_tight:s3fifo` | 1.610s | 2.274s | 2.295s | 1.896s |  0.00% |
| 15 | `lora_freqweighted:lru` | 1.610s | 2.273s | 2.295s | 1.969s |  0.00% |
| 16 | `lora_budget:lru_k` | 1.611s | 2.307s | 2.327s | 1.888s |  0.00% |
| 17 | `lora_ghost:lru_k` | 1.611s | 2.226s | 2.246s | 1.880s |  0.00% |
| 18 | `lora_loose:lru` | 1.613s | 2.115s | 2.131s | 1.865s |  0.00% |
| 19 | `lora_position:s3fifo` | 1.613s | 2.083s | 2.092s | 1.896s |  0.00% |
| 20 | `lora_twolevel` | 1.614s | 2.255s | 2.275s | 1.949s |  0.00% |
| 21 | `lora_correlated:lru_k` | 1.615s | 2.211s | 2.231s | 1.987s |  0.00% |
| 22 | `lora_ghost:lru` | 1.616s | 2.259s | 2.279s | 1.922s |  0.00% |
| 23 | `lora_prefixtree:sieve` | 1.616s | 2.332s | 2.352s | 1.903s |  0.00% |
| 24 | `lora_loose:tinylfu` | 1.617s | 2.194s | 2.214s | 1.863s |  0.00% |
| 25 | `lora_soft:tinylfu` | 1.618s | 2.185s | 2.206s | 1.974s |  0.00% |
| 26 | `lora_costaware:lru_k` | 1.618s | 2.187s | 2.212s | 2.034s |  0.00% |
| 27 | `lora_loose:sieve` | 1.621s | 2.391s | 2.411s | 2.069s |  0.00% |
| 28 | `lora_adabudget:lru_k` | 1.622s | 2.234s | 2.254s | 2.023s |  0.00% |
| 29 | `lora_tight:lru_k` | 1.624s | 2.200s | 2.220s | 2.004s |  0.00% |
| 30 | `lora_costaware:tinylfu` | 1.625s | 2.336s | 2.356s | 2.024s |  0.00% |
| 31 | `lora_correlated:sieve` | 1.626s | 2.173s | 2.199s | 1.963s |  0.00% |
| 32 | `lora_tight:sieve` | 1.627s | 2.217s | 2.237s | 1.958s |  0.00% |
| 33 | `lora_tight:tinylfu` | 1.627s | 2.293s | 2.326s | 1.873s |  0.00% |
| 34 | `lora_freqweighted:s3fifo` | 1.628s | 2.277s | 2.297s | 1.971s |  0.00% |
| 35 | `lora_position:lru_k` | 1.628s | 2.255s | 2.280s | 1.973s |  0.00% |
| 36 | `lora_position:lru` | 1.635s | 2.204s | 2.226s | 1.866s |  0.00% |
| 37 | `lora_tight:lru` | 1.636s | 2.176s | 2.202s | 2.085s |  0.00% |
| 38 | `lora_budget:s3fifo` | 1.636s | 2.178s | 2.198s | 2.020s |  0.00% |
| 39 | `lora_correlated:s3fifo` | 1.640s | 2.211s | 2.231s | 1.913s |  0.00% |
| 40 | `lora_costaware:s3fifo` | 1.640s | 2.150s | 2.169s | 1.982s |  0.00% |
| 41 | `lora_freqweighted:tinylfu` | 1.641s | 2.184s | 2.204s | 2.031s |  0.00% |
| 42 | `sieve` | 1.650s | 2.366s | 2.386s | 2.055s |  0.00% |
| 43 | `lora_position:tinylfu` | 1.650s | 2.328s | 2.349s | 2.051s |  0.00% |
| 44 | `lora_hysteresis:lru` | 1.651s | 2.171s | 2.197s | 1.906s |  0.00% |
| 45 | `lora_freqweighted:sieve` | 1.652s | 2.061s | 2.073s | 2.007s |  0.00% |
| 46 | `lora_adabudget:lru` | 1.652s | 2.194s | 2.215s | 1.955s |  0.00% |
| 47 | `lora_prefixtree:lru_k` | 1.652s | 2.302s | 2.322s | 1.994s |  0.00% |
| 48 | `lora_loose:s3fifo` | 1.655s | 2.250s | 2.275s | 1.991s |  0.00% |
| 49 | `lora_budget:lru` | 1.657s | 2.355s | 2.375s | 1.967s |  0.00% |
| 50 | `lora_freqweighted:lru_k` | 1.658s | 2.385s | 2.405s | 2.070s |  0.00% |
| 51 | `lora_correlated:lru` | 1.660s | 2.193s | 2.214s | 1.930s |  0.00% |
| 52 | `lora_budget:tinylfu` | 1.660s | 2.177s | 2.203s | 2.015s |  0.00% |
| 53 | `lora_costaware:sieve` | 1.660s | 2.187s | 2.207s | 1.906s |  0.00% |
| 54 | `lora_ghost:tinylfu` | 1.661s | 2.273s | 2.294s | 1.945s |  0.00% |
| 55 | `lora_hysteresis:lru_k` | 1.665s | 2.226s | 2.246s | 1.896s |  0.00% |
| 56 | `lora_prefixtree:s3fifo` | 1.667s | 2.201s | 2.222s | 1.902s |  0.00% |
| 57 | `lora_soft:s3fifo` | 1.679s | 2.281s | 2.300s | 2.012s |  0.00% |
| 58 | `lora_adabudget:s3fifo` | 1.680s | 2.230s | 2.250s | 2.112s |  0.00% |
| 59 | `lora_soft:sieve` | 1.684s | 2.222s | 2.235s | 2.019s |  0.00% |
| 60 | `lora_ghost:sieve` | 1.693s | 2.214s | 2.235s | 2.083s |  0.00% |
| 61 | `lora_correlated:tinylfu` | 1.695s | 2.244s | 2.270s | 2.015s |  0.00% |
| 62 | `lora_ghost:s3fifo` | 1.720s | 2.234s | 2.254s | 2.019s |  0.00% |
| 63 | `lora_adabudget:sieve` | 1.721s | 2.219s | 2.239s | 1.969s |  0.00% |
| 64 | `lora_hysteresis:tinylfu` | 1.739s | 2.312s | 2.332s | 2.061s |  0.00% |
| 65 | `lora_hysteresis:s3fifo` | 1.741s | 2.242s | 2.268s | 2.043s |  0.00% |
| 66 | `lru` | 1.794s | 3.353s | 3.373s | 2.215s |  0.00% |

## Scenario: `mixed_popularity`

_Zipfian (alpha=1.2) over 16 adapters — a few hot adapters dominate._

Sorted ascending by **TTFT P50** (lower is better). Hit rate / E2E shown for context.

| Rank | Policy | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | Hit Rate |
|-----:|:-------|---------:|---------:|---------:|--------:|---------:|
| 1 | `lora_hysteresis:lru` | 0.937s | 4.541s | 5.323s | 1.404s |  0.00% |
| 2 | `lora_budget:lru` | 0.985s | 3.603s | 4.652s | 1.513s |  0.00% |
| 3 | `tinylfu` | 1.029s | 4.204s | 4.476s | 1.421s |  0.00% |
| 4 | `lora_hysteresis:lru_k` | 1.041s | 3.486s | 5.254s | 1.510s |  0.00% |
| 5 | `lora_costaware:tinylfu` | 1.041s | 4.532s | 5.310s | 1.554s |  0.00% |
| 6 | `lora_hysteresis:tinylfu` | 1.047s | 4.669s | 5.227s | 1.674s |  0.00% |
| 7 | `lora_prefixtree:lru_k` | 1.058s | 4.485s | 5.322s | 1.495s |  0.00% |
| 8 | `lora_soft:lru` | 1.061s | 4.803s | 5.756s | 1.420s |  0.00% |
| 9 | `lora_correlated:lru_k` | 1.062s | 4.734s | 5.242s | 1.502s |  0.00% |
| 10 | `lora_correlated:s3fifo` | 1.066s | 4.231s | 4.662s | 1.581s |  0.00% |
| 11 | `lora_loose:tinylfu` | 1.071s | 4.663s | 5.256s | 1.559s |  0.00% |
| 12 | `lora_budget:lru_k` | 1.071s | 4.181s | 4.504s | 1.392s |  0.00% |
| 13 | `lora_adabudget:tinylfu` | 1.076s | 4.087s | 5.225s | 1.438s |  0.00% |
| 14 | `lora_costaware:sieve` | 1.078s | 4.316s | 4.706s | 1.447s |  0.00% |
| 15 | `lora_position:lru` | 1.080s | 4.741s | 5.152s | 1.547s |  0.00% |
| 16 | `lora_hysteresis:s3fifo` | 1.081s | 4.517s | 5.548s | 1.453s |  0.00% |
| 17 | `sieve` | 1.085s | 3.625s | 5.331s | 1.669s |  0.00% |
| 18 | `lora_soft:s3fifo` | 1.089s | 4.612s | 5.279s | 1.448s |  0.00% |
| 19 | `lora_budget:s3fifo` | 1.091s | 4.270s | 4.479s | 1.414s |  0.00% |
| 20 | `lora_adabudget:lru` | 1.092s | 4.820s | 5.429s | 1.555s |  0.00% |
| 21 | `s3fifo` | 1.093s | 4.528s | 5.915s | 1.477s |  0.00% |
| 22 | `lora_adabudget:sieve` | 1.094s | 5.087s | 5.269s | 1.588s |  0.00% |
| 23 | `lora_prefixtree:s3fifo` | 1.096s | 4.831s | 5.485s | 1.541s |  0.00% |
| 24 | `lora_position:tinylfu` | 1.098s | 4.479s | 5.310s | 1.531s |  0.00% |
| 25 | `lora_costaware:lru` | 1.100s | 4.696s | 5.539s | 1.611s |  0.00% |
| 26 | `lora_tight:s3fifo` | 1.101s | 3.484s | 5.609s | 1.564s |  0.00% |
| 27 | `lora_ghost:lru` | 1.101s | 4.633s | 5.553s | 1.505s |  0.00% |
| 28 | `lora_costaware:s3fifo` | 1.103s | 4.467s | 5.958s | 1.563s |  0.00% |
| 29 | `lora_soft:lru_k` | 1.104s | 4.575s | 5.412s | 1.494s |  0.00% |
| 30 | `lora_budget:sieve` | 1.104s | 4.747s | 5.653s | 1.562s |  0.00% |
| 31 | `lora_loose:lru` | 1.106s | 4.788s | 5.308s | 1.559s |  0.00% |
| 32 | `lora_ghost:s3fifo` | 1.106s | 4.478s | 5.149s | 1.538s |  0.00% |
| 33 | `lora_position:sieve` | 1.106s | 4.484s | 5.226s | 1.541s |  0.00% |
| 34 | `lora_loose:sieve` | 1.107s | 4.111s | 5.440s | 1.482s |  0.00% |
| 35 | `lora_prefixtree:tinylfu` | 1.107s | 3.876s | 5.408s | 1.638s |  0.00% |
| 36 | `lora_tight:tinylfu` | 1.108s | 4.760s | 5.287s | 1.638s |  0.00% |
| 37 | `lora_soft:sieve` | 1.108s | 3.602s | 5.222s | 1.677s |  0.00% |
| 38 | `lora_soft:tinylfu` | 1.108s | 4.481s | 5.303s | 1.538s |  0.00% |
| 39 | `lru` | 1.110s | 4.685s | 5.701s | 1.625s |  0.00% |
| 40 | `lora_correlated:lru` | 1.110s | 4.527s | 5.511s | 1.729s |  0.00% |
| 41 | `lora_adabudget:lru_k` | 1.111s | 4.169s | 5.466s | 1.664s |  0.00% |
| 42 | `lora_tight:lru_k` | 1.112s | 4.171s | 4.541s | 1.625s |  0.00% |
| 43 | `lora_twolevel` | 1.115s | 4.276s | 4.760s | 1.441s |  0.00% |
| 44 | `lora_hysteresis:sieve` | 1.115s | 4.229s | 4.844s | 1.498s |  0.00% |
| 45 | `lora_loose:s3fifo` | 1.122s | 4.720s | 5.175s | 1.646s |  0.00% |
| 46 | `lora_prefixtree:lru` | 1.130s | 4.886s | 5.447s | 1.629s |  0.00% |
| 47 | `lora_budget:tinylfu` | 1.134s | 5.045s | 5.149s | 1.607s |  0.00% |
| 48 | `lora_tight:lru` | 1.136s | 4.618s | 5.156s | 1.516s |  0.00% |
| 49 | `lora_ghost:tinylfu` | 1.136s | 4.538s | 5.320s | 1.571s |  0.00% |
| 50 | `lora_freqweighted:tinylfu` | 1.143s | 3.663s | 5.168s | 1.568s |  0.00% |
| 51 | `lora_position:lru_k` | 1.144s | 4.651s | 5.169s | 1.561s |  0.00% |
| 52 | `lora_tight:sieve` | 1.145s | 4.528s | 5.191s | 1.643s |  0.00% |
| 53 | `lora_correlated:sieve` | 1.148s | 4.562s | 5.140s | 1.588s |  0.00% |
| 54 | `lora_adabudget:s3fifo` | 1.148s | 3.649s | 5.148s | 1.638s |  0.00% |
| 55 | `lora_position:s3fifo` | 1.148s | 4.659s | 5.229s | 1.441s |  0.00% |
| 56 | `lora_freqweighted:sieve` | 1.159s | 4.647s | 5.209s | 1.554s |  0.00% |
| 57 | `lora_freqweighted:s3fifo` | 1.159s | 4.623s | 5.208s | 1.605s |  0.00% |
| 58 | `lora_prefixtree:sieve` | 1.164s | 3.602s | 5.168s | 1.677s |  0.00% |
| 59 | `lora_loose:lru_k` | 1.170s | 3.611s | 5.378s | 1.571s |  0.00% |
| 60 | `lora_ghost:lru_k` | 1.171s | 3.479s | 5.215s | 1.691s |  0.00% |
| 61 | `lora_costaware:lru_k` | 1.179s | 3.811s | 5.521s | 1.677s |  0.00% |
| 62 | `lora_freqweighted:lru` | 1.190s | 4.052s | 4.578s | 1.590s |  0.00% |
| 63 | `lora_freqweighted:lru_k` | 1.192s | 4.615s | 5.250s | 1.501s |  0.00% |
| 64 | `lru_k` | 1.212s | 4.762s | 5.336s | 1.572s |  0.00% |
| 65 | `lora_correlated:tinylfu` | 1.224s | 4.492s | 5.286s | 1.597s |  0.00% |
| 66 | `lora_ghost:sieve` | 1.236s | 4.071s | 4.908s | 1.531s |  0.00% |

## Overall ranking — avg TTFT-P50 rank

Average rank across the **two non-degenerate scenarios** (`adapter_thrashing` + `mixed_popularity`). Lower is better. `adapter_locality` is excluded because every policy's hit rate is 0% there.

| Rank | Policy | Avg Rank | thrashing | mixed |
|-----:|:-------|---------:|:---------:|:-----:|
| 1 | `lora_prefixtree:lru_k` | 4.5 | 2 | 7 |
| 2 | `lora_hysteresis:lru_k` | 11.5 | 19 | 4 |
| 3 | `lora_hysteresis:lru` | 16.0 | 31 | 1 |
| 4 | `lora_prefixtree:s3fifo` | 17.0 | 11 | 23 |
| 5 | `lora_budget:lru` | 17.0 | 32 | 2 |
| 6 | `lora_tight:tinylfu` | 18.5 | 1 | 36 |
| 7 | `lora_costaware:lru` | 19.5 | 14 | 25 |
| 8 | `lora_soft:tinylfu` | 20.5 | 3 | 38 |
| 9 | `lora_prefixtree:tinylfu` | 21.5 | 8 | 35 |
| 10 | `lora_soft:s3fifo` | 21.5 | 25 | 18 |
| 11 | `lora_soft:lru_k` | 22.0 | 15 | 29 |
| 12 | `s3fifo` | 22.0 | 23 | 21 |
| 13 | `tinylfu` | 22.0 | 41 | 3 |
| 14 | `lora_budget:sieve` | 23.5 | 17 | 30 |
| 15 | `lora_adabudget:lru_k` | 24.0 | 7 | 41 |
| 16 | `lora_budget:s3fifo` | 24.0 | 29 | 19 |
| 17 | `lora_costaware:sieve` | 24.0 | 34 | 14 |
| 18 | `lora_position:sieve` | 24.5 | 16 | 33 |
| 19 | `lora_correlated:s3fifo` | 26.0 | 42 | 10 |
| 20 | `lora_soft:lru` | 26.5 | 45 | 8 |
| 21 | `sieve` | 27.0 | 37 | 17 |
| 22 | `lora_position:lru` | 27.0 | 39 | 15 |
| 23 | `lora_twolevel` | 28.0 | 13 | 43 |
| 24 | `lora_adabudget:tinylfu` | 28.0 | 43 | 13 |
| 25 | `lora_soft:sieve` | 28.5 | 20 | 37 |
| 26 | `lora_loose:tinylfu` | 28.5 | 46 | 11 |
| 27 | `lora_budget:lru_k` | 30.5 | 49 | 12 |
| 28 | `lora_prefixtree:sieve` | 31.0 | 4 | 58 |
| 29 | `lora_correlated:sieve` | 31.0 | 9 | 53 |
| 30 | `lora_freqweighted:s3fifo` | 31.5 | 6 | 57 |

## Recommendations

- **adapter_thrashing**: best TTFT P50 = `lora_tight:tinylfu` (1.220s, hit_rate=0.00%).
- **adapter_locality**: best TTFT P50 = `lora_prefixtree:lru` (1.586s, hit_rate=0.00%).
- **mixed_popularity**: best TTFT P50 = `lora_hysteresis:lru` (0.937s, hit_rate=0.00%).