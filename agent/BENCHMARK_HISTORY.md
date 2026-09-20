# Official benchmark history

Scores are hidden-workload geometric means. Failed runs are ineligible.

| Experiment | State | Score (tok/s) | Public 0 / 1 / 2 (tok/s) | Peak GB |
| --- | --- | ---: | --- | ---: |
| [baseline](results/baseline.json) | succeeded | 192.3 | 43.4 / 135.5 / 637.7 | 14.48 |
| [stage1](results/stage1.json) | latency_limit | — | 54.1 / 144.1 / 712.7 | 14.55 |
| [stage1_prefill](results/stage1_prefill.json) | succeeded | 118.6 | 25.5 / 88.4 / 376.9 | 17.05 |
| [stage2_graph](results/stage2_graph.json) | latency_limit | — | 95.2 / 168.6 / 777.7 | 17.77 |
| [stage3_norm](results/stage3_norm.json) | succeeded | 339.7 | 120.9 / 197.6 / 905.9 | 17.15 |
| [stage4_flash](results/stage4_flash.json) | succeeded | 600.5 | 152.5 / 333.3 / 1889.3 | 16.89 |
| [stage5_packed](results/stage5_packed.json) | succeeded | 609.7 | 155.4 / 337.3 / 1919.1 | 18.28 |
| [stage6_qkv](results/stage6_qkv.json) | succeeded | 740.3 | 192.6 / 379.9 / 2330.0 | 18.27 |
| [stage7_prefill](results/stage7_prefill.json) | succeeded | 747.8 | 192.6 / 393.6 / 2364.4 | 16.09 |
| [stage8_linear](results/stage8_linear.json) | succeeded | 753.6 | 193.5 / 398.1 / 2387.6 | 16.26 |
| [stage9_overlap](results/stage9_overlap.json) | succeeded | 775.6 | 197.3 / 398.4 / 2439.2 | 16.26 |
| [stage10_grouped](results/stage10_grouped.json) | succeeded | 787.3 | 199.1 / 403.3 / 2471.6 | 16.32 |
| [stage11_pointwise](results/stage11_pointwise.json) | unstable_timing | — | 201.0 / 410.7 / 2566.0 | 16.32 |
| [stage11a_primed](results/stage11a_primed.json) | succeeded | 813.4 | 205.8 / 415.5 / 2572.7 | 16.31 |
| [stage12_wide](results/stage12_wide.json) | succeeded | 815.5 | 209.2 / 421.9 / 2571.2 | 16.32 |
| [stage13_prefill_fused](results/stage13_prefill_fused.json) | incorrect_output | — | 218.3 / 450.9 / 2672.9 | 16.20 |
| [stage14_prefill_graph](results/stage14_prefill_graph.json) | succeeded | 811.9 | 213.5 / 411.3 / 2553.9 | 16.36 |
| [stage15_layouts](results/stage15_layouts.json) | succeeded | 819.3 | 224.1 / 414.7 / 2550.2 | 16.52 |
| [stage16_attention](results/stage16_attention.json) | succeeded | 872.5 | 220.3 / 435.7 / 2726.7 | 16.52 |
| [stage17_rotary](results/stage17_rotary.json) | succeeded | 889.0 | 223.5 / 460.2 / 2783.5 | 16.55 |
| [stage18_matvec](results/stage18_matvec.json) | succeeded | 875.6 | 228.9 / 456.3 / 2783.6 | 16.55 |
| [stage19_padding](results/stage19_padding.json) | succeeded | 879.8 | 228.9 / 457.9 / 2615.6 | 16.55 |
| [stage20_attention_tiles](results/stage20_attention_tiles.json) | succeeded | 916.7 | 235.1 / 472.7 / 2838.3 | 16.55 |
| [stage21_prefill_pointwise](results/stage21_prefill_pointwise.json) | succeeded | 925.0 | 234.8 / 477.5 / 2850.7 | 17.22 |
| [stage22_prefill_lt](results/stage22_prefill_lt.json) | succeeded | 900.4 | 229.8 / 465.7 / 2809.8 | 17.22 |
| [stage22a_decode_lt](results/stage22a_decode_lt.json) | succeeded | 881.9 | 230.3 / 461.0 / 2625.5 | 17.32 |
| [stage23_fused_gate_up](results/stage23_fused_gate_up.json) | succeeded | 913.0 | 232.0 / 469.0 / 2843.1 | 17.22 |
