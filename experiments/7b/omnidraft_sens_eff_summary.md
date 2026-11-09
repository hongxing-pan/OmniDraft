# Sensitivity and efficiency, Qwen2.5-Omni-7B

Sensitivity: 30% kept, 128 frames, full splits. Probe time starts at the draft
input and ends when the shared demand signal is ready; E2E adds media reading.

## Sensitivity: draft token percentage

| Draft % | WorldSense Acc. | Probe | E2E | DailyOmni Acc. | Probe | E2E |
|---|---:|---:|---:|---:|---:|---:|
| 5 | 43.7 (1387/3172) | 96 ms | 5.50 s | 58.3 (698/1197) | 94 ms | 3.92 s |
| 10 | 43.8 (1390/3172) | 106 ms | 5.49 s | 57.3 (686/1197) | 99 ms | 3.89 s |
| 15 | 44.0 (1395/3172) | 124 ms | 5.48 s | 57.5 (688/1197) | 117 ms | 3.89 s |
| 20 | 43.8 (1389/3172) | 143 ms | 5.47 s | 57.6 (689/1197) | 131 ms | 3.88 s |

## Sensitivity: probe depth

| Depth | WorldSense Acc. | Probe | E2E | DailyOmni Acc. | Probe | E2E |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 43.3 (1372/3172) | 40 ms | 5.46 s | 58.3 (698/1197) | 40 ms | 3.86 s |
| 8 | 42.3 (1343/3172) | 67 ms | 5.48 s | 57.2 (685/1197) | 67 ms | 3.87 s |
| 12 | 43.7 (1387/3172) | 96 ms | 5.50 s | 58.3 (698/1197) | 94 ms | 3.92 s |
| 16 | 43.8 (1388/3172) | 123 ms | 5.53 s | 57.7 (691/1197) | 121 ms | 3.92 s |

## Sensitivity: demand mixing coefficient

| lambda | WorldSense Acc. | DailyOmni Acc. |
|---|---:|---:|
| 0.00 | 43.7 (1387/3172) | 57.0 (682/1197) |
| 0.25 | 43.6 (1384/3172) | 57.6 (690/1197) |
| 0.50 | 43.7 (1387/3172) | 58.3 (698/1197) |
| 0.75 | 43.1 (1368/3172) | 57.5 (688/1197) |
| 1.00 | 42.6 (1352/3172) | 57.3 (686/1197) |

## Efficiency (30 fixed questions, 3 repeats, batch 1)

Acc. is the full-split main-table number; DyCoke is blank because the timing run
uses 55% while the main table uses 50%.

| Benchmark | Method | GPU Mem. (GB) | LLM Prefill (ms) | Acc. | Latency (s) | E2E (s) | ok/attempted |
|---|---|---:|---:|---:|---:|---:|---:|
| WorldSense | Full Tokens | 25.46 | 984 | 46.0 | 2.825 | 6.434 | 90/90 |
| WorldSense | DyCoke (V&A, 55%) | 23.50 | 509 |  | 2.281 | 5.934 | 90/90 |
| WorldSense | OmniSelect (45%) | 23.18 | 418 | 43.4 | 46.181 | 69.801 | 90/90 |
| WorldSense | OmniSelect (30%) | 22.53 | 291 | 41.2 | 44.550 | 70.571 | 90/90 |
| WorldSense | OmniDraft (45%) | 23.05 | 412 | 44.8 | 3.735 | 7.101 | 90/90 |
| WorldSense | OmniDraft (30%) | 22.50 | 285 | 43.7 | 3.547 | 6.909 | 90/90 |
| DailyOmni | Full Tokens | 24.66 | 761 | 62.5 | 2.394 | 4.054 | 90/90 |
| DailyOmni | DyCoke (V&A, 55%) | 23.04 | 411 |  | 2.182 | 3.850 | 90/90 |
| DailyOmni | OmniSelect (45%) | 22.82 | 343 | 58.4 | 48.699 | 59.261 | 90/90 |
| DailyOmni | OmniSelect (30%) | 22.28 | 267 | 55.3 | 48.842 | 59.738 | 90/90 |
| DailyOmni | OmniDraft (45%) | 22.68 | 337 | 59.4 | 2.887 | 4.905 | 90/90 |
| DailyOmni | OmniDraft (30%) | 22.38 | 251 | 58.3 | 2.874 | 4.846 | 90/90 |
| VideoMME | Full Tokens | 36.40 | 4446 | 66.4 | 9.734 | 26.877 | 90/90 |
| VideoMME | DyCoke (V&A, 55%) | 29.71 | 2066 |  | 7.247 | 24.537 | 90/90 |
| VideoMME | OmniSelect (45%) | 28.69 | 1620 | 65.3 | 71.722 | 106.103 | 90/90 |
| VideoMME | OmniSelect (30%) | 26.49 | 1015 | 65.1 | 71.420 | 105.116 | 90/90 |
| VideoMME | OmniDraft (45%) | 28.22 | 1651 | 66.8 | 16.928 | 96.498 | 90/90 |
| VideoMME | OmniDraft (30%) | 26.95 | 1024 | 66.6 | 16.270 | 96.810 | 90/90 |
