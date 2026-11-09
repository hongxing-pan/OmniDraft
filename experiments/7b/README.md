# OmniDraft · Qwen2.5-Omni-7B 敏感性与效率结果

打包时间：2026-11-09
生成环境：8 × 64GB DCU，torch 2.7.1，transformers 4.57.6，batch size 1

## 目录

    omnidraft_tables_filled.tex          论文用表格（四张，已填数字）
    omnidraft_sens_eff_summary.md        同上内容的 markdown 版
    results/sensitivity/<dataset>/<config_id>/
        config.json                      该配置的完整参数
        predictions.jsonl                逐题记录：预测、对错、probe_ms、e2e_s、token 数
        summary_shard0..7.json           8 个分片的汇总
    results/efficiency_7b/<dataset>/
        manifest.json                    固定 30 题清单 + 3 题预热 + 抽样种子
        <method>/
            config.json
            measurements.jsonl           逐题逐遍的计时与显存
            summary.json                 逐题取中位数再平均后的结果
    experiments/                         本次使用的脚本
        eval_kit.py                     模型加载、预处理、分段计时
        run_sensitivity.py              11 个配置的敏感性扫描
        run_efficiency.py               6 方法 × 30 题 × 3 遍的效率测试
        run_sens_eff.sh                 两阶段调度
        report_sens_eff.py              合并分片、生成 tex
        env_7b.sh                       环境变量

## 口径

- 敏感性：最终保留 30%，128 帧，WorldSense 3172 题 / DailyOmni 1197 题全量。
- Probe (ms)：从草稿输入探针开始，到共享时间需求信号准备完成；含草稿前向、
  真问题与空问题两次后缀前向、注意力提取、对比、时间片汇总和平滑。
- E2E (s)：从读取原始媒体前，到答案文本就绪。敏感性中一题只解码一次，
  由 11 个配置共享，E2E 记为该题实测预处理时间 + 该配置自身延迟。
- 效率：6 方法 × 3 基准 × 30 题 × 3 遍，每遍独立重新解码，不复用 AudioCLIP
  特征、探针缓存或 keep mask；显存为单次请求的峰值 allocated。
- 效率表的 Acc. 引用同输入设置下的全量主表；DyCoke 留空，因为主表按 50% 跑，
  而计时按 55% 跑。
- 抽样种子 20261108。VideoMME 30 题 = short/medium/long 各 10；DailyOmni 6 类
  各 5；WorldSense 8 个 domain 轮转（4/4/4/4/4/4/3/3）。30 题均来自 30 个不同视频。

## 结果

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

