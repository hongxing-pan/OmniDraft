# 跑 Qwen2.5-Omni-7B

3B 和 7B 用同一套评测。7B 只换模型路径和结果目录。不要和 3B 的分数混在一张表里比较不同机器的结果；同一张表里的 7B 行必须来自同一次、同一套代码。

仓库：https://github.com/hongxing-pan/OmniDraft

## 1. 代码放哪

本仓库是方法，不是完整的 Qwen 工程。评测脚本假定自己在 OmniZip 工程里面：

```text
OmniZip/
  eval/run_bench.py
  omnizip/          # 本仓库的 omnizip/*.py 覆盖到这里
  qwen-omni-utils/
```

把本仓库的 `omnizip/*.py` 复制到 OmniZip 的 `omnizip/`，把 `eval/run_bench.py` 复制到 OmniZip 的 `eval/`。`modeling_qwen2_5_omni.py` 必须能识别这些 `--method`：`full`、`random`、`dycoke`、`omnizip`、`omniselect`、`omnidraft`。上游原版 OmniZip 没有 `full` 和 `omniselect`，用我们正在跑 3B 的那份 `omnizip/modeling_qwen2_5_omni.py`。

OmniSelect 还依赖 AudioCLIP 权重和 NLTK。权重路径写在 `omnizip/omniselect_proto.py` 里。不跑 OmniSelect 可以先不管。

## 2. 环境

```bash
export OSELECT_DATA=/path/to/data
export OSELECT_FULL=1
export PYTHONPATH=/path/to/OmniZip:/path/to/OmniZip/qwen-omni-utils/src
export PATH=/path/to/ffmpeg:$PATH
export NLTK_DATA=/path/to/nltk_data
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

`OSELECT_FULL=1` 让 DailyOmni 读全量 `qa.json`。不设的话，如果数据目录里有 `pack1000/daily_qa.json`，只会跑前 1000 题。

ffmpeg 必须在 `PATH` 里，否则 mp4 的音频会报 `NoBackendError`。一张卡一个进程，用 `CUDA_VISIBLE_DEVICES` 指定。不要两个任务共用一张卡。

模型用 Hugging Face 的 `Qwen/Qwen2.5-Omni-7B`，或你已经下好的本地目录。加载是 bfloat16 自动精度，FlashAttention 2。7B 建议一张 80GB 卡。512 帧的 VideoMME 比 128 帧大约多四倍视频 token。

## 3. 数据目录

```text
$OSELECT_DATA/DailyOmni/qa.json
$OSELECT_DATA/DailyOmni/Videos/{video_id}/{video_id}_video.mp4
$OSELECT_DATA/WorldSense/worldsense_qa.json
$OSELECT_DATA/WorldSense/videos/{video_id}.mp4
$OSELECT_DATA/VideoMME/videomme/test-00000-of-00001.parquet
$OSELECT_DATA/VideoMME/videos/data/{videoID}.mp4
```

全量规模：DailyOmni 1197 题，WorldSense 1662 个视频（约 3172 道题），VideoMME 2700 题。VideoMME 不要使用 `pack1000/videomme_1000.json`，那个文件在的话入口会只用前 1000 题。官方 parquet 才有 short / medium / long。

## 4. 帧数和像素

| 数据集 | `--dataset` | `--nframes` | `--limit` |
|---|---|---:|---:|
| DailyOmni | `daily` | 128 | 2000 |
| WorldSense | `ws` | 128 | 4000 |
| VideoMME | `vme` | 512 | 3000 |

像素预算是 `nframes * 128 * 28 * 28`，脚本里写死了，不要改。音频超过 300 秒会被官方预处理器截断，这是模型自己的上限。

## 5. 要跑的行

下面每条都换成你的模型路径。`--offset 0`。结果目录互相分开，脚本看到已有的成功 json 会跳过，中断后重跑同一条命令即可。`fail` 和 `oom` 的 json 会被重算。

公共参数：

```bash
PY=python
MODEL=/path/to/Qwen2.5-Omni-7B
ROOT=/path/to/OmniZip
OUT=/path/to/results/full7b
```

不压缩：

```bash
$PY $ROOT/eval/run_bench.py --dataset daily --method full \
  --model_path $MODEL --output_dir $OUT/full/daily --nframes 128 --limit 2000
```

Random，保留 55%：

```bash
--method random --keep_ratio 0.55
```

DyCoke，音视频都保留 50%：

```bash
--method dycoke --keep_ratio 0.5
```

OmniZip。这里的 `--rho` 是剪掉的比例，不是留下的比例。`contextual_ratio=0`，`g=3`，脚本默认就是这样。

| 留下 | 参数 |
|---|---|
| 45% | `--method omnizip --rho 0.55 --keep_ratio 0.45` |
| 30% | `--method omnizip --rho 0.70 --keep_ratio 0.30` |

OmniSelect。`--keep_ratio` 是留下的比例，脚本内部把剪枝比例设成 `1 - keep_ratio`。θ = 2。

| 留下 | 参数 |
|---|---|
| 45% | `--method omniselect --keep_ratio 0.45` |
| 30% | `--method omniselect --keep_ratio 0.30` |

OmniDraft。这是我们的方法，不是 baseline。

```bash
--method omnidraft \
  --boost gap \
  --draft_ratio 0.05 \
  --probe_layer 12 \
  --lam 0.5 \
  --gate 0 \
  --exec rerun \
  --keep_ratio 0.45
```

30% 那一行只把 `--keep_ratio` 改成 `0.30`。不要开 `splice`，不要改空问题。

三个数据集把 `--dataset`、`--nframes`、`--limit`、`--output_dir` 按第 4 节换掉。9 行 × 3 个数据集 = 27 个任务。

## 6. 参数是什么意思

| 参数 | 主表取值 | 含义 |
|---|---|---|
| `--keep_ratio` | 0.45 或 0.30；Random 0.55；DyCoke 0.50 | 每个模态最终留下的 token 占全部 token 的比例 |
| `--rho` | OmniZip：0.55 或 0.70 | 只给 OmniZip 用，表示剪掉多少 |
| `--draft_ratio` | 0.05 | 草稿占该模态全部 token 的比例，不是占保留预算的比例。视频另外至少每个时间片 1 个 |
| `--probe_layer` | 12 | 在第几层读草稿 attention |
| `--lam` | 0.5 | 剩余名额里按问题需求分配的比例。0 等于纯时空锚点 |
| `--boost gap` | 开 | 代表缺口，只改视频各时间片的配额 |
| `--gate` | 0 | 关闭按尖锐度自动调 λ |
| `--exec rerun` | rerun | 选完从第 0 层完整重跑 |
| `--null_text` | What is happening in the video? | 用来做减法的空问题 |
| `--null_sub` | 1 | 用真问题的 attention 减去空问题 |
| `--signal` | quota | 按时间片配额分配。主表不要改 |
| `--nframes` | 见第 4 节 | 均匀抽帧，不是原视频的帧率 |

## 7. 怎么算分数

每个题一个 json。`is_correct` 为 true 算对。`fail` 或 `oom` 不进分母，在表里注明题数。

```bash
python - << 'PY'
import json, os, sys
root = sys.argv[1]
n = ok = bad = 0
for f in os.listdir(root):
    if not f.endswith(".json"):
        continue
    o = json.load(open(os.path.join(root, f)))
    if o.get("fail") or o.get("oom"):
        bad += 1
        continue
    n += 1
    ok += int(bool(o.get("is_correct")))
print(ok, "/", n, "bad", bad)
PY
```

分类不要只报总准确率：

- VideoMME：用 parquet 里的 `duration`，分成 short、medium、long，再平均。
- DailyOmni：用 `qa.json` 的 `Type`。对应表头是 Context understanding、Event Sequence、AV Event Alignment、Comparative、Inference、Reasoning。
- WorldSense：用标注里的 domain，对齐 Tech & Science、Culture & Politics、Daily Life、Film & TV、Performance、Games、Sports、Music。

FLOPs、显存、prefill 时间和单题延迟另测，不要从这批准确率日志里估。效率表里的 DyCoke 若写 55%，主表仍按 50% 跑。

## 8. 不要跑的东西

主表不要跑消融，也不要跑 λ 扫描。那些是 3B 上的分析。7B 先交第 5 节的 27 个任务。前 300 题在 3B 上用来选过参数，7B 仍然跑全量，不要改成只跑后 200 题。
