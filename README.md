# OmniDraft

Qwen2.5-Omni 上的问题相关音视频 token 压缩。项目名是 OmniDraft。OmniZip、OmniSelect、DyCoke、Random 都是对照方法。

## 当前方法

预算按模态分开。草稿是该模态**全部 token** 的 15%，并且不超过最终预算。视频草稿还至少覆盖每个时间片一个 token。探针层是第 12 层。λ = 0.5。门控关闭。执行是完整重跑，不拼接。

草稿带着原来的三维位置走 Thinker 前 12 层。真问题和固定空问题 “What is happening in the video?” 各读一次缓存。

\[
e_j = \max(0, a_q(j) - a_\varnothing(j))
\]

按视频帧分成时间片。音频和视频各自归一化后平均，再用 \([1/4, 1/2, 1/4]\) 平滑，得到共享需求 \(R_t\)。

视频再乘代表缺口：草稿放好之后，时间片里还没选中的 token 离草稿邻居越远，这一片的视频配额越高。音频配额仍用 \(R_t\)。

剩下的名额是

\[
d_t = (1-\lambda) u_t + \lambda \tilde R_t
\]

最大余数法取整。片内仍按原来的时空锚点补。λ = 0 时，选择和时空锚点逐题相同。留下的是原始 embedding、原始顺序、原始 TMRoPE。探针缓存用完就丢，最终生成从第 0 层重跑。

## 对照

| 方法 | 角色 |
|---|---|
| OmniZip | baseline |
| OmniSelect | baseline |
| DyCoke | baseline |
| Random | baseline |

7B 怎么跑见 `experiments/RUN_7B.md`。3B 主表和消融见 `experiments/PLAN.md`。

## 文件

这些文件放进 OmniZip 的 `omnizip/`，包名保持 `omnizip`。

- `omnidraft.py`：草稿、探针、代表缺口、配额、完整重跑
- `fastsel.py`：带配额的时空锚点
- `omnianchor.py`：音频锚点合并
- `omnispan.py`：时间坐标和 mel 对齐

依赖见 `requirements-test.txt` 和 `.env.example`。`modeling_qwen2_5_omni.py` 和 `qwen-omni-utils/` 都在仓库里。视频数据不在仓库里，路径写在 `.env.example`。

跑之前：

```bash
export PYTHONPATH=$PWD:$PWD/qwen-omni-utils/src
export OSELECT_DATA=/path/to/data
export PATH=/path/to/ffmpeg:$PATH
```
