# OMINI-DRAFT

Qwen2.5-Omni-3B 上的问题相关 token 压缩。选择和拼接执行与 AutoDL 上的 `OmniZip_dev` 一致。

## 选择

预算 30%。视频先按时空锚点贪心排好顺序，音频按只留原始锚点的合并树。草稿取这个顺序的前一半。系统文本全部保留。

草稿带着原来的三维位置走 Thinker 前 12 层。真问题和固定空问题 “What is happening in the video?” 各读一次缓存。

\[
e_j = \max(0, a_q(j) - a_\varnothing(j))
\]

按视频帧分成时间片，音频、视频各自归一化后平均，再用 \([1/4, 1/2, 1/4]\) 平滑。剩下的名额是

\[
d_t = (1-\lambda) u_t + \lambda R_t
\]

最大余数法取整。片内仍按锚点原顺序补。\(\lambda = 0.5\)。门控默认关闭。留下的是原始 embedding、原始顺序、原始 TMRoPE。

## 执行

`exec=rerun`：探针只用完就丢，保留集合从第 0 层重跑 36 层。

`exec=splice`：草稿在前 \(L\) 层只看草稿。新增 token 按原始下标读取草稿 KV。从第 \(L\) 层起按原始顺序一起算。\(r > 0\) 时，按第 0 层 hidden 的偏差重算比例为 \(r\) 的草稿 token。

## 文件

这些文件放进 OmniZip 的 `omnizip/` 即可，包名保持 `omnizip`。

- `omnidraft.py`：探针、配额、重跑、reuse4、拼接
- `fastsel.py`：带配额的时空锚点
- `omnianchor.py`：音频锚点合并
- `omnispan.py`：时间坐标和 mel 对齐

依赖 PyTorch、Transformers、Numba。模型是 Qwen2.5-Omni-3B。
