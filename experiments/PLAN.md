# 3B 实验计划

7B 用同一套命令和数据，换模型路径，由别人跑。这里只排 3B。

模型：`/HDDDATA/phx/tmp/omniselect/models/Qwen2.5-Omni-3B`。

协议：`OmniZip/eval/run_bench.py`。VideoMME 512 帧，DailyOmni 和 WorldSense 128 帧。像素预算 `nframes * 128 * 28 * 28`。保留率是最终留下的 token 占该模态全部 token 的比例。OmniDraft 的草稿比例是全部 token 的 15%，探针层 12，λ=0.5，代表缺口，完整重跑。

前 300 题是开发集，不写进主表。主表用全量。消融用第 301 题起的 200 题。

## 主表命令

| 行 | 命令要点 |
|---|---|
| Full Tokens | `--method full` |
| Random 55% | `--method random --keep_ratio 0.55` |
| DyCoke 50% | `--method dycoke --keep_ratio 0.5` |
| OmniZip 45% | `--method omnizip --rho 0.55 --keep_ratio 0.45` |
| OmniZip 30% | `--method omnizip --rho 0.70 --keep_ratio 0.30` |
| OmniSelect 45% | `--method omniselect --keep_ratio 0.45` |
| OmniSelect 30% | `--method omniselect --keep_ratio 0.30` |
| OmniDraft 45% | `--method omnidraft --boost gap --draft_ratio 0.15 --probe_layer 12 --lam 0.5 --gate 0 --exec rerun --keep_ratio 0.45` |
| OmniDraft 30% | 同上，`--keep_ratio 0.30` |

OmniZip 的 ρ 是剪掉的比例，所以 45% 保留对应 ρ=0.55，30% 保留对应 ρ=0.70。`contextual_ratio` 为 0，`g=3`。

效率表里的 DyCoke 写成了 55%，主表和对照图是 50%。跑主表时按 50%，这个 50% 是视频和音频各自留下的 token 占全部 token 的比例。DyCoke 先整段留下每一对时间片里的第一片，第二片只补足剩余预算；`--keep_ratio 0.5` 时第二片不留，整体正好一半。

VideoMME 要报 Short / Medium / Long。现在的 `pack1000` 只有 short 和 medium，全量划分用 parquet：`data/VideoMME/videomme/test-00000-of-00001.parquet`。长视频文件已经在 `data/VideoMME/videos/data/`。

DailyOmni 的六类用 `qa.json` 的 `Type` 字段：Context understanding、Event Sequence、AV Event Alignment、Comparative、Inference、Reasoning。

WorldSense 的类别用标注里的 domain 字段，和表头对齐后再汇总。FLOPs 单独测，不和准确率混在一次跑里。

## 消融

开发集之外，每个数据集 200 题（offset 300，limit 200）。帧数和主表相同。相对完整方法只改一处。

| 行 | 改动 |
|---|---|
| 完整方法 | 上面的 OmniDraft 30% |
| 去掉问题 attention | `--lam 0` |
| 打乱需求 | `--signal shuffle` |
| 不减空问题 | `--null_sub 0` |
| 去掉代表缺口 | `--boost ""` |

敏感性另做，不放进这张表。

## 主结果表

\subsection{Experiments}

\begin{table*}[t]
\centering
\caption{Main results on VideoMME and DailyOmni.
Accuracy is reported in percentages.}
\label{tab:main_results}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}lc*{11}{c}@{}}
\toprule
\multirow{2}{*}{Method}
& \multirow{2}{*}{\makecell{Retained\\Ratio}}
& \multicolumn{4}{c}{VideoMME ($\uparrow$)}
& \multicolumn{7}{c}{DailyOmni ($\uparrow$)} \\
\cmidrule(lr){3-6}
\cmidrule(lr){7-13}
& & Short & Medium & Long & Avg.
& Con. & Event & AV Event & Com. & Inf. & Rea. & Avg. \\
\midrule

\multicolumn{13}{c}{\textit{Qwen2.5-Omni-3B}} \\
\midrule
Full Tokens          & 100\% & & & & & & & & & & & \\
Random               & 55\%  & & & & & & & & & & & \\
DyCoke (V\&A)        & 50\%  & & & & & & & & & & & \\
OmniZip              & 45\%  & & & & & & & & & & & \\
OmniZip              & 30\%  & & & & & & & & & & & \\
OmniSelect           & 45\%  & & & & & & & & & & & \\
OmniSelect           & 30\%  & & & & & & & & & & & \\
\textbf{OmniDraft}    & 45\%  & & & & & & & & & & & \\
\textbf{OmniDraft}    & 30\%  & & & & & & & & & & & \\
\midrule

\multicolumn{13}{c}{\textit{Qwen2.5-Omni-7B}} \\
\midrule
Full Tokens          & 100\% & & & & & & & & & & & \\
Random               & 55\%  & & & & & & & & & & & \\
DyCoke (V\&A)        & 50\%  & & & & & & & & & & & \\
OmniZip              & 45\%  & & & & & & & & & & & \\
OmniZip              & 30\%  & & & & & & & & & & & \\
OmniSelect           & 45\%  & & & & & & & & & & & \\
OmniSelect           & 30\%  & & & & & & & & & & & \\
\textbf{OmniDraft}    & 45\%  & & & & & & & & & & & \\
\textbf{OmniDraft}    & 30\%  & & & & & & & & & & & \\
\bottomrule
\end{tabular}
}
\end{table*}

\begin{table*}[t]
\centering
\caption{Performance comparison on WorldSense.
All category scores and the average accuracy are reported in percentages.}
\label{tab:worldsense}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}lc*{10}{c}@{}}
\toprule
Method
& \makecell{Retained\\Ratio}
& \makecell{FLOPs\\(T)}
& \makecell{Tech \&\\Science}
& \makecell{Culture \&\\Politics}
& \makecell{Daily\\Life}
& \makecell{Film \&\\TV}
& Performance
& Games
& Sports
& Music
& Avg. \\
\midrule

\multicolumn{12}{c}{\textit{Qwen2.5-Omni-3B}} \\
\midrule
Full Tokens          & 100\% & & & & & & & & & & \\
Random               & 55\%  & & & & & & & & & & \\
DyCoke (V\&A)        & 50\%  & & & & & & & & & & \\
OmniZip              & 45\%  & & & & & & & & & & \\
OmniZip              & 30\%  & & & & & & & & & & \\
OmniSelect           & 45\%  & & & & & & & & & & \\
OmniSelect           & 30\%  & & & & & & & & & & \\
\textbf{OmniDraft}    & 45\%  & & & & & & & & & & \\
\textbf{OmniDraft}    & 30\%  & & & & & & & & & & \\
\midrule

\multicolumn{12}{c}{\textit{Qwen2.5-Omni-7B}} \\
\midrule
Full Tokens          & 100\% & & & & & & & & & & \\
Random               & 55\%  & & & & & & & & & & \\
DyCoke (V\&A)        & 50\%  & & & & & & & & & & \\
OmniZip              & 45\%  & & & & & & & & & & \\
OmniZip              & 30\%  & & & & & & & & & & \\
OmniSelect           & 45\%  & & & & & & & & & & \\
OmniSelect           & 30\%  & & & & & & & & & & \\
\textbf{OmniDraft}    & 45\%  & & & & & & & & & & \\
\textbf{OmniDraft}    & 30\%  & & & & & & & & & & \\
\bottomrule
\end{tabular}
}
\end{table*}

\begin{table*}[t]
\centering
\caption{Efficiency comparison on Qwen2.5-Omni.}
\label{tab:efficiency}
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}lcccc@{}}
\toprule
Method
& \makecell{GPU Mem.\\(GB) $\downarrow$}
& \makecell{Prefill Time\\(ms) $\downarrow$}
& \makecell{Acc.\\(\%) $\uparrow$}
& \makecell{Latency per Example\\(s) $\downarrow$} \\
\midrule

\multicolumn{5}{c}{\textit{Qwen2.5-Omni-3B}} \\
\midrule
Full Tokens             & & & & \\
DyCoke (55\%)           & & & & \\
\textbf{Ours (45\%)}    & & & & \\
\textbf{Ours (30\%)}    & & & & \\
\midrule

\multicolumn{5}{c}{\textit{Qwen2.5-Omni-7B}} \\
\midrule
Full Tokens             & & & & \\
DyCoke (55\%)           & & & & \\
\textbf{Ours (45\%)}    & & & & \\
\textbf{Ours (30\%)}    & & & & \\
\bottomrule
\end{tabular}
}
\end{table*}
