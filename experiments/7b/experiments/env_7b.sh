# Source me before running any 7B benchmark row.
# DTK/ROCm runtime used by the local torch build.
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/hip/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/dtk-26.04/dcc/gcvm/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/opt/dtk
export ROCM_HOME=/opt/dtk
export DS_BUILD_OPS=0
export DS_BUILD_AIO=0
export DS_BUILD_FP_QUANTIZER=0

export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"

# Project and protocol.
export ROOT=/data/source_new02/olizhan/OmniDraft
export PYTHONPATH=$ROOT:$ROOT/qwen-omni-utils/src
export PY=python
export MODEL=/data/source_new02/olizhan/models/Qwen2.5-Omni-7B
export OSELECT_DATA=/data/source_new02/olizhan/data/omniselect_data
# Full DailyOmni qa.json and the official VideoMME parquet (2700 rows).
export OSELECT_FULL=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# ffmpeg is required or mp4 audio raises NoBackendError.
export PATH=/data/source_new02/olizhan/bin:$PATH

# OmniSelect only: AudioCLIP weights and code, plus NLTK tokenizer data.
# omniselect_proto.py hardcodes /HDDDATA/phx/tmp/omniselect, symlinked to these.
export NLTK_DATA=/data/source_new02/olizhan/nltk_data

OUT=/data/source_new02/olizhan/results/full7b
