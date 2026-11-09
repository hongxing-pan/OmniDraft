"""Shared pieces for the sensitivity and efficiency runs.

Both runs need the same three things: load the 7B thinker once, turn one sample
into the exact tensors eval/run_bench.py uses, and run one method config on
those tensors. Timing definitions follow OmniDraft_实验计划.md:

  probe_ms  draft forward + both suffix forwards + attention read + contrast +
            per-slice aggregation + smoothing, i.e. until the shared demand
            signal is ready. Read from omnizip.omnidraft.LAST["probe_sig_sec"].
  prefill   the final answer pass only: the single Qwen2_5OmniThinkerTextModel
            forward over the kept sequence, taken from a wrapper on that module.
  latency   inputs already on device -> answer text ready.
  e2e       before the raw media is read -> answer text ready.

`prep` is measured separately, so a caller that shares one decode across several
method configs can still report a per-config e2e as prep_s + latency_s. All
timing is wall clock, synchronised at the request boundaries.
"""
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "qwen-omni-utils" / "src"), str(ROOT / "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration  # noqa: E402
from transformers import Qwen2_5OmniProcessor  # noqa: E402
from qwen_omni_utils import process_mm_info  # noqa: E402

import run_bench as rb  # noqa: E402

SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
          "capable of perceiving auditory and visual inputs, as well as generating text and speech.")

BASE_CFG = {
    "rho_audio": 0.55, "rho_video": 0.55, "g": 3, "contextual_ratio": 0.0,
    "audio_mode": "span", "video_mode": "span", "anchor_mode": "full",
    "compress": "both", "read_mode": "usage", "budget_mode": "shared", "emit": "merge",
    "fac_mode": "full", "pair_mode": "joint", "witness_mode": "pair",
    "query_mode": "both", "refine_mode": "grad", "late_mode": "raw",
    "exec": "rerun", "splice_l": 12, "splice_r": 0.0, "signal": "quota",
    "null_sub": 1, "vis": "flash", "rounds": 1, "round_l1": 4,
}


def load(model_path, max_new_tokens=8):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    kit = Kit(model, processor, max_new_tokens)
    return kit


class Kit:
    def __init__(self, model, processor, max_new_tokens=8):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.prefill_ms = None
        self._arm = False
        self._install_prefill_timer()
        self._base = {
            "null_question_ids": processor.tokenizer("What is happening in the video?", add_special_tokens=False).input_ids,
            "special_ids": list(processor.tokenizer.all_special_ids),
            "im_end_id": processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "option_token_ids": [self._first_tok(c) for c in "ABCDEFGH"],
        }
        self._feat_cache = {}

    def _first_tok(self, ch):
        tok = self.processor.tokenizer.encode(ch, add_special_tokens=False)
        return int(tok[0]) if len(tok) == 1 else -1

    def _install_prefill_timer(self):
        """Time only the full-sequence call into the thinker text model."""
        tm = self.model.thinker.model
        inner = tm.forward
        kit = self

        def timed(*args, **kwargs):
            emb = kwargs.get("inputs_embeds")
            n = emb.shape[1] if emb is not None else 1
            if not kit._arm or n <= 1:
                return inner(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = inner(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            kit.prefill_ms = (time.perf_counter() - t0) * 1000.0
            kit._arm = False   # only the first full-length call is the prefill
            return out

        tm.forward = timed

    # ---- inputs -----------------------------------------------------------
    def conversation(self, sample, nframes):
        opts = sample["options"]
        if isinstance(opts, list):
            opts = "\n".join(str(x) for x in opts)
        prompt = (f"{sample['question']}\nOptions:\n{opts}\n"
                  "Answer with the option's letter from the given choices directly.")
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "video", "video": sample["video"], "nframes": nframes,
                 "min_pixels": 128 * 28 * 28, "max_pixels": 768 * 28 * 28,
                 "total_pixels": nframes * 128 * 28 * 28},
                {"type": "text", "text": prompt},
            ]},
        ]

    def prep(self, sample, nframes):
        """Returns (inputs, audios, images, videos, prep_sec, short)."""
        conv = self.conversation(sample, nframes)
        try:
            return self._prep_once(conv) + (False,)
        except ValueError as e:
            if "nframes" not in str(e):
                raise
            conv[1]["content"][0].pop("nframes", None)
            return self._prep_once(conv) + (True,)

    def _prep_once(self, conv):
        t0 = time.perf_counter()
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
        self.model.thinker.nframes = videos[0].shape[0]
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        return inputs, audios, images, videos, time.perf_counter() - t0

    # ---- methods ----------------------------------------------------------
    def cfg(self, **over):
        c = dict(BASE_CFG)
        c.update(self._base)
        c.update(over)
        return c

    def run(self, cfg, inputs, videos, audios, question, cache_key=None):
        """One method config on already prepared inputs. Returns (resp, ok_pred, gen_sec, stats)."""
        cfg = dict(cfg)
        cfg.update(self._base)
        self.model.thinker.omnizip_config = cfg
        self.model.thinker._omniselect_need = None
        self.last_expert_s = 0.0
        if cfg.get("method") == "omniselect":
            from omnizip.omniselect_proto import prepare_need
            _te = time.perf_counter()
            if cache_key in self._feat_cache:
                self._feat_cache.move_to_end(cache_key)
            self.model.thinker._omniselect_need = prepare_need(
                question, videos, audios, inputs["input_ids"],
                self.model.thinker.config.audio_token_id,
                self.model.thinker.config.video_token_id,
                float(cfg.get("keep_ratio", 0.45)),
                feat_cache=self._feat_cache, cache_key=cache_key,
            )
            while len(self._feat_cache) > 16:
                self._feat_cache.popitem(last=False)
            self.last_expert_s = time.perf_counter() - _te
        self.prefill_ms = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._arm = True
        with torch.no_grad():
            cont = self.model.generate(**inputs, return_audio=False, do_sample=False,
                                       thinker_do_sample=False,
                                       max_new_tokens=self.max_new_tokens,
                                       use_audio_in_video=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_sec = time.perf_counter() - t0
        self._arm = False
        new = cont[0][inputs.input_ids.shape[1]:]
        resp = self.processor.tokenizer.decode(new, skip_special_tokens=True)
        stats = getattr(self.model.thinker, "_omnspan_last", None)
        return resp, rb.letter(resp), gen_sec, (stats or {}), int(new.numel())

    def counts(self, inputs):
        ids = inputs["input_ids"][0]
        return (int((ids == self.model.thinker.config.audio_token_id).sum()),
                int((ids == self.model.thinker.config.video_token_id).sum()))

    @staticmethod
    def peak_mem_gb(reset=True):
        if not torch.cuda.is_available():
            return 0.0
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return torch.cuda.max_memory_allocated() / 1024 ** 3
