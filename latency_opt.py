"""
Batch = 1 ultra-low-latency optimization for LiquidAI/d1-omni-600M on Intel Iris Xe.
--------------------------------------------------------------------------------------
Builds on ``d1_iris_xe.D1OmniIrisXe`` (dynamic-shape OpenVINO trunk) and adds:

  1. Zero-copy tensor binding
     - Shared host memory via ``ov.Tensor(..., shared_memory=True)``.
     - Persistent USM-friendly input/output buffers bound once per shape.
     - A reused ``InferRequest`` per shape (no per-call allocation / dict building).

  2. Static shape bucket compilation
     - Static IR per (batch, sequence) shape. The standard sequence buckets are
       [64, 128, 256]; inputs are padded to the selected static length.
     - Padding is skipped when it would cost more than 12.5% extra work, in which
       case the length is snapped up to the next multiple of 16 (the same static
       compilation, just without a wasteful pad).

  3. Sub-module profiling
     - Tokenizer encoding vs Embeddings vs GPU Trunk vs Head, measured on the real
       ``system_one`` path with transparent timing wrappers.

  4. GPU decision head (extra)
     - The head is ~40% of end-to-end CPU time; an explicit, graph-convertible
       re-implementation is compiled to the same GPU. Original CPU head is kept
       as a fallback and as the correctness reference.

  5. Before/after benchmark + probability verification against ``d1_iris_xe.py``.

Usage:
    python latency_opt.py
    python latency_opt.py --iterations 20 --no-head-gpu
    python latency_opt.py --buckets 64,128,256 --device GPU
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from collections import OrderedDict

import numpy as np
import openvino as ov
import torch
import torch.nn.functional as F

from d1_iris_xe import CACHE_DIR, D1OmniIrisXe

BUCKETS = (64, 128, 256)
SNAP = 16
PAD_SLACK = 1.125
MAX_STATIC = 2048

STATE = (
    "Production Alert: Primary payment gateway returned HTTP 500 error code for 12 consecutive requests. "
    "Transactions failing. Customer checkout blocked. Severity automatically elevated."
)

QUESTIONS = {
    "is_urgent_outage": {
        "type": "noul",
        "instructions": "Is this a critical payment outage requiring immediate escalation?",
    },
    "routing_team": {
        "type": "choice",
        "instructions": "Which response team should receive this incident page?",
        "criteria": {
            "payments_sre": "Billing, payment gateways, banking APIs, core infra",
            "frontend_dev": "Web styling, CSS layout, client-side UI bugs",
            "marketing": "Customer acquisition, newsletters, promotion campaigns",
        },
    },
    "severity_score": {
        "type": "score",
        "instructions": "Assess business impact level",
        "criteria": ["Minor non-blocking glitch", "Degraded performance", "Immediate revenue loss"],
    },
}

SINGLE_QUESTION = {"is_urgent_outage": QUESTIONS["is_urgent_outage"]}

COMPILE_PROPS = {
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_DIR": CACHE_DIR,
    "NUM_STREAMS": "1",
    "PERFORMANCE_HINT_NUM_REQUESTS": "1",
}


def target_length(length: int, buckets=BUCKETS, snap: int = SNAP, slack: float = PAD_SLACK) -> int:
    """Smallest static length to compile: a standard bucket when padding is cheap, else a fine snap."""
    for bucket in buckets:
        if length <= bucket <= length * slack:
            return bucket
    step = snap if length <= buckets[-1] else 64
    return ((length + step - 1) // step) * step


def _convert_static(module: torch.nn.Module, example: tuple, shapes: list) -> ov.Model:
    """Trace ``module`` for one concrete shape and freeze that shape in the IR.

    OpenVINO's torch frontend derives the graph from ``example`` but leaves the input
    partial shapes dynamic. Freezing them to the traced values keeps every shape-derived
    constant (RoPE ranges, masks, slices) consistent, which is what makes the static
    kernels correct. Reshaping a model traced for a *different* length does not.
    """
    ov_model = ov.convert_model(module, example_input=example)
    for port, shape in zip(ov_model.inputs, shapes):
        port.get_node().set_partial_shape(ov.PartialShape(shape))
    ov_model.validate_nodes_and_infer_types()
    return ov_model


class _StaticSlot:
    """One compiled static model plus its persistent, shared input/output tensors."""

    def __init__(self, compiled: ov.CompiledModel, batch: int, length: int):
        self.compiled = compiled
        self.batch, self.length = batch, length
        self.request = compiled.create_infer_request()

        hidden = 1024
        self.h = np.zeros((batch, length, hidden), np.float32)
        self.pad = np.zeros((batch, length), np.bool_)
        self.offset = np.zeros((batch,), np.int64)

        self.h_tensor = ov.Tensor(self.h, shared_memory=True)
        self.pad_tensor = ov.Tensor(self.pad, shared_memory=True)
        self.offset_tensor = ov.Tensor(self.offset, shared_memory=True)
        for index, tensor in enumerate((self.h_tensor, self.pad_tensor, self.offset_tensor)):
            self.request.set_input_tensor(index, tensor)

        self.out_tensor = ov.Tensor(ov.Type.f32, [batch, length, hidden])
        self.request.set_output_tensor(0, self.out_tensor)

    def infer(self) -> np.ndarray:
        self.request.infer()
        return self.request.get_output_tensor(0).data

    def release(self):
        self.compiled.release_memory()


class StaticBucketTrunk(torch.nn.Module):
    """Drop-in replacement for ``OpenVINOEncoderWrapper`` using static-shape buckets.

    Batch = 1 (and any other batch) is keyed by (batch, static length). Each slot owns
    one ``InferRequest`` with pre-bound tensors, so steady-state calls only fill the
    padding tail and run inference. Shapes that need no padding are bound directly to
    the caller's PyTorch memory (true zero-copy on the host side).
    """

    def __init__(
        self,
        trunk: torch.nn.Module,
        core: ov.Core,
        device: str = "GPU",
        buckets=BUCKETS,
        snap: int = SNAP,
        pad_slack: float = PAD_SLACK,
        cache_dir: str = CACHE_DIR,
        max_static: int = MAX_STATIC,
        max_cached: int = 2,
        verbose: bool = True,
    ):
        super().__init__()
        self.embed_tokens = trunk.embed_tokens
        self._trunk = trunk
        self._core = core
        self._device = device
        self._buckets = tuple(sorted(buckets))
        self._snap = snap
        self._pad_slack = pad_slack
        self._cache_dir = cache_dir
        self._max_static = max_static
        self._max_cached = max_cached
        self._verbose = verbose

        self.hidden = int(trunk.embed_tokens.embedding_dim)
        self._slots: "OrderedDict[tuple[int, int], _StaticSlot]" = OrderedDict()
        self._dynamic = None

        self.stats = {
            "hits": 0,
            "misses": 0,
            "compiles": 0,
            "compile_seconds": 0.0,
            "direct_calls": 0,
            "padded_calls": 0,
            "shapes": [],
        }

    def _target(self, length: int) -> int:
        return target_length(length, self._buckets, self._snap, self._pad_slack)

    def _get_slot(self, batch: int, length: int) -> _StaticSlot:
        key = (batch, length)
        slot = self._slots.get(key)
        if slot is not None:
            self._slots.move_to_end(key)
            self.stats["hits"] += 1
            return slot

        self.stats["misses"] += 1
        while len(self._slots) >= self._max_cached:
            _, old = self._slots.popitem(last=False)
            old.release()
            del old
            gc.collect()

        start = time.perf_counter()
        example = (
            torch.zeros(batch, length, self.hidden),
            torch.ones(batch, length, dtype=torch.bool),
            torch.zeros(batch, dtype=torch.long),
        )
        ov_model = _convert_static(
            self._trunk,
            example,
            [(batch, length, self.hidden), (batch, length), (batch,)],
        )
        compiled = self._core.compile_model(ov_model, self._device, COMPILE_PROPS)
        del ov_model

        slot = _StaticSlot(compiled, batch, length)
        self._slots[key] = slot
        elapsed = time.perf_counter() - start
        self.stats["compiles"] += 1
        self.stats["compile_seconds"] += elapsed
        self.stats["shapes"].append((batch, length))
        if self._verbose:
            print(f"  [buckets] compiled static trunk batch={batch} length={length} in {elapsed:.2f}s")
        return slot

    def _forward_dynamic(self, h_np: np.ndarray, pad_np: np.ndarray, offset_np: np.ndarray):
        if self._dynamic is None:
            ov_model = ov.convert_model(
                self._trunk,
                example_input=(
                    torch.zeros(1, 64, self.hidden),
                    torch.ones(1, 64, dtype=torch.bool),
                    torch.zeros(1, dtype=torch.long),
                ),
            )
            for port in ov_model.inputs:
                rank = port.get_partial_shape().rank.get_length()
                if rank == 3:
                    shape = ov.PartialShape([-1, -1, self.hidden])
                elif rank == 2:
                    shape = ov.PartialShape([-1, -1])
                else:
                    shape = ov.PartialShape([-1])
                port.get_node().set_partial_shape(shape)
            ov_model.validate_nodes_and_infer_types()
            self._dynamic = self._core.compile_model(ov_model, self._device, COMPILE_PROPS)
            del ov_model
        result = self._dynamic(
            [
                ov.Tensor(h_np, shared_memory=True),
                ov.Tensor(pad_np, shared_memory=True),
                ov.Tensor(offset_np, shared_memory=True),
            ],
            share_inputs=True,
            share_outputs=True,
        )
        return torch.from_numpy(result[self._dynamic.output(0)])

    def forward(self, h: torch.Tensor, pad: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        h_np = h.detach().numpy()
        if h_np.dtype != np.float32:
            h_np = h_np.astype(np.float32)
        if not h_np.flags.c_contiguous:
            h_np = np.ascontiguousarray(h_np)
        pad_np = pad.detach().numpy()
        if not pad_np.flags.c_contiguous:
            pad_np = np.ascontiguousarray(pad_np)
        offset_np = offsets.detach().numpy()
        if not offset_np.flags.c_contiguous:
            offset_np = np.ascontiguousarray(offset_np)

        batch, length = int(h_np.shape[0]), int(h_np.shape[1])
        static_length = self._target(length)
        if static_length > self._max_static:
            return self._forward_dynamic(h_np, pad_np, offset_np)

        slot = self._get_slot(batch, static_length)
        if length == static_length:
            slot.request.set_input_tensor(0, ov.Tensor(h_np, shared_memory=True))
            slot.request.set_input_tensor(1, ov.Tensor(pad_np, shared_memory=True))
            slot.request.set_input_tensor(2, ov.Tensor(offset_np, shared_memory=True))
            self.stats["direct_calls"] += 1
        else:
            slot.h[:batch, :length] = h_np
            slot.h[:batch, length:] = 0.0
            slot.pad[:batch, :length] = pad_np
            slot.pad[:batch, length:] = False
            slot.offset[:batch] = offset_np
            self.stats["padded_calls"] += 1

        output = slot.infer()
        return torch.from_numpy(output[:batch, :length])


class FastHead(torch.nn.Module):
    """Explicit, graph-convertible equivalent of ``DecisionHead``.

    ``nn.TransformerEncoder`` takes a fused CPU path that cannot be traced; this
    restates the two pre-norm layers with explicit MHA + FFN ops. It reuses the
    pretrained weights, so outputs match the original head to float32 precision.
    """

    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.type_emb = head.type_emb
        self.layers = getattr(head, "head", head).layers
        self.scorer = head.scorer

    def forward(self, h, pad, marker_pos, marker_mask, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        length = h.shape[1]
        allowed = pad[:, None, None, :]
        for layer in self.layers:
            x = layer.norm1(h)
            attn = layer.self_attn
            weight, bias = attn.in_proj_weight, attn.in_proj_bias
            dim = weight.shape[1]
            qkv = F.linear(x, weight, bias)
            q, k, v = qkv[..., :dim], qkv[..., dim:2 * dim], qkv[..., 2 * dim:]
            batch, heads = x.shape[0], attn.num_heads
            head_dim = dim // heads
            q = q.reshape(batch, length, heads, head_dim).transpose(1, 2)
            k = k.reshape(batch, length, heads, head_dim).transpose(1, 2)
            v = v.reshape(batch, length, heads, head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=head_dim ** -0.5)
            y = y.transpose(1, 2).reshape(batch, length, dim)
            h = h + attn.out_proj(y)
            z = layer.norm2(h)
            z = layer.linear2(layer.activation(layer.linear1(z)))
            h = h + z
        gathered = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.shape[-1]))
        return self.scorer(gathered).squeeze(-1).float().masked_fill(~marker_mask, -1e4)


class _HeadSlot:
    def __init__(self, compiled: ov.CompiledModel, batch: int, length: int, markers: int):
        self.compiled = compiled
        self.batch, self.length, self.markers = batch, length, markers
        self.request = compiled.create_infer_request()

        hidden = 1024
        self.h = np.zeros((batch, length, hidden), np.float32)
        self.pad = np.zeros((batch, length), np.bool_)
        self.marker_pos = np.zeros((batch, markers), np.int64)
        self.marker_mask = np.zeros((batch, markers), np.bool_)
        self.qtype = np.zeros((batch,), np.int64)

        self.tensors = (
            ov.Tensor(self.h, shared_memory=True),
            ov.Tensor(self.pad, shared_memory=True),
            ov.Tensor(self.marker_pos, shared_memory=True),
            ov.Tensor(self.marker_mask, shared_memory=True),
            ov.Tensor(self.qtype, shared_memory=True),
        )
        for index, tensor in enumerate(self.tensors):
            self.request.set_input_tensor(index, tensor)

        self.out_tensor = ov.Tensor(ov.Type.f32, [batch, markers])
        self.request.set_output_tensor(0, self.out_tensor)

    def infer(self) -> np.ndarray:
        self.request.infer()
        return self.request.get_output_tensor(0).data

    def release(self):
        self.compiled.release_memory()


class GpuDecisionHead(torch.nn.Module):
    """Static-shape GPU decision head with a transparent CPU fallback."""

    def __init__(
        self,
        head: torch.nn.Module,
        core: ov.Core,
        device: str = "GPU",
        buckets=BUCKETS,
        snap: int = SNAP,
        pad_slack: float = PAD_SLACK,
        cache_dir: str = CACHE_DIR,
        max_cached: int = 2,
        verbose: bool = True,
    ):
        super().__init__()
        self.impl = FastHead(head)
        self._core = core
        self._device = device
        self._buckets = tuple(sorted(buckets))
        self._snap = snap
        self._pad_slack = pad_slack
        self._cache_dir = cache_dir
        self._max_cached = max_cached
        self._verbose = verbose
        self._slots: "OrderedDict[tuple[int, int, int], _HeadSlot]" = OrderedDict()
        self._failed = False
        self.stats = {"compiles": 0, "compile_seconds": 0.0, "hits": 0, "misses": 0, "shapes": []}

    def _targets(self, length: int, markers: int) -> tuple[int, int]:
        length_target = target_length(length, self._buckets, self._snap, self._pad_slack)
        marker_target = next((bucket for bucket in (2, 4, 8) if bucket >= markers), markers)
        return length_target, marker_target

    def _get_slot(self, batch: int, length: int, markers: int) -> _HeadSlot:
        key = (batch, length, markers)
        slot = self._slots.get(key)
        if slot is not None:
            self._slots.move_to_end(key)
            self.stats["hits"] += 1
            return slot

        self.stats["misses"] += 1
        while len(self._slots) >= self._max_cached:
            _, old = self._slots.popitem(last=False)
            old.release()
            del old
            gc.collect()

        start = time.perf_counter()
        example = (
            torch.zeros(batch, length, 1024),
            torch.ones(batch, length, dtype=torch.bool),
            torch.zeros(batch, markers, dtype=torch.long),
            torch.ones(batch, markers, dtype=torch.bool),
            torch.zeros(batch, dtype=torch.long),
        )
        ov_model = _convert_static(
            self.impl,
            example,
            [(batch, length, 1024), (batch, length), (batch, markers), (batch, markers), (batch,)],
        )
        compiled = self._core.compile_model(ov_model, self._device, COMPILE_PROPS)
        del ov_model

        slot = _HeadSlot(compiled, batch, length, markers)
        self._slots[key] = slot
        elapsed = time.perf_counter() - start
        self.stats["compiles"] += 1
        self.stats["compile_seconds"] += elapsed
        self.stats["shapes"].append((batch, length, markers))
        if self._verbose:
            print(f"  [head] compiled static head batch={batch} length={length} markers={markers} in {elapsed:.2f}s")
        return slot

    def forward(self, h, pad, marker_pos, marker_mask, qtype):
        if self._failed:
            return self.impl(h, pad, marker_pos, marker_mask, qtype)

        h_np = h.detach().numpy()
        if h_np.dtype != np.float32:
            h_np = h_np.astype(np.float32)
        if not h_np.flags.c_contiguous:
            h_np = np.ascontiguousarray(h_np)
        pad_np = pad.detach().numpy()
        pos_np = marker_pos.detach().numpy()
        mask_np = marker_mask.detach().numpy()
        qtype_np = qtype.detach().numpy()

        batch, length = int(h_np.shape[0]), int(h_np.shape[1])
        markers = int(pos_np.shape[1])
        length_target, marker_target = self._targets(length, markers)

        try:
            slot = self._get_slot(batch, length_target, marker_target)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"  [head] GPU head unavailable ({exc}); using CPU head")
            self._failed = True
            return self.impl(h, pad, marker_pos, marker_mask, qtype)

        slot.h[:batch, :length] = h_np
        slot.h[:batch, length:] = 0.0
        slot.pad[:batch, :length] = pad_np
        slot.pad[:batch, length:] = False
        slot.marker_pos[:batch, :markers] = pos_np
        slot.marker_mask[:batch, :markers] = mask_np
        slot.marker_pos[:batch, markers:] = 0
        slot.marker_mask[:batch, markers:] = False
        slot.qtype[:batch] = qtype_np

        output = slot.infer()
        return torch.from_numpy(output[:batch, :markers])


class _Timed(torch.nn.Module):
    """Transparent timing wrapper used by ``profile_stages``."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.reset()

    def reset(self):
        self.seconds = 0.0
        self.calls = 0

    def forward(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self.inner(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - start
            self.calls += 1

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


class _TimedTokenizer:
    def __init__(self, tokenizer):
        object.__setattr__(self, "_tokenizer", tokenizer)
        self.reset()

    def reset(self):
        object.__setattr__(self, "seconds", 0.0)
        object.__setattr__(self, "calls", 0)

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return self._tokenizer(*args, **kwargs)
        finally:
            object.__setattr__(self, "seconds", self.seconds + time.perf_counter() - start)
            object.__setattr__(self, "calls", self.calls + 1)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def profile_stages(engine: D1OmniIrisXe, state: str, questions: dict, iterations: int = 6) -> dict:
    """Measure tokenizer, embeddings, trunk, head and residual Python overhead on ``system_one``."""
    model = engine.model
    original_encoder = model.encoder
    original_head = model.head
    original_tokenizer = model.__dict__.get("tokenizer")

    timed_encoder = _Timed(original_encoder)
    timed_head = _Timed(original_head)
    timed_embed = _Timed(original_encoder.embed_tokens)
    timed_encoder.embed_tokens = timed_embed
    timed_tokenizer = _TimedTokenizer(original_tokenizer) if original_tokenizer is not None else None

    model.encoder = timed_encoder
    model.head = timed_head
    if timed_tokenizer is not None:
        model.__dict__["tokenizer"] = timed_tokenizer

    try:
        engine.system_one(state, questions)
        rows = []
        for _ in range(iterations):
            for timer in (timed_encoder, timed_head, timed_embed):
                timer.reset()
            if timed_tokenizer is not None:
                timed_tokenizer.reset()

            start = time.perf_counter()
            engine.system_one(state, questions)
            total = time.perf_counter() - start
            rows.append(
                {
                    "total": total,
                    "tokenize": timed_tokenizer.seconds if timed_tokenizer is not None else 0.0,
                    "embed": timed_embed.seconds,
                    "trunk": timed_encoder.seconds,
                    "head": timed_head.seconds,
                }
            )
    finally:
        del timed_encoder.embed_tokens
        model.encoder = original_encoder
        model.head = original_head
        if timed_tokenizer is not None:
            model.__dict__["tokenizer"] = original_tokenizer

    medians = {key: statistics.median(row[key] for row in rows) * 1000 for key in rows[0]}
    medians["other"] = max(
        0.0, medians["total"] - medians["tokenize"] - medians["embed"] - medians["trunk"] - medians["head"]
    )
    return medians


def _benchmark(fn, iterations: int) -> list[float]:
    fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _summarize(samples: list[float]) -> dict:
    return {
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "p95": float(np.percentile(samples, 95)),
    }


def _decisions(result: dict) -> dict:
    """Discrete answer each question resolves to, independent of f16 probability drift."""
    out = {}
    for name, answer in result["answers"].items():
        if answer["type"] == "noul":
            out[name] = answer["noul"] >= 0.5
        elif answer["type"] == "choice":
            out[name] = answer["choice"]
        else:
            out[name] = round(answer["score"])
    return out


def _max_numeric_delta(left, right) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        return max((_max_numeric_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        return max((_max_numeric_delta(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float, np.floating)) and isinstance(right, (int, float, np.floating)):
        a, b = float(left), float(right)
        if math.isnan(a) or math.isnan(b):
            return float("inf")
        return abs(a - b)
    return 0.0 if left == right else float("inf")


def _release_dynamic_trunk(engine: D1OmniIrisXe):
    """Free the dynamic GPU trunk so static buckets fit in shared iGPU memory."""
    wrapper = engine.model.encoder
    engine.model.encoder = None
    try:
        wrapper.compiled = None
    except AttributeError:
        pass
    engine.__dict__.pop("compiled_trunk", None)
    del wrapper
    gc.collect()


def _print_breakdown(name: str, stages: dict):
    print(f"  {name:<10} {stages['total']:8.2f} ms  |  "
          f"tokenize {stages['tokenize']:5.2f}  embed {stages['embed']:5.2f}  "
          f"trunk {stages['trunk']:7.2f}  head {stages['head']:6.2f}  other {stages['other']:5.2f}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Batch=1 latency optimization for d1-omni on Iris Xe")
    parser.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--buckets", default="64,128,256", help="comma-separated standard sequence buckets")
    parser.add_argument("--no-head-gpu", action="store_true", help="keep the decision head on the CPU")
    parser.add_argument("--max-cached", type=int, default=2, help="compiled static models kept in iGPU memory")
    parser.add_argument("--parity-tol", type=float, default=2e-2,
                        help="fp16 tolerance for static-vs-dynamic probability drift")
    args = parser.parse_args()

    buckets = tuple(int(part) for part in args.buckets.split(",") if part.strip())
    iterations = max(3, args.iterations)

    print("=" * 92)
    print("  d1-omni-600M  |  Batch = 1 ultra-low-latency optimization  |  Intel Iris Xe")
    print("=" * 92)

    engine = D1OmniIrisXe(device=args.device)
    model = engine.model

    for tower in ("vision", "audio"):
        if getattr(model, tower, None) is not None:
            setattr(model, tower, None)
    gc.collect()
    print("  [memory] released the text-unused vision/audio towers")

    print("\n[1/6] Baseline measurements (dynamic shapes, d1_iris_xe.py path)")
    baseline_probs = model.probabilities(STATE, list(QUESTIONS.values()))
    baseline_single_probs = model.probabilities(STATE, list(SINGLE_QUESTION.values()))
    baseline_result = engine.system_one(STATE, QUESTIONS)
    baseline_single_result = engine.system_one(STATE, SINGLE_QUESTION)

    before_latency = _benchmark(lambda: engine.system_one(STATE, QUESTIONS), iterations)
    before_single = _benchmark(lambda: engine.system_one(STATE, SINGLE_QUESTION), iterations)
    before_stages = profile_stages(engine, STATE, QUESTIONS, max(4, iterations // 2))
    before_single_stages = profile_stages(engine, STATE, SINGLE_QUESTION, max(4, iterations // 2))
    _print_breakdown("3 questions", before_stages)
    _print_breakdown("1 question", before_single_stages)

    print("\n[2/6] Releasing dynamic GPU trunk to make room for static buckets")
    _release_dynamic_trunk(engine)
    print("  released")

    print("\n[3/6] Installing static-shape bucket trunk")
    static_trunk = StaticBucketTrunk(
        engine.trunk_fp32,
        engine.core,
        device=engine.device,
        buckets=buckets,
        max_cached=args.max_cached,
    )
    model.encoder = static_trunk

    head_gpu = None
    if not args.no_head_gpu:
        print("[4/6] Installing static-shape GPU decision head")
        head_gpu = GpuDecisionHead(model.head, engine.core, device=engine.device, max_cached=args.max_cached)
        model.head = head_gpu
    else:
        print("[4/6] Keeping the decision head on the CPU")

    print("\n[5/6] Cold compilation of the measured shapes")
    cold_start = time.perf_counter()
    engine.system_one(STATE, QUESTIONS)
    engine.system_one(STATE, SINGLE_QUESTION)
    cold_seconds = time.perf_counter() - cold_start
    print(f"  cold static compile + first inference: {cold_seconds:.2f}s")

    after_probs = model.probabilities(STATE, list(QUESTIONS.values()))
    after_single_probs = model.probabilities(STATE, list(SINGLE_QUESTION.values()))
    after_result = engine.system_one(STATE, QUESTIONS)
    after_single_result = engine.system_one(STATE, SINGLE_QUESTION)

    after_latency = _benchmark(lambda: engine.system_one(STATE, QUESTIONS), iterations)
    after_single = _benchmark(lambda: engine.system_one(STATE, SINGLE_QUESTION), iterations)
    after_stages = profile_stages(engine, STATE, QUESTIONS, max(4, iterations // 2))
    after_single_stages = profile_stages(engine, STATE, SINGLE_QUESTION, max(4, iterations // 2))

    print("\n[6/6] Results")
    before = _summarize(before_latency)
    after = _summarize(after_latency)
    before_one = _summarize(before_single)
    after_one = _summarize(after_single)

    print("\n" + "=" * 92)
    print("  LATENCY")
    print("=" * 92)
    print(f"  {'Workload':<26}{'Before (ms)':>14}{'After (ms)':>14}{'Speedup':>12}{'Best after':>14}")
    print(f"  {'3 questions (system_one)':<26}{before['median']:>14.2f}{after['median']:>14.2f}"
          f"{before['median'] / after['median']:>11.2f}x{after['min']:>14.2f}")
    print(f"  {'1 question (batch=1)':<26}{before_one['median']:>14.2f}{after_one['median']:>14.2f}"
          f"{before_one['median'] / after_one['median']:>11.2f}x{after_one['min']:>14.2f}")
    print(f"\n  p95: 3 questions {before['p95']:.2f} -> {after['p95']:.2f} ms   |   "
          f"1 question {before_one['p95']:.2f} -> {after_one['p95']:.2f} ms")

    print("\n" + "=" * 92)
    print("  SUB-MODULE BREAKDOWN (median, ms)")
    print("=" * 92)
    _print_breakdown("before 3q", before_stages)
    _print_breakdown("after 3q", after_stages)
    _print_breakdown("before 1q", before_single_stages)
    _print_breakdown("after 1q", after_single_stages)

    delta_three = _max_numeric_delta(baseline_probs, after_probs)
    delta_one = _max_numeric_delta(baseline_single_probs, after_single_probs)
    result_delta = _max_numeric_delta(baseline_result, after_result)
    single_result_delta = _max_numeric_delta(baseline_single_result, after_single_result)

    print("\n" + "=" * 92)
    print("  VERIFICATION vs d1_iris_xe.py (dynamic OpenVINO trunk)")
    print("=" * 92)
    print(f"  max |delta probability|, 3 questions : {delta_three:.3e}")
    print(f"  max |delta probability|, 1 question  : {delta_one:.3e}")
    print(f"  max |delta answer|,      3 questions : {result_delta:.3e}")
    max_delta = max(delta_three, delta_one)
    matches = _decisions(baseline_result) == _decisions(after_result)
    print(f"  max |delta answer|,      1 question  : {single_result_delta:.3e}")
    print(f"  decision fields identical            : {'yes' if matches else 'NO'}")
    print(f"  max |delta probability| overall      : {max_delta:.3e}  "
          f"(f16 tolerance {args.parity_tol:.1e})")
    verdict = "MATCH" if matches and max_delta <= args.parity_tol else "MISMATCH"
    print(f"  verdict: {verdict}"
          + ("   [static/dynamic GPU kernels differ in f16 accumulation order]"
             if verdict == "MATCH" and max_delta > 2e-3 else ""))

    print("\n" + "=" * 92)
    print("  OPTIMIZATION TELEMETRY")
    print("=" * 92)
    print(f"  static trunk shapes compiled : {static_trunk.stats['shapes']}")
    print(f"  static trunk cache           : hits={static_trunk.stats['hits']} misses={static_trunk.stats['misses']} "
          f"compiles={static_trunk.stats['compiles']} compile={static_trunk.stats['compile_seconds']:.2f}s")
    print(f"  zero-copy calls / padded     : {static_trunk.stats['direct_calls']} / {static_trunk.stats['padded_calls']}")
    if head_gpu is not None:
        print(f"  static head shapes compiled  : {head_gpu.stats['shapes']}")
        print(f"  static head cache            : hits={head_gpu.stats['hits']} misses={head_gpu.stats['misses']} "
              f"compiles={head_gpu.stats['compiles']} compile={head_gpu.stats['compile_seconds']:.2f}s")
    print(f"  cold compile + first call    : {cold_seconds:.2f}s")

    print("\n  Baseline output:")
    print("  " + json.dumps(baseline_result, indent=2).replace("\n", "\n  "))
    print("\n  Optimized output:")
    print("  " + json.dumps(after_result, indent=2).replace("\n", "\n  "))
    print("=" * 92)


if __name__ == "__main__":
    main()
