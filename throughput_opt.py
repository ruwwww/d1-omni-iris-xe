"""
Throughput and batch-scaling harness for d1-omni-600M on Intel Iris Xe.

Implements OpenVINO multi-stream asynchronous execution (AsyncInferQueue,
PERFORMANCE_HINT=THROUGHPUT, NUM_STREAMS=AUTO), heterogeneous GPU/CPU/MULTI
device comparison, and an automated batch sweep from 1 to 256 queries.

Usage:
    python throughput_opt.py
    python throughput_opt.py --device GPU --batch-sizes 1 8 32 --repeats 2
    python throughput_opt.py --merge-only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import openvino as ov
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, PreTrainedTokenizerFast
from transformers.dynamic_module_utils import get_class_from_dynamic_module

try:
    import psutil
except ImportError:
    psutil = None

ROOT = Path(__file__).resolve().parent
MODEL_ID = "LiquidAI/d1-omni-600M"
CACHE_DIR = Path(os.path.expanduser("~/.cache/openvino_iris_xe"))
BENCH_DIR = CACHE_DIR / "throughput_bench"
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_DEVICES = ["GPU", "CPU", "MULTI:GPU,CPU"]
MAX_POOL_WORKERS = 16
TOKEN_BUDGET_PER_QUERY = 3

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
        "criteria": [
            "Minor non-blocking glitch",
            "Degraded performance",
            "Immediate revenue loss",
        ],
    },
}

STATE_TEMPLATES = [
    "Production Incident Report: Primary payment gateway returned HTTP 500 error code for 12 consecutive requests. Transactions failing. Customer checkout blocked. Severity automatically elevated.",
    "Customer ticket #8841: I was double billed on my October subscription renewal. Please reverse the duplicate charge of $49.00 to my Visa.",
    "SSH daemon on bast-gw-01 detected 142 failed login attempts within 60 seconds from IP 198.51.100.44 using root username.",
    "Passenger Smith on flight AA1290 from DFW to ORD: Flight cancelled due to mechanical failure. Stranded overnight.",
    "Kubernetes probe reports readiness failures across 6 replicas of checkout-service in us-east-1. Error budget burn rate is 14x.",
    "Inventory sync job lag exceeds 45 minutes after the ERP connector upgrade. Warehouse pick lists are stale.",
    "Marketing newsletter send to 240k recipients bounced for 31 percent of addresses. Domain reputation dropped to neutral.",
    "Mobile app crash rate jumped from 0.2 percent to 4.8 percent after release 7.3.1 on Android 14 devices.",
    "Database connection pool exhausted on the analytics cluster. Query latency rose from 40 ms to 2.4 seconds.",
    "Fraud model flagged 73 transactions from a single merchant in the last hour. Total exposure is $18,400.",
    "CDN origin shield in Frankfurt reports elevated 5xx responses for static assets in the EU region.",
    "Support queue backlog reached 1,200 tickets. First response time breached the 4 hour service target.",
    "Billing invoice batch failed validation because of a currency code mismatch for accounts in Switzerland.",
    "Authentication service latency increased after certificate rotation. 8 percent of logins time out.",
    "Search relevance experiment variant B shows a 6 percent drop in click-through rate over seven days.",
    "Nightly ETL pipeline missed its SLA. Downstream executive dashboards are showing yesterday's revenue.",
]


def build_requests(batch_size: int) -> list[tuple]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    requests = []
    for i in range(batch_size):
        state = f"{STATE_TEMPLATES[i % len(STATE_TEMPLATES)]} (request #{i + 1})"
        requests.append((state, QUESTIONS))
    return requests


def _rss_bytes() -> int:
    if psutil is None:
        return 0
    return psutil.Process(os.getpid()).memory_info().rss


class RSSMonitor:
    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.wait(self.interval_s):
            self.peak = max(self.peak, _rss_bytes())

    def stop(self) -> tuple[int, int, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        after = _rss_bytes()
        self.peak = max(self.peak, after)
        return self.baseline, self.peak, after


def _mb(value: int | float) -> float:
    return round(float(value) / (1024 * 1024), 2)


@dataclass
class Job:
    result: np.ndarray | None = None
    error: BaseException | None = None
    event: threading.Event = field(default_factory=threading.Event)


class AsyncOpenVINOEncoder(torch.nn.Module):
    def __init__(self, compiled_model, embed_tokens, num_requests: int):
        super().__init__()
        self.compiled = compiled_model
        self.embed_tokens = embed_tokens
        self.num_requests = max(1, int(num_requests))
        self.queue = self._make_queue(self.num_requests)
        self.queue.set_callback(self._on_complete)
        self._submit_lock = threading.Lock()
        self._closed = False

    def _make_queue(self, num_requests: int) -> ov.AsyncInferQueue:
        attempts = []
        current = max(1, int(num_requests))
        while current >= 1:
            attempts.append(current)
            try:
                return ov.AsyncInferQueue(self.compiled, current)
            except (MemoryError, RuntimeError) as exc:
                message = str(exc).lower()
                if current == 1 or "alloc" not in message and not isinstance(exc, MemoryError):
                    raise
                current //= 2
        raise RuntimeError(f"could not create an infer queue (attempts: {attempts})")

    @staticmethod
    def _on_complete(infer_request, job: Job):
        try:
            job.result = np.array(infer_request.get_output_tensor(0).data, copy=True)
        except BaseException as exc:
            job.error = exc
        finally:
            job.event.set()

    def forward(self, h, pad, offsets):
        if self._closed:
            raise RuntimeError("inference queue is closed")
        h_np = np.ascontiguousarray(h.detach().cpu().numpy())
        pad_np = np.ascontiguousarray(pad.detach().cpu().numpy())
        off_np = np.ascontiguousarray(offsets.detach().cpu().numpy())
        job = Job()
        with self._submit_lock:
            self.queue.start_async({0: h_np, 1: pad_np, 2: off_np}, job)
        if not job.event.wait(timeout=900):
            raise TimeoutError("OpenVINO async inference did not complete in 900 s")
        if job.error is not None:
            raise job.error
        return torch.from_numpy(job.result).float()

    def close(self):
        self._closed = True


class ThroughputEngine:
    def __init__(
        self,
        device: str,
        model_id: str = MODEL_ID,
        threads: int = 4,
        precision: str = "auto",
    ):
        self.device = device.upper()
        self.model_id = model_id
        self.init_started = time.perf_counter()
        self.available_devices = []
        self.gpu_name = None
        self.cpu_name = None
        self.gpu_capabilities = None
        self.streams_property = None
        self.optimal_infer_requests = 1
        self.compile_time_s = 0.0
        self.model_load_time_s = 0.0
        self.precision = precision.lower()
        self.inference_precision = self._resolve_precision()
        self.input_dtype = torch.float32

        torch.set_num_threads(max(1, int(threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.core = ov.Core()
        self.available_devices = list(self.core.available_devices)
        self._validate_device()
        self._configure_devices()

        if "GPU" in self.available_devices:
            self.gpu_name = self.core.get_property("GPU", "FULL_DEVICE_NAME")
            self.gpu_capabilities = list(
                self.core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")
            )
        if "CPU" in self.available_devices:
            self.cpu_name = self.core.get_property("CPU", "FULL_DEVICE_NAME")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self._compile_trunk()

        self.pool = ThreadPoolExecutor(
            max_workers=self.num_pool_workers(), thread_name_prefix="ov-stream"
        )
        self.init_time_s = time.perf_counter() - self.init_started

    def _resolve_precision(self) -> str:
        if self.precision == "fp16":
            return "f16"
        if self.precision == "fp32":
            return "f32"
        if self.precision != "auto":
            raise ValueError(f"unknown precision {self.precision!r}")
        return "f16" if "GPU" in self.resolved_devices else "f32"

    @property
    def resolved_devices(self) -> list[str]:
        if self.device.startswith("MULTI:"):
            return [d for d in self.device.split(":", 1)[1].split(",") if d]
        return [self.device]

    def _validate_device(self):
        if self.device.startswith("MULTI:"):
            missing = [d for d in self.resolved_devices if d not in self.available_devices]
            if missing:
                raise RuntimeError(
                    f"device {self.device!r} requires {missing}; available: {self.available_devices}"
                )
        elif self.device not in self.available_devices:
            raise RuntimeError(
                f"device {self.device!r} not available; available: {self.available_devices}"
            )

    def _configure_devices(self):
        properties = {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": "AUTO",
            "CACHE_DIR": str(CACHE_DIR),
        }
        for device in self.resolved_devices:
            if device in self.available_devices:
                self.core.set_property(device, properties)

    def _load_tokenizer(self):
        print("[throughput] loading tokenizer...")
        tokenizer_path = hf_hub_download(self.model_id, "tokenizer.json")
        return PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="<|startoftext|>",
            eos_token="<|im_end|>",
            pad_token="<|pad|>",
            mask_token="<|mask|>",
        )

    def _load_model(self):
        print(f"[throughput] selectively loading text trunk + head from {self.model_id}...")
        started = time.perf_counter()
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        config.max_length = 16384
        model_cls = get_class_from_dynamic_module(
            "modeling_d1.D1OmniModel", self.model_id
        )
        with torch.device("meta"):
            model = model_cls(config)
        model.eval()
        model.config.max_length = 16384
        model.name_or_path = self.model_id
        model.__dict__["tokenizer"] = self.tokenizer

        checkpoint = hf_hub_download(self.model_id, "model.safetensors")
        text_state = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(("encoder.", "head.")):
                    text_state[key] = handle.get_tensor(key)
        missing, unexpected = model.load_state_dict(
            text_state, strict=False, assign=True
        )
        del text_state
        model.vision = None
        model.audio = None
        model.head = model.head.float()
        gc.collect()
        self.model_load_time_s = time.perf_counter() - started
        print(
            f"[throughput] weights loaded in {self.model_load_time_s:.2f} s "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return model

    def _compile_trunk(self):
        print(
            f"[throughput] converting and compiling trunk for {self.device} "
            f"as {str(self.input_dtype).replace('torch.', '')}..."
        )
        started = time.perf_counter()
        example_h = torch.randn(3, 64, 1024, dtype=self.input_dtype)
        example_pad = torch.ones(3, 64, dtype=torch.bool)
        example_offsets = torch.zeros(3, dtype=torch.long)
        trunk = self.model.encoder.to(self.input_dtype)
        embed_tokens = trunk.embed_tokens
        ov_trunk = ov.convert_model(
            trunk, example_input=(example_h, example_pad, example_offsets)
        )
        for inp in ov_trunk.inputs:
            rank = inp.get_partial_shape().rank.get_length()
            if rank == 3:
                shape = [-1, -1, 1024]
            elif rank == 2:
                shape = [-1, -1]
            else:
                shape = [-1]
            inp.get_node().set_partial_shape(ov.PartialShape(shape))
        ov_trunk.validate_nodes_and_infer_types()

        ir_dir = CACHE_DIR / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        model_slug = re.sub(r"[^A-Za-z0-9]+", "_", self.model_id).strip("_").lower()
        ir_path = ir_dir / f"{model_slug}_{self.inference_precision}_v1.xml"
        ov.save_model(
            ov_trunk,
            str(ir_path),
            compress_to_fp16=self.inference_precision == "f16",
        )

        self.model.encoder = None
        del trunk, ov_trunk
        gc.collect()
        print(f"[throughput] torch trunk released, rss={_mb(_rss_bytes()):.0f} MB")

        compile_properties = {
            "PERFORMANCE_HINT": "THROUGHPUT",
            "NUM_STREAMS": "AUTO",
            "CACHE_DIR": str(CACHE_DIR),
            "INFERENCE_PRECISION_HINT": self.inference_precision,
        }
        self.compiled_trunk = self.core.compile_model(
            str(ir_path), self.device, compile_properties
        )
        self.compile_time_s = time.perf_counter() - started

        try:
            self.optimal_infer_requests = int(
                self.compiled_trunk.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
            )
        except Exception:
            self.optimal_infer_requests = 1
        try:
            streams = self.compiled_trunk.get_property("NUM_STREAMS")
            self.streams_property = int(streams)
        except Exception:
            self.streams_property = None

        queue_requests = self.optimal_infer_requests
        if self.streams_property:
            queue_requests = min(queue_requests, self.streams_property)
        self.optimal_infer_requests = max(1, min(queue_requests, MAX_POOL_WORKERS))

        self.encoder = AsyncOpenVINOEncoder(
            self.compiled_trunk, embed_tokens, self.optimal_infer_requests
        )
        self.model.encoder = self.encoder
        print(
            f"[throughput] compiled in {self.compile_time_s:.2f} s | "
            f"streams={self.streams_property} infer_requests={self.optimal_infer_requests} | "
            f"rss={_mb(_rss_bytes()):.0f} MB"
        )

    def num_pool_workers(self) -> int:
        return max(1, min(self.optimal_infer_requests, MAX_POOL_WORKERS))

    def chunk_count(self, batch_size: int) -> int:
        if batch_size < 8:
            return 1
        return max(1, min(self.num_pool_workers(), batch_size // 4))

    @staticmethod
    def _split(requests: Sequence, parts: int) -> list[list]:
        parts = max(1, min(parts, len(requests)))
        base, extra = divmod(len(requests), parts)
        chunks, start = [], 0
        for i in range(parts):
            size = base + (1 if i < extra else 0)
            chunks.append(list(requests[start : start + size]))
            start += size
        return chunks

    def run_batch(self, requests: Sequence[tuple]) -> tuple[list[dict], float, int]:
        chunks = self._split(requests, self.chunk_count(len(requests)))
        started = time.perf_counter()
        if len(chunks) == 1:
            outputs = [self.model.system_one_batch(chunks[0])]
        else:
            futures = [self.pool.submit(self.model.system_one_batch, chunk) for chunk in chunks]
            outputs = [future.result() for future in futures]
        elapsed = time.perf_counter() - started

        results: list[dict] = []
        for output in outputs:
            results.extend(output)
        if len(results) != len(requests):
            raise RuntimeError(
                f"expected {len(requests)} answers, received {len(results)}"
            )
        tokens = sum(int(r.get("usage", {}).get("input_tokens", 0)) for r in results)
        return results, elapsed, tokens

    def settings(self) -> dict:
        return {
            "requested_device": self.device,
            "resolved_devices": self.resolved_devices,
            "performance_hint": "THROUGHPUT",
            "num_streams_requested": "AUTO",
            "num_streams_effective": self.streams_property,
            "inference_precision": self.inference_precision,
            "io_dtype": str(self.input_dtype).replace("torch.", ""),
            "optimal_infer_requests": self.optimal_infer_requests,
            "pool_workers": self.num_pool_workers(),
            "torch_threads": torch.get_num_threads(),
            "chunk_policy": "1 chunk for B<8, else min(pool_workers, B//4)",
            "cache_dir": str(CACHE_DIR),
        }

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.encoder.close()


def _checksum(results: Sequence[dict]) -> float:
    total = 0.0
    for item in results:
        for payload in item.get("answers", {}).values():
            if "confidence" in payload:
                total += float(payload["confidence"])
            elif "noul" in payload:
                total += float(payload["noul"])
            elif "score" in payload:
                total += float(payload["score"])
    return round(total, 6)


def benchmark_device(
    device: str,
    batch_sizes: Sequence[int],
    repeats: int,
    warmup_per_size: bool,
    threads: int,
    precision: str = "auto",
) -> dict:
    engine = ThroughputEngine(device=device, threads=threads, precision=precision)
    result_rows = []
    try:
        print("[throughput] warmup pass (B=1, B=4)...")
        for warm_batch in (1, 4):
            if warm_batch <= max(batch_sizes):
                engine.run_batch(build_requests(warm_batch))

        for batch_size in batch_sizes:
            requests = build_requests(batch_size)
            if warmup_per_size:
                engine.run_batch(requests)

            latencies = []
            peak_rss = 0
            baselines = []
            after_rss = []
            token_counts = []
            checksums = []
            for run_index in range(repeats):
                gc.collect()
                monitor = RSSMonitor()
                monitor.start()
                outputs, elapsed, tokens = engine.run_batch(requests)
                baseline, peak, after = monitor.stop()
                latencies.append(elapsed * 1000.0)
                peak_rss = max(peak_rss, peak)
                baselines.append(baseline)
                after_rss.append(after)
                token_counts.append(tokens)
                checksums.append(_checksum(outputs))
                print(
                    f"  {device:>13} B={batch_size:<3} run {run_index + 1}/{repeats}: "
                    f"{elapsed * 1000.0:9.2f} ms | {batch_size / elapsed:8.2f} q/s | "
                    f"RSS {_mb(peak):7.1f} MB"
                )

            median_ms = statistics.median(latencies)
            baseline = statistics.median(baselines)
            throughput = batch_size * 1000.0 / median_ms if median_ms > 0 else 0.0
            result_rows.append(
                {
                    "batch_size": batch_size,
                    "total_latency_ms": round(median_ms, 3),
                    "total_latency_ms_mean": round(statistics.fmean(latencies), 3),
                    "total_latency_ms_min": round(min(latencies), 3),
                    "total_latency_ms_max": round(max(latencies), 3),
                    "runs_ms": [round(v, 3) for v in latencies],
                    "per_query_latency_ms": round(median_ms / batch_size, 4),
                    "throughput_qps": round(throughput, 3),
                    "memory": {
                        "baseline_rss_mb": _mb(baseline),
                        "peak_rss_mb": _mb(peak_rss),
                        "incremental_peak_mb": _mb(max(0, peak_rss - baseline)),
                        "after_rss_mb": _mb(statistics.median(after_rss)),
                    },
                    "input_tokens": int(statistics.median(token_counts)),
                    "input_tokens_per_query": round(
                        statistics.median(token_counts) / batch_size, 2
                    ),
                    "checksum": checksums[0],
                    "checksum_stable": len(set(checksums)) == 1,
                    "chunks": engine.chunk_count(batch_size),
                }
            )
    finally:
        payload = {
            "device": device,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "settings": engine.settings(),
            "hardware": {
                "available_devices": engine.available_devices,
                "gpu_name": engine.gpu_name,
                "cpu_name": engine.cpu_name,
                "gpu_capabilities": engine.gpu_capabilities,
            },
            "timing": {
                "init_time_s": round(engine.init_time_s, 3),
                "model_load_time_s": round(engine.model_load_time_s, 3),
                "trunk_compile_time_s": round(engine.compile_time_s, 3),
            },
            "results": result_rows,
        }
        engine.close()
    return payload


def _slug(device: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", device).strip("_").lower()


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def _host_info() -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "openvino": ov.__version__,
        "numpy": np.__version__,
        "cpu_count_logical": os.cpu_count(),
        "ram_total_mb": _mb(psutil.virtual_memory().total) if psutil else None,
    }


def _scaling_rows(results: Sequence[dict]) -> list[dict]:
    base = results[0] if results else None
    base_qps = base["throughput_qps"] if base else 0.0
    rows = []
    for row in results:
        speedup = row["throughput_qps"] / base_qps if base_qps else 0.0
        efficiency = speedup / row["batch_size"] * 100.0
        rows.append(
            {
                "batch_size": row["batch_size"],
                "qps": row["throughput_qps"],
                "speedup_vs_b1": round(speedup, 2),
                "scaling_efficiency_pct": round(efficiency, 2),
            }
        )
    return rows


def merge_results(device_payloads: Sequence[dict], benchmark_config: dict) -> dict:
    devices = {}
    for payload in device_payloads:
        device = payload["device"]
        rows = payload["results"]
        devices[device] = {
            "settings": payload["settings"],
            "hardware": payload["hardware"],
            "timing": payload["timing"],
            "results": rows,
            "scaling": _scaling_rows(rows),
        }
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": _host_info(),
        "model": {
            "id": MODEL_ID,
            "questions_per_query": len(QUESTIONS),
            "question_names": list(QUESTIONS),
            "max_length": 16384,
        },
        "benchmark": benchmark_config,
        "devices": devices,
    }


def _fmt(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def _ascii_curve(device: str, scaling: Sequence[dict], max_qps: float) -> list[str]:
    lines = [f"{device}", f"{'B':>4}  {'q/s':>9}  {'speedup':>8}  curve"]
    for row in scaling:
        width = int(round(40 * row["qps"] / max_qps)) if max_qps else 0
        bar = "#" * max(1, width)
        lines.append(
            f"{row['batch_size']:>4}  {_fmt(row['qps'], 2):>9}  "
            f"{row['speedup_vs_b1']:>7.2f}x  {bar}"
        )
    return lines


def _key_findings(merged: dict) -> list[str]:
    findings = []
    devices = merged["devices"]
    batch_sizes = merged["benchmark"]["batch_sizes"]
    for device, data in devices.items():
        rows = {r["batch_size"]: r for r in data["results"]}
        best = max(data["results"], key=lambda r: r["throughput_qps"])
        b1 = rows.get(1)
        findings.append(
            f"{device}: peak {best['throughput_qps']:.2f} q/s at B={best['batch_size']}"
            + (
                f" ({best['throughput_qps'] / b1['throughput_qps']:.1f}x over B=1)"
                if b1 and b1["throughput_qps"]
                else ""
            )
            + f", {best['per_query_latency_ms']:.2f} ms/query, "
            f"{best['memory']['peak_rss_mb']:.0f} MB peak RSS."
        )
    if len(devices) > 1:
        for batch in batch_sizes:
            candidates = []
            for device, data in devices.items():
                row = next((r for r in data["results"] if r["batch_size"] == batch), None)
                if row:
                    candidates.append((row["throughput_qps"], device))
            if candidates:
                qps, device = max(candidates)
                findings.append(f"B={batch}: fastest device is {device} at {qps:.2f} q/s.")
    return findings


def build_markdown(merged: dict) -> str:
    devices = merged["devices"]
    host = merged["host"]
    bench = merged["benchmark"]
    batch_sizes = bench["batch_sizes"]
    gpu_name = next(
        (d["hardware"].get("gpu_name") for d in devices.values() if d["hardware"].get("gpu_name")),
        "Intel Iris Xe Graphics",
    )
    cpu_name = next(
        (d["hardware"].get("cpu_name") for d in devices.values() if d["hardware"].get("cpu_name")),
        "Intel Core i7-1360P",
    )

    lines = []
    lines.append("# Batch Scaling & Throughput Sweep (B = 1 to 256)")
    lines.append("")
    lines.append(
        "Automated throughput benchmark for **LiquidAI/d1-omni-600M** on "
        f"**{gpu_name}** + **{cpu_name}** via OpenVINO."
    )
    lines.append("")
    lines.append("## Test configuration")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Generated | {merged['generated_at']} |")
    lines.append(f"| Host | {host['hostname']} ({host['platform']}) |")
    lines.append(f"| CPU | {cpu_name} ({host['cpu_count_logical']} logical cores) |")
    lines.append(f"| GPU | {gpu_name} |")
    lines.append(f"| Model | `{merged['model']['id']}` |")
    lines.append(
        f"| Questions per query | {merged['model']['questions_per_query']} "
        f"({', '.join(merged['model']['question_names'])}) |"
    )
    lines.append(f"| Batch sizes | {', '.join(str(b) for b in batch_sizes)} |")
    lines.append(f"| Repeats per batch | {bench['repeats']} (median reported) |")
    lines.append(f"| Per-size warmup | {'yes' if bench['warmup_per_size'] else 'no'} |")
    lines.append(
        "| Execution | `ov.AsyncInferQueue`, `PERFORMANCE_HINT=THROUGHPUT`, "
        "`NUM_STREAMS=AUTO` |"
    )
    lines.append(f"| OpenVINO / PyTorch | {host['openvino']} / {host['torch']} |")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Each query is one state evaluated against three typed questions (noul, choice, score). "
        "A batch of B queries is submitted as one workload, split across asynchronous OpenVINO "
        "infer requests on the target device. Reported total latency is wall-clock time from "
        "submission to all B answers; per-query latency is total/B; throughput is B/total. "
        "Memory is process RSS sampled every 10 ms during the timed run."
    )
    lines.append("")

    for device, data in devices.items():
        settings = data["settings"]
        lines.append(f"## {device}")
        lines.append("")
        lines.append(
            f"Effective streams: **{settings['num_streams_effective'] or 'auto'}**, "
            f"optimal infer requests: **{settings['optimal_infer_requests']}**, "
            f"pool workers: **{settings['pool_workers']}**. "
            f"Init {data['timing']['init_time_s']:.1f} s, trunk compile "
            f"{data['timing']['trunk_compile_time_s']:.1f} s."
        )
        lines.append("")
        lines.append(
            "| Batch | Total latency (ms) | Per-query latency (ms) | Throughput (q/s) "
            "| Speedup vs B=1 | Scaling efficiency | Peak RSS (MB) | Incremental (MB) |"
        )
        lines.append(
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        scaling = {row["batch_size"]: row for row in data["scaling"]}
        for row in data["results"]:
            scale = scaling[row["batch_size"]]
            lines.append(
                f"| {row['batch_size']} | {_fmt(row['total_latency_ms'])} "
                f"| {_fmt(row['per_query_latency_ms'], 3)} "
                f"| {_fmt(row['throughput_qps'])} "
                f"| {scale['speedup_vs_b1']:.2f}x "
                f"| {scale['scaling_efficiency_pct']:.1f}% "
                f"| {_fmt(row['memory']['peak_rss_mb'], 1)} "
                f"| {_fmt(row['memory']['incremental_peak_mb'], 1)} |"
            )
        lines.append("")

    if len(devices) > 1:
        lines.append("## Device comparison (throughput, q/s)")
        lines.append("")
        header = "| Batch | " + " | ".join(devices) + " | Fastest |"
        lines.append(header)
        lines.append("| ---: | " + " | ".join("---:" for _ in devices) + " | :--- |")
        for batch in batch_sizes:
            cells, candidates = [], []
            for device, data in devices.items():
                row = next((r for r in data["results"] if r["batch_size"] == batch), None)
                if row:
                    cells.append(_fmt(row["throughput_qps"]))
                    candidates.append((row["throughput_qps"], device))
                else:
                    cells.append("n/a")
            fastest = max(candidates)[1] if candidates else "n/a"
            lines.append(f"| {batch} | " + " | ".join(cells) + f" | {fastest} |")
        lines.append("")

    lines.append("## Throughput scaling curves")
    lines.append("")
    max_qps = max(
        (row["throughput_qps"] for data in devices.values() for row in data["results"]),
        default=0.0,
    )
    lines.append("```text")
    for device, data in devices.items():
        lines.extend(_ascii_curve(device, data["scaling"], max_qps))
        lines.append("")
    lines.append("```")
    lines.append("")

    lines.append("## Key findings")
    lines.append("")
    for finding in _key_findings(merged):
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Throughput scales until the device's EU/thread and memory-bandwidth budget saturates; "
        "the scaling-efficiency column shows the fraction of linear scaling retained."
    )
    lines.append(
        "- Peak RSS includes the Python process, the PyTorch model, and OpenVINO runtime buffers. "
        "On an iGPU, device allocations are drawn from shared system memory."
    )
    lines.append(
        "- `MULTI:GPU,CPU` distributes infer requests across both devices; small batches may be "
        "slower than GPU-only because of cross-device scheduling overhead."
    )
    lines.append("")
    return "\n".join(lines)


def print_summary(merged: dict):
    devices = merged["devices"]
    batch_sizes = merged["benchmark"]["batch_sizes"]
    print("\n" + "=" * 78)
    print("  THROUGHPUT & BATCH SCALING SUMMARY")
    print("=" * 78)
    for device, data in devices.items():
        print(f"\n{device}  (streams={data['settings']['num_streams_effective']}, "
              f"infer_reqs={data['settings']['optimal_infer_requests']})")
        print(f"  {'B':>4} {'total ms':>10} {'ms/query':>10} {'q/s':>9} {'speedup':>9} {'peak MB':>9}")
        scaling = {row["batch_size"]: row for row in data["scaling"]}
        for row in data["results"]:
            scale = scaling[row["batch_size"]]
            print(
                f"  {row['batch_size']:>4} {row['total_latency_ms']:>10.2f} "
                f"{row['per_query_latency_ms']:>10.3f} {row['throughput_qps']:>9.2f} "
                f"{scale['speedup_vs_b1']:>8.2f}x {row['memory']['peak_rss_mb']:>9.1f}"
            )
    max_qps = max(
        (row["throughput_qps"] for data in devices.values() for row in data["results"]),
        default=0.0,
    )
    print("\nThroughput scaling curves")
    print("-" * 78)
    for device, data in devices.items():
        for line in _ascii_curve(device, data["scaling"], max_qps):
            print(line)
        print()
    print("Key findings")
    print("-" * 78)
    for finding in _key_findings(merged):
        print(f"* {finding}")
    print("=" * 78)


BENCHMARK_PROCESS_HINTS = (
    "latency_opt",
    "throughput_opt",
    "d1_iris_xe",
    "demo_run",
    "web_app",
    "probe",
)


def _competing_benchmarks() -> list[str]:
    if psutil is None:
        return []
    current = {os.getpid(), os.getppid()}
    competitors = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.pid in current:
                continue
            name = (process.info.get("name") or "").lower()
            if not name.startswith("python"):
                continue
            cmdline = process.info.get("cmdline") or []
            command = " ".join(cmdline).lower()
            if any(hint in command for hint in BENCHMARK_PROCESS_HINTS):
                script = next(
                    (os.path.basename(part) for part in cmdline if part.endswith(".py")),
                    name,
                )
                competitors.append(f"{process.pid}:{script}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return competitors


def wait_for_memory(
    min_available_mb: int = 5500, timeout_s: int = 7200, poll_s: float = 5.0
) -> None:
    if psutil is None:
        return
    started = time.perf_counter()
    warned = False
    stable_since = None
    while True:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        competitors = _competing_benchmarks()
        ready = available_mb >= min_available_mb and not competitors
        if ready:
            if stable_since is None:
                stable_since = time.perf_counter()
            elif time.perf_counter() - stable_since >= 8:
                if warned:
                    print(
                        f"[throughput] resources ready ({available_mb:.0f} MB free, "
                        "no competing benchmark)"
                    )
                return
        else:
            stable_since = None
            if not warned:
                detail = f"{available_mb:.0f} MB free"
                if competitors:
                    detail += f", competing: {', '.join(competitors[:3])}"
                print(f"[throughput] waiting for exclusive benchmark resources ({detail})")
                warned = True
        if time.perf_counter() - started > timeout_s:
            raise RuntimeError(
                f"resources unavailable after {timeout_s} s "
                f"({available_mb:.0f} MB free, competitors={competitors})"
            )
        time.sleep(poll_s)


def _run_worker(args) -> int:
    output_path = Path(args.worker_output)
    try:
        wait_for_memory()
        payload = benchmark_device(
            device=args.worker_device,
            batch_sizes=args.batch_sizes,
            repeats=args.repeats,
            warmup_per_size=args.warmup_per_size,
            threads=args.torch_threads,
            precision=args.precision,
        )
        _write_json(output_path, payload)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


def _stream_subprocess(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            list(cmd),
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def _run_all(args) -> int:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for device in args.devices:
        output_path = BENCH_DIR / f"{_slug(device)}.json"
        log_path = BENCH_DIR / f"{_slug(device)}.log"
        if args.resume and output_path.exists():
            print(f"[throughput] reusing existing result for {device}: {output_path}")
            continue
        print("\n" + "#" * 78)
        print(f"# Benchmarking device: {device}")
        print("#" * 78)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-device",
            device,
            "--worker-output",
            str(output_path),
            "--repeats",
            str(args.repeats),
            "--torch-threads",
            str(args.torch_threads),
            "--precision",
            args.precision,
            "--batch-sizes",
            *[str(b) for b in args.batch_sizes],
        ]
        cmd.append("--warmup-per-size" if args.warmup_per_size else "--no-warmup-per-size")
        code = _stream_subprocess(cmd, log_path)
        if code != 0:
            print(f"[throughput] device {device} failed with exit code {code}; see {log_path}")
        else:
            print(f"[throughput] device {device} complete -> {output_path}")
    return _merge_artifacts(args)


def _load_payloads(devices: Sequence[str]) -> list[dict]:
    payloads = []
    for device in devices:
        path = BENCH_DIR / f"{_slug(device)}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        else:
            print(f"[throughput] warning: no results for {device} at {path}")
    return payloads


def _merge_artifacts(args) -> int:
    payloads = _load_payloads(args.devices)
    if not payloads:
        print("[throughput] no result files found; nothing to merge.")
        return 1
    benchmark_config = {
        "batch_sizes": [int(b) for b in args.batch_sizes],
        "repeats": int(args.repeats),
        "warmup_per_size": bool(args.warmup_per_size),
        "statistic": "median",
        "requests_per_batch": "batch_size",
        "questions_per_request": len(QUESTIONS),
        "asynchronous": True,
        "performance_hint": "THROUGHPUT",
        "num_streams": "AUTO",
    }
    merged = merge_results(payloads, benchmark_config)
    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    _write_json(json_path, merged)
    md_path.write_text(build_markdown(merged), encoding="utf-8")
    print(f"\n[throughput] wrote {json_path}")
    print(f"[throughput] wrote {md_path}")
    print_summary(merged)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch-scaling and throughput benchmark for d1-omni-600M."
    )
    parser.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES)
    parser.add_argument("--device", default=None, help="Benchmark a single device.")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmup-per-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one untimed warmup pass for every batch size.",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "fp32"],
        default="auto",
        help="Trunk inference precision; auto uses FP16 when a GPU is involved.",
    )
    parser.add_argument("--output-json", default=str(ROOT / "benchmark_results.json"))
    parser.add_argument("--output-md", default=str(ROOT / "BENCHMARK_SWEEP.md"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.device:
        args.devices = [args.device]
    args.batch_sizes = sorted({int(b) for b in args.batch_sizes if int(b) > 0})
    if not args.batch_sizes:
        parser.error("at least one positive batch size is required")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.worker and (not args.worker_device or not args.worker_output):
        parser.error("--worker requires --worker-device and --worker-output")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.worker:
        return _run_worker(args)
    if args.merge_only:
        return _merge_artifacts(args)
    return _run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
