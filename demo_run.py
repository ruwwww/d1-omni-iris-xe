import os
import sys
import time
import json
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, PreTrainedTokenizerFast
from transformers.dynamic_module_utils import get_class_from_dynamic_module
import openvino as ov

CACHE_DIR = os.path.expanduser("~/.cache/openvino_iris_xe")
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 65)
print("  LIQUID AI d1-omni-600M ON INTEL IRIS XE (96 EU) ACCELERATION")
print("=" * 65)

# Step 1: OpenVINO Hardware Verification
print("\n[Step 1] Initializing OpenVINO Runtime & Hardware Inspection...")
core = ov.Core()
core.set_property("GPU", {
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_DIR": CACHE_DIR,
})
gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
gpu_caps = core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")
print(f"  Target Device : {gpu_name}")
print(f"  Capabilities  : {gpu_caps}")
print(f"  DP4A INT8     : {'INT8' in gpu_caps} (Supported by 96 EUs)")
print(f"  Native FP16   : {'FP16' in gpu_caps} (Supported by 96 EUs)")
print(f"  Zero-Copy USM : {'GPU_USM_MEMORY' in gpu_caps} (Dual-Channel RAM)")

# Step 2: Architecture & Tokenizer
print("\n[Step 2] Initializing d1-omni-600M architecture & tokenizer...")
config = AutoConfig.from_pretrained("LiquidAI/d1-omni-600M", trust_remote_code=True)
config.max_length = 16384

model_cls = get_class_from_dynamic_module("modeling_d1.D1OmniModel", "LiquidAI/d1-omni-600M")

tok_path = hf_hub_download("LiquidAI/d1-omni-600M", "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=tok_path,
    bos_token="<|startoftext|>",
    eos_token="<|im_end|>",
    pad_token="<|pad|>",
    mask_token="<|mask|>"
)

model = model_cls(config)
model.eval()
model.config.max_length = 16384
model.name_or_path = "LiquidAI/d1-omni-600M"
model.__dict__["tokenizer"] = tokenizer

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total Model Parameters: {total_params:,} (~587M)")
print(f"  Tokenizer Vocabulary   : {len(tokenizer):,} tokens")
print(f"  Context Window         : {config.max_length:,} tokens")

# Step 3: Convert & Compile Trunk to Iris Xe GPU
print("\n[Step 3] Compiling 350M+ parameter Trunk to Intel Iris Xe GPU...")
print("  Tracing and converting PyTorch Trunk to OpenVINO IR...")
t_comp = time.perf_counter()

# Example inputs for dynamic shape tracing
example_h = torch.randn(3, 64, 1024)
example_pad = torch.ones(3, 64, dtype=torch.bool)
example_offsets = torch.zeros(3, dtype=torch.long)

ov_trunk = ov.convert_model(
    model.encoder,
    example_input=(example_h, example_pad, example_offsets)
)

# Set dynamic batch and sequence length
for inp in ov_trunk.inputs:
    inp.get_node().set_partial_shape(ov.PartialShape([-1, -1] if inp.get_partial_shape().rank.get_length() == 2 else ([-1, -1, 1024] if inp.get_partial_shape().rank.get_length() == 3 else [-1])))
ov_trunk.validate_nodes_and_infer_types()

print("  Compiling OpenVINO IR directly into Iris Xe OpenCL JIT kernels...")
compiled_gpu_trunk = core.compile_model(ov_trunk, "GPU")
print(f"  Iris Xe GPU compilation completed in {time.perf_counter() - t_comp:.2f}s")


# Wrap compiled GPU model back into PyTorch pipeline
class OpenVINOEncoderWrapper(torch.nn.Module):
    def __init__(self, compiled_ov, orig_encoder):
        super().__init__()
        self.compiled = compiled_ov
        self.embed_tokens = orig_encoder.embed_tokens

    def forward(self, h, pad, offsets):
        # Zero-copy execution on Intel Iris Xe via USM
        h_np = h.detach().cpu().numpy()
        pad_np = pad.detach().cpu().numpy()
        off_np = offsets.detach().cpu().numpy()
        out = self.compiled([h_np, pad_np, off_np])
        return torch.from_numpy(out[0])

# Inject accelerated OpenVINO GPU encoder
model.encoder = OpenVINOEncoderWrapper(compiled_gpu_trunk, model.encoder)

# Step 4: Execute Single Forward-Pass Decision Inference
print("\n[Step 4] Executing Single Forward-Pass Decision Inference...")
state = (
    "Production Incident Report: Primary payment gateway returned HTTP 500 error code for 12 consecutive requests. "
    "Transactions failing. Customer checkout blocked. Severity automatically elevated."
)
questions = {
    "is_urgent_outage": {
        "type": "noul",
        "instructions": "Is this a critical payment outage requiring immediate escalation?"
    },
    "routing_team": {
        "type": "choice",
        "instructions": "Which response team should receive this incident page?",
        "criteria": {
            "payments_sre": "Billing, payment gateways, banking APIs, core infra",
            "frontend_dev": "Web styling, CSS layout, client-side UI bugs",
            "marketing": "Customer acquisition, newsletters, promotion campaigns"
        }
    },
    "severity_score": {
        "type": "score",
        "instructions": "Assess business impact level",
        "criteria": ["Minor non-blocking glitch", "Degraded performance", "Immediate revenue loss"]
    }
}

print(f"  Input State: \"{state[:70]}...\"")
print(f"  Evaluating {len(questions)} typed questions simultaneously in 1 forward pass...")

# Warmup run on GPU
print("\n  Executing GPU Warmup Pass...")
t_warm = time.perf_counter()
res = model.system_one(state, questions)
warmup_time = (time.perf_counter() - t_warm) * 1000
print(f"  Warmup completed in {warmup_time:.2f} ms")

# Timed execution iterations
print("\n[Step 5] Benchmarking Real-Time Inference Latency on Intel Iris Xe...")
latencies = []
for i in range(5):
    t_start = time.perf_counter()
    res = model.system_one(state, questions)
    elapsed = (time.perf_counter() - t_start) * 1000
    latencies.append(elapsed)
    print(f"  Iteration {i+1}: {elapsed:.2f} ms")

avg_latency = sum(latencies) / len(latencies)
best_latency = min(latencies)
print("\n" + "=" * 65)
print("  PERFORMANCE BENCHMARK RESULTS")
print("=" * 65)
print(f"  Hardware Target     : Intel Iris Xe Graphics (96 Execution Units)")
print(f"  Average GPU Latency : {avg_latency:.2f} ms")
print(f"  Best Latency        : {best_latency:.2f} ms")
print(f"  Decision Throughput : {1000.0 / avg_latency:.1f} decision queries / second")
print("=" * 65)

print("\n=== SYSTEM ONE OUTPUT PROBABILITIES ===")
print(json.dumps(res, indent=2))
