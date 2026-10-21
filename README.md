# Liquid AI d1-omni-600M on Intel Iris Xe (96 EU)

Hardware-accelerated deployment of **Liquid AI's `d1-omni-600M`** multimodal decision model optimized specifically for **Intel Iris Xe Graphics (Xe-LP / Gen 12, 96 EUs)** and **13th Gen Intel Core i7-1360P**.

Includes an OpenVINO JIT compilation pipeline, zero-copy memory wrappers, and a live web demo interface.

---

## Highlights

- **Native Intel Iris Xe Acceleration**: Compiles the 350M+ parameter bidirectional LFM trunk to OpenCL JIT kernels targeting 96 Execution Units.
- **Hardware DP4A & FP16**: Emits hardware dot-product instructions (`cl_khr_integer_dot_product`) and IEEE 754 half-precision pipelines.
- **Zero-Copy Memory**: Uses `GPU_USM_MEMORY` (Unified Shared Memory) over dual-channel DDR4-3200 system RAM to avoid PCIe host-to-device buffer copy overhead.
- **Single Forward-Pass Latency**: **~125 ms – 290 ms** on Iris Xe GPU with **0 generated tokens** (pure structured decision extraction).
- **Interactive Web Demo**: Built-in modern web UI with animated confidence meters, multi-question rubric scoring, and live latency telemetry.

---

## Performance Benchmark

Tested on **HP Laptop 14-ep0xxx (Intel Core i7-1360P, Iris Xe 96 EUs, 16GB Dual-Channel DDR4-3200)**:

| Metric | Result |
| :--- | :--- |
| **Warmup + OpenCL JIT Emission** | ~1.18 s |
| **Single Forward-Pass Latency (Iris Xe GPU)** | **~288 ms – 296 ms** |
| **Decision Query Throughput** | **~3.4 queries / sec** |
| **Output Generated Tokens** | **0** (Immediate calibrated probability distribution) |
| **Model Size in Memory** | ~1.12 GB (FP16) |

### Sample Evaluation (Pre-Trained Weights)
```json
{
  "answers": {
    "is_urgent_outage": {
      "type": "noul",
      "noul": 0.920979
    },
    "routing_team": {
      "type": "choice",
      "choice": "payments_sre",
      "confidence": 0.997424,
      "probabilities": {
        "payments_sre": 0.997424,
        "frontend_dev": 0.001127,
        "marketing": 0.001449
      }
    },
    "severity_score": {
      "type": "score",
      "score": 1.053989,
      "confidence": 0.848866
    }
  },
  "usage": {
    "input_tokens": 240,
    "output_tokens": 0
  }
}
```

---

## Project Structure

```
d1-omni-iris-xe/
├── d1_iris_xe.py        # Core accelerated inference engine & benchmark harness
├── web_app.py           # Real-time web demo server (http://localhost:7860)
├── interactive.py       # Interactive CLI evaluator (text & image states)
├── demo_run.py          # Standalone demonstration and JIT verification script
├── requirements.txt     # Python runtime dependencies
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/ruwwww/d1-omni-iris-xe.git
cd d1-omni-iris-xe

pip install -r requirements.txt
```

### 2. Run Web Demo

```bash
python web_app.py
```
Open **`http://localhost:7860`** in your browser.

### 3. Run Benchmark

```bash
python d1_iris_xe.py --device GPU --benchmark
```

### 4. Interactive CLI

```bash
python interactive.py --state "Database connection pool exhausted. Latency spiked to 2.4s."
```

---

## Technical Architecture & Patches

1. **Direct OpenVINO IR Graph Conversion**:
   Liquid AI's architecture (`modeling_d1.py`) combines a FastConformer audio encoder, SigLIP2 vision tower, and an LFM2.5 trunk. Standard PyTorch Dynamo / Triton lowering fails on Windows; we trace and compile the LFM trunk directly into OpenVINO IR with dynamic batch and sequence dimensions (`[-1, -1, 1024]`).
2. **Zero-Copy USM Bridge (`OpenVINOEncoderWrapper`)**:
   PyTorch embeddings feed directly into OpenVINO's execution memory without host-to-device buffer duplication.
3. **On-Disk OpenCL JIT Caching**:
   Compiled GPU kernel binaries are cached in `~/.cache/openvino_iris_xe/` to guarantee instant cold starts.

---

## License

This optimization harness is provided under the MIT License. Model weights and architecture are governed by Liquid AI's model license on [Hugging Face](https://huggingface.co/LiquidAI/d1-omni-600M).
