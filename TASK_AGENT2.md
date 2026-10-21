# Task for Agent 2: Large Batch Scaling & Throughput Saturation (Batch 8 to 256)

You are the **Throughput & Batch Scaling Specialist** for the `d1-omni-iris-xe` project in `C:\Users\darlis\d1-omni-iris-xe`.

## Context
We have `LiquidAI/d1-omni-600M` running on Intel Iris Xe (96 EUs) via OpenVINO (`d1_iris_xe.py`).
In production decision-routing pipelines, requests arrive concurrently in batches.
At large batches ($B \ge 8$), arithmetic intensity increases dramatically, allowing Iris Xe to saturate its 96 EUs (and 12-core CPU).

## Your Mission
1. Inspect `d1_iris_xe.py`.
2. Build `throughput_opt.py` featuring:
   - **Multi-Stream Asynchronous Execution**:
     Use OpenVINO's `ov.AsyncInferQueue` configured with `PERFORMANCE_HINT = "THROUGHPUT"` and auto-streams (`ov.properties.streams.num(ov.properties.streams.Num.AUTO)`).
   - **Dual-Device Heterogeneous Execution**:
     Compare:
     - `device = "GPU"` (Iris Xe 96 EUs)
     - `device = "CPU"` (13th Gen Core i7-1360P 12 cores with AVX-VNNI)
     - `device = "MULTI:GPU,CPU"` (Heterogeneous pipeline splitting large batches across both GPU and CPU simultaneously)
3. Implement an Automated Benchmark Sweep:
   - Test batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128, 256]`.
   - Measure for each batch:
     - Total Latency (ms)
     - Per-query Latency (ms)
     - Throughput (decision queries / second)
     - Memory footprint (MB)
4. Output Artifacts:
   - Save full JSON metrics to `benchmark_results.json`.
   - Generate a clean Markdown summary table with comparisons in `BENCHMARK_SWEEP.md`.
5. When finished, print a summary of your results and throughput scaling curves.
