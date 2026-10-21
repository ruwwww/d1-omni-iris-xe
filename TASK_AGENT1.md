# Task for Agent 1: Batch = 1 Ultra-Low Latency Optimization on Intel Iris Xe

You are the **Latency Specialist** for the `d1-omni-iris-xe` project in `C:\Users\darlis\d1-omni-iris-xe`.

## Context
We have `LiquidAI/d1-omni-600M` running on Intel Iris Xe (96 EUs) via OpenVINO (`d1_iris_xe.py`).
Currently, Batch = 1 latency is ~280ms (cold/warm), but the theoretical Roofline limit for FP16/INT8 on this hardware is ~20–35ms. We are losing time to Python dispatch, dynamic shapes, and buffer conversions.

## Your Mission
1. Inspect `d1_iris_xe.py`.
2. Implement **Zero-Copy Tensor Binding**:
   - In `OpenVINOEncoderWrapper.forward`, replace unnecessary conversions with zero-copy shared memory:
     Use `ov.Tensor(array, shared_memory=True)` or wrap PyTorch memory directly so no buffer reallocation occurs.
3. Implement **Static Shape Bucket Compilation**:
   - OpenVINO dynamic shapes `[-1, -1, 1024]` cause generic OpenCL loop bounds.
   - For Batch = 1, create a bucketed cache of compiled models for standard sequence lengths: $L \in [64, 128, 256]$.
   - Pad inputs to the nearest bucket so OpenCL compiles fixed-size register-allocated kernels with maximum EU vectorization.
4. Profile Sub-Module Breakdown:
   - Measure exactly where time is spent: Tokenizer encoding vs Embeddings vs GPU Trunk vs Head.
5. Create `latency_opt.py`:
   - Implement these optimizations.
   - Benchmark Batch = 1 latency before and after.
   - Verify that output probabilities match `d1_iris_xe.py` outputs.
6. When finished, print a summary of your results and timings.
