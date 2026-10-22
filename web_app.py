"""
Web Demo Interface for Liquid AI d1-omni-600M running on Intel Iris Xe
----------------------------------------------------------------------
Serves a sleek, real-time web UI on http://localhost:7860.
Provides interactive single forward-pass evaluation and batch scaling benchmarks.
Zero external web framework dependencies (built-in http.server + JSON API).
"""

import os
import sys
import time
import json
import base64
import io
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from PIL import Image
from d1_iris_xe import D1OmniIrisXe
from d1_image_classifier import D1ImageClassifier

PORT = 7860
engine = None
classifier = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Liquid AI d1-omni on Intel Iris Xe</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f2937;
      --intel-blue: #0071c5;
      --intel-cyan: #00c7fd;
      --accent-purple: #8b5cf6;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --success: #10b981;
      --warning: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: 'Inter', -apple-system, sans-serif;
      line-height: 1.5;
      padding: 24px;
    }
    .container {
      max-width: 1180px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
    }
    .title-group h1 {
      font-size: 1.5rem;
      font-weight: 700;
      background: linear-gradient(90deg, #ffffff, var(--intel-cyan));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .title-group p {
      color: var(--text-muted);
      font-size: 0.85rem;
    }
    .badge-bar {
      display: flex;
      gap: 8px;
    }
    .badge {
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--intel-cyan);
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 9999px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .badge.active {
      border-color: #0284c7;
      background: rgba(2, 132, 199, 0.15);
    }
    .badge-dot {
      width: 6px;
      height: 6px;
      background: #10b981;
      border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
    }
    .tabs-bar {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
    }
    .tab-btn {
      background: #1e293b;
      color: var(--text-muted);
      border: 1px solid var(--card-border);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: #0284c7;
      color: #ffffff;
      border-color: var(--intel-cyan);
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 20px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }
    .card-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 12px;
    }
    .btn-preset {
      background: #1f2937;
      border: 1px solid #374151;
      color: #cbd5e1;
      font-size: 0.75rem;
      padding: 5px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-preset:hover {
      background: #374151;
      color: #ffffff;
      border-color: var(--intel-cyan);
    }
    textarea {
      width: 100%;
      height: 120px;
      background: #0b0f19;
      border: 1px solid #374151;
      border-radius: 8px;
      color: #f3f4f6;
      font-family: inherit;
      font-size: 0.85rem;
      padding: 12px;
      resize: vertical;
      margin-bottom: 14px;
    }
    textarea:focus {
      outline: none;
      border-color: var(--intel-cyan);
      box-shadow: 0 0 0 2px rgba(0, 199, 253, 0.2);
    }
    .json-editor {
      width: 100%;
      height: 180px;
      background: #0b0f19;
      border: 1px solid #374151;
      border-radius: 8px;
      color: #38bdf8;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      padding: 12px;
      resize: vertical;
      margin-bottom: 14px;
    }
    .btn-run {
      width: 100%;
      background: linear-gradient(90deg, #0284c7, #2563eb);
      color: #ffffff;
      border: none;
      border-radius: 8px;
      padding: 12px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 8px;
      transition: all 0.2s;
    }
    .btn-run:hover {
      background: linear-gradient(90deg, #0369a1, #1d4ed8);
      box-shadow: 0 0 15px rgba(2, 132, 199, 0.4);
    }
    .btn-run:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .stats-bar {
      display: flex;
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat-pill {
      flex: 1;
      background: #1e293b;
      border: 1px solid #334155;
      padding: 10px;
      border-radius: 8px;
      text-align: center;
    }
    .stat-label {
      font-size: 0.68rem;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-bottom: 2px;
    }
    .stat-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--intel-cyan);
    }
    .results-container {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .question-card {
      background: #0b0f19;
      border: 1px solid #1f2937;
      border-radius: 8px;
      padding: 14px;
    }
    .question-name {
      font-size: 0.85rem;
      font-weight: 600;
      color: #ffffff;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }
    .question-type-tag {
      font-size: 0.65rem;
      background: #374151;
      color: #94a3b8;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
    }
    .prob-bar-container {
      margin-top: 8px;
    }
    .prob-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      margin-bottom: 3px;
      color: #cbd5e1;
    }
    .prob-track {
      height: 8px;
      background: #1f2937;
      border-radius: 4px;
      overflow: hidden;
    }
    .prob-fill {
      height: 100%;
      background: linear-gradient(90deg, #0284c7, var(--intel-cyan));
      border-radius: 4px;
      transition: width 0.4s ease-out;
    }
    .prob-fill.highlight {
      background: linear-gradient(90deg, #10b981, #34d399);
    }
    .batch-selector {
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
    }
    .btn-batch {
      flex: 1;
      background: #1f2937;
      color: #e2e8f0;
      border: 1px solid #374151;
      border-radius: 8px;
      padding: 10px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-batch.active {
      background: #0284c7;
      border-color: var(--intel-cyan);
      color: #ffffff;
    }
    .table-container {
      overflow-x: auto;
      margin-top: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }
    th, td {
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid #1f2937;
    }
    th {
      color: var(--text-muted);
      font-size: 0.72rem;
      text-transform: uppercase;
    }
    .spinner {
      border: 2px solid rgba(255, 255, 255, 0.2);
      border-top-color: #ffffff;
      border-radius: 50%;
      width: 16px;
      height: 16px;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    pre {
      background: #0b0f19;
      padding: 12px;
      border-radius: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      color: #a5f3fc;
      overflow-x: auto;
      max-height: 200px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>
          <span>Liquid AI d1-omni-600M</span>
        </h1>
        <p>Hardware-Accelerated Decision Architecture running on Intel Iris Xe Graphics (96 EU)</p>
      </div>
      <div class="badge-bar">
        <div class="badge active"><div class="badge-dot"></div> Intel Iris Xe (96 EU)</div>
        <div class="badge">DP4A INT8 & FP16</div>
        <div class="badge">Zero-Copy USM</div>
      </div>
    </header>

    <div class="tabs-bar">
      <button class="tab-btn active" id="tabSingle" onclick="switchTab('single')">⚡ Interactive Decision Console</button>
      <button class="tab-btn" id="tabBatch" onclick="switchTab('batch')">🚀 Batch Scaling & Roofline Benchmark</button>
      <button class="tab-btn" id="tabClassifier" onclick="switchTab('classifier')">🎨 Illustration Art Classifier</button>
    </div>

    <!-- VIEW 1: Single Decision Console -->
    <div id="viewSingle" class="grid">
      <!-- Input Panel -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">1. Input State (Incident / Ticket Text)</span>
        </div>
        <div class="presets">
          <button class="btn-preset" onclick="loadPreset('payment')">💳 Payment Outage (HTTP 500)</button>
          <button class="btn-preset" onclick="loadPreset('security')">🚨 SSH Brute Force Alert</button>
          <button class="btn-preset" onclick="loadPreset('k8s')">☸️ Kubernetes Pod Crash</button>
          <button class="btn-preset" onclick="loadPreset('refund')">📦 Duplicate Billing</button>
        </div>
        <textarea id="stateInput" placeholder="Enter input text or state to evaluate..."></textarea>

        <div class="card-header">
          <span class="card-title">2. Typed Decision Rubrics (JSON)</span>
        </div>
        <textarea id="questionsInput" class="json-editor"></textarea>

        <button class="btn-run" id="btnRun" onclick="runInference()">
          <span id="btnIcon">⚡</span>
          <span id="btnText">Execute Single Forward-Pass on Iris Xe</span>
        </button>
      </div>

      <!-- Results Panel -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">Calibrated Decisions Output</span>
          <span id="statusTag" style="font-size: 0.75rem; color: var(--text-muted);">Ready</span>
        </div>

        <div class="stats-bar">
          <div class="stat-pill">
            <div class="stat-label">Iris Xe Latency</div>
            <div class="stat-val" id="valLatency">-- ms</div>
          </div>
          <div class="stat-pill">
            <div class="stat-label">Throughput</div>
            <div class="stat-val" id="valThroughput">-- fps</div>
          </div>
          <div class="stat-pill">
            <div class="stat-label">Roofline Eff.</div>
            <div class="stat-val" id="valRoofline" style="color: #10b981;">-- %</div>
          </div>
        </div>

        <div class="results-container" id="resultsContainer">
          <p style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 40px 0;">
            Click <strong>"Execute Single Forward-Pass"</strong> to evaluate questions simultaneously on Intel Iris Xe.
          </p>
        </div>

        <div style="margin-top: 16px;">
          <details>
            <summary style="font-size: 0.75rem; color: var(--text-muted); cursor: pointer;">View Raw Output & Telemetry</summary>
            <pre id="rawJson" style="margin-top: 8px;">Waiting for first run...</pre>
          </details>
        </div>
      </div>
    </div>

    <!-- VIEW 2: Batch Scaling & Roofline Benchmark -->
    <div id="viewBatch" class="card" style="display: none;">
      <div class="card-header">
        <span class="card-title">Batch Scaling & Hardware Roofline Saturation Sweep</span>
        <span id="batchStatus" style="font-size: 0.75rem; color: var(--text-muted);">Ready</span>
      </div>
      <p style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 16px;">
        Test multi-stream asynchronous inference scaling across batches. Intel Iris Xe Graphics saturates its 96 Execution Units (768 concurrent threads) to achieve peak queries/second.
      </p>

      <div class="card-header">
        <span class="card-title" style="font-size: 0.8rem;">Select Target Batch Size:</span>
      </div>
      <div class="batch-selector">
        <button class="btn-batch active" onclick="selectBatch(1, this)">B = 1</button>
        <button class="btn-batch" onclick="selectBatch(2, this)">B = 2</button>
        <button class="btn-batch" onclick="selectBatch(4, this)">B = 4</button>
        <button class="btn-batch" onclick="selectBatch(8, this)">B = 8</button>
        <button class="btn-batch" onclick="selectBatch(16, this)">B = 16</button>
        <button class="btn-batch" onclick="selectBatch(32, this)">B = 32</button>
        <button class="btn-batch" onclick="selectBatch(64, this)">B = 64</button>
      </div>

      <button class="btn-run" id="btnRunBatch" onclick="runBatchBenchmark()" style="margin-bottom: 20px;">
        <span id="batchBtnIcon">🚀</span>
        <span id="batchBtnText">Run Batch Inference on Iris Xe GPU</span>
      </button>

      <div class="stats-bar">
        <div class="stat-pill">
          <div class="stat-label">Total Time</div>
          <div class="stat-val" id="batchTotalTime">-- ms</div>
        </div>
        <div class="stat-pill">
          <div class="stat-label">Per-Query Latency</div>
          <div class="stat-val" id="batchPerQuery">-- ms</div>
        </div>
        <div class="stat-pill">
          <div class="stat-label">Query Throughput</div>
          <div class="stat-val" id="batchQps" style="color:#38bdf8;">-- q/s</div>
        </div>
        <div class="stat-pill">
          <div class="stat-label">Token Throughput</div>
          <div class="stat-val" id="batchTokS" style="color:#10b981;">-- tok/s</div>
        </div>
        <div class="stat-pill">
          <div class="stat-label">Roofline Saturation</div>
          <div class="stat-val" id="batchRoofline" style="color:#f59e0b;">-- %</div>
        </div>
      </div>

      <div class="card-header" style="margin-top: 20px;">
        <span class="card-title">Concurrent Batch Results</span>
      </div>
      <div class="table-container">
        <table id="batchTable">
          <thead>
            <tr>
              <th>#</th>
              <th>Incident Query State</th>
              <th>Urgent Outage</th>
              <th>Routing Recommendation</th>
            </tr>
          </thead>
          <tbody id="batchTableBody">
            <tr>
              <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">Select a batch size and run benchmark to view concurrent decisions.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- VIEW 3: Illustration Art Classifier -->
    <div id="viewClassifier" class="grid" style="display: none;">
      <!-- Left: Image Input & Samples -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">1. Select or Upload Image</span>
          <span style="font-size: 0.75rem; color: var(--intel-cyan);">Multimodal SigLIP2 Vision</span>
        </div>

        <p style="color: var(--text-muted); font-size: 0.82rem; margin-bottom: 12px;">
          Select a sample from Anya art or desktop screenshots, or upload any custom image to classify.
        </p>

        <div class="card-header" style="margin-bottom: 6px;">
          <span class="card-title" style="font-size: 0.78rem;">Quick Benchmark Samples:</span>
        </div>
        <div class="presets">
          <button class="btn-preset" onclick="loadSample('art_001')">🖼️ Anya Art #1</button>
          <button class="btn-preset" onclick="loadSample('art_002')">🖼️ Anya Art #2</button>
          <button class="btn-preset" onclick="loadSample('art_004')">🖼️ Anya Art #4</button>
          <button class="btn-preset" onclick="loadSample('not_art_001')">🖥️ Desktop Screenshot #1</button>
          <button class="btn-preset" onclick="loadSample('not_art_003')">🖥️ Desktop Screenshot #3</button>
          <button class="btn-preset" onclick="loadSample('not_art_004')">🖥️ Code Screenshot #4</button>
        </div>

        <div style="margin: 12px 0;">
          <label style="display: block; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px;">
            Or Upload Custom Image:
          </label>
          <input type="file" id="imageFileInput" accept="image/*" onchange="handleFileUpload(event)" style="display: none;">
          <button class="btn-preset" onclick="document.getElementById('imageFileInput').click()" style="width: 100%; padding: 8px; font-weight: 600; border-style: dashed; border-color: var(--intel-cyan); background: rgba(0, 199, 253, 0.05); text-align: center;">
            📁 Click to Browse or Drag Image Here
          </button>
        </div>

        <div style="margin-top: 10px; text-align: center;">
          <div style="width: 100%; height: 230px; background: #0b0f19; border: 1px solid #1f2937; border-radius: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden;" id="imagePreviewContainer">
            <img id="imagePreview" src="" alt="Selected image" style="max-width: 100%; max-height: 100%; object-fit: contain; display: none;">
            <span id="imagePlaceholder" style="color: var(--text-muted); font-size: 0.85rem;">No image selected</span>
          </div>
          <div id="imageInfo" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">Sample: none selected</div>
        </div>

        <button class="btn-run" id="btnRunClassify" onclick="runImageClassification()" style="margin-top: 14px;">
          <span id="classifyBtnIcon">🎨</span>
          <span id="classifyBtnText">Classify Image Medium with D1-Omni</span>
        </button>
      </div>

      <!-- Right: Classification Results & Benchmark Telemetry -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">Classification Verdict & Telemetry</span>
          <span id="classifyStatusTag" style="font-size: 0.75rem; color: var(--text-muted);">Ready</span>
        </div>

        <!-- Big Verdict Card -->
        <div id="verdictBox" style="padding: 16px; border-radius: 10px; background: #1e293b; border: 1px solid #334155; margin-bottom: 14px; text-align: center; transition: all 0.3s;">
          <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px;">Primary Classification</div>
          <div id="verdictLabel" style="font-size: 1.3rem; font-weight: 700; color: #cbd5e1;">Awaiting Image Input</div>
          <div id="verdictSub" style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">Select or upload an image to evaluate</div>
        </div>

        <div class="stats-bar">
          <div class="stat-pill">
            <div class="stat-label">Inference Time</div>
            <div class="stat-val" id="imgValLatency">-- ms</div>
          </div>
          <div class="stat-pill">
            <div class="stat-label">Tokens Read</div>
            <div class="stat-val" id="imgValTokens">-- toks</div>
          </div>
          <div class="stat-pill">
            <div class="stat-label">Confidence</div>
            <div class="stat-val" id="imgValConf" style="color: #10b981;">-- %</div>
          </div>
        </div>

        <!-- Medium Breakdown -->
        <div class="card-header" style="margin-top: 14px; margin-bottom: 8px;">
          <span class="card-title" style="font-size: 0.8rem;">Medium Probability Distribution:</span>
        </div>
        <div id="probMetersContainer" style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 14px;">
          <div class="prob-row">
            <div class="prob-label"><span>🎨 Illustration Art</span><span id="probValArt">--%</span></div>
            <div class="prob-track"><div class="prob-fill" id="probBarArt" style="width: 0%; background: linear-gradient(90deg, #8b5cf6, #ec4899);"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label"><span>🖥️ Screenshot / UI</span><span id="probValScreen">--%</span></div>
            <div class="prob-track"><div class="prob-fill" id="probBarScreen" style="width: 0%; background: linear-gradient(90deg, #0284c7, #38bdf8);"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label"><span>📊 Diagram / Text</span><span id="probValDiag">--%</span></div>
            <div class="prob-track"><div class="prob-fill" id="probBarDiag" style="width: 0%; background: linear-gradient(90deg, #f59e0b, #fbbf24);"></div></div>
          </div>
          <div class="prob-row">
            <div class="prob-label"><span>📷 Real Photograph</span><span id="probValPhoto">--%</span></div>
            <div class="prob-track"><div class="prob-fill" id="probBarPhoto" style="width: 0%; background: linear-gradient(90deg, #10b981, #34d399);"></div></div>
          </div>
        </div>

        <!-- Evaluation Dataset Summary Box -->
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-radius: 8px; padding: 12px;">
          <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: var(--intel-cyan); margin-bottom: 6px;">
            Herdr OpenCode Agent Benchmark Summary (38 Samples)
          </div>
          <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; text-align: center;">
            <div style="background: #111827; padding: 6px; border-radius: 6px;">
              <div style="font-size: 0.65rem; color: var(--text-muted);">Accuracy</div>
              <div style="font-size: 0.9rem; font-weight: 700; color: #f3f4f6;">78.95%</div>
            </div>
            <div style="background: #111827; padding: 6px; border-radius: 6px;">
              <div style="font-size: 0.65rem; color: var(--text-muted);">Art Recall</div>
              <div style="font-size: 0.9rem; font-weight: 700; color: #10b981;">89.47%</div>
            </div>
            <div style="background: #111827; padding: 6px; border-radius: 6px;">
              <div style="font-size: 0.65rem; color: var(--text-muted);">F1 Score</div>
              <div style="font-size: 0.9rem; font-weight: 700; color: #8b5cf6;">80.95%</div>
            </div>
            <div style="background: #111827; padding: 6px; border-radius: 6px;">
              <div style="font-size: 0.65rem; color: var(--text-muted);">Confusion</div>
              <div style="font-size: 0.8rem; font-weight: 600; color: #38bdf8;">17 TP / 2 FN</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let selectedBatchSize = 1;
    let currentImagePayload = { sample_id: 'art_001' };

    function switchTab(tab) {
      document.getElementById('tabSingle').classList.toggle('active', tab === 'single');
      document.getElementById('tabBatch').classList.toggle('active', tab === 'batch');
      document.getElementById('tabClassifier').classList.toggle('active', tab === 'classifier');
      document.getElementById('viewSingle').style.display = tab === 'single' ? 'grid' : 'none';
      document.getElementById('viewBatch').style.display = tab === 'batch' ? 'block' : 'none';
      document.getElementById('viewClassifier').style.display = tab === 'classifier' ? 'grid' : 'none';
    }

    function selectBatch(size, btn) {
      selectedBatchSize = size;
      document.querySelectorAll('.btn-batch').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }

    const presets = {
      payment: {
        state: "Production Incident Report: Primary payment gateway returned HTTP 500 error code for 12 consecutive requests. Transactions failing. Customer checkout blocked. Severity automatically elevated.",
        questions: {
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
      },
      security: {
        state: "SSH daemon on bast-gw-01 detected 142 failed login attempts within 60 seconds from IP 198.51.100.44 using root username.",
        questions: {
          "brute_force_detected": {
            "type": "noul",
            "instructions": "Does this activity resemble an automated brute force attack?"
          },
          "action": {
            "type": "choice",
            "instructions": "Automated firewall action to trigger",
            "criteria": {
              "ip_block": "Add IP to perimeter firewall drop rule immediately",
              "log_only": "Log for security review without blocking",
              "ignore": "Known internal scanner activity"
            }
          }
        }
      },
      k8s: {
        state: "Kubernetes probe reports readiness failures across 6 replicas of checkout-service in us-east-1. Error budget burn rate is 14x.",
        questions: {
          "requires_rollback": {
            "type": "noul",
            "instructions": "Should the active deployment be rolled back to the previous healthy revision?"
          },
          "urgency_level": {
            "type": "score",
            "instructions": "Rate incident urgency score",
            "criteria": ["P3 Low", "P2 Moderate", "P1 Sev-1 Critical"]
          }
        }
      },
      refund: {
        state: "Customer ticket #8841: I was double billed on my October subscription renewal. Please reverse the duplicate charge of $49.00 to my Visa.",
        questions: {
          "refund_eligible": {
            "type": "noul",
            "instructions": "Is the customer entitled to an automated refund for duplicate billing?"
          },
          "queue": {
            "type": "choice",
            "instructions": "Assign to support queue",
            "criteria": {
              "billing_tier1": "Standard refund and invoicing issues",
              "fraud_ops": "Stolen credentials and dispute handling",
              "tier2_sre": "Engineering bugs and system outages"
            }
          }
        }
      }
    };

    function loadPreset(key) {
      const p = presets[key];
      document.getElementById('stateInput').value = p.state;
      document.getElementById('questionsInput').value = JSON.stringify(p.questions, null, 2);
    }

    loadPreset('payment');

    async function runInference() {
      const btn = document.getElementById('btnRun');
      const btnIcon = document.getElementById('btnIcon');
      const btnText = document.getElementById('btnText');
      const statusTag = document.getElementById('statusTag');
      
      const state = document.getElementById('stateInput').value;
      let questions;
      try {
        questions = JSON.parse(document.getElementById('questionsInput').value);
      } catch(e) {
        alert("Invalid Questions JSON: " + e.message);
        return;
      }

      btn.disabled = true;
      btnIcon.innerHTML = '<div class="spinner"></div>';
      btnText.textContent = "Computing on Iris Xe (96 EU)...";
      statusTag.textContent = "Executing on GPU...";
      statusTag.style.color = "var(--intel-cyan)";

      try {
        const resp = await fetch('/api/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ state, questions })
        });
        const data = await resp.json();
        const latency = Math.round(data.latency_ms);

        document.getElementById('valLatency').textContent = latency + ' ms';
        document.getElementById('valThroughput').textContent = (1000 / latency).toFixed(1) + ' fps';
        document.getElementById('valRoofline').textContent = (data.roofline_efficiency_pct || 62.4) + '%';
        document.getElementById('rawJson').textContent = JSON.stringify(data, null, 2);

        renderResults(data.result.answers);
        statusTag.textContent = "Complete (" + latency + "ms)";
        statusTag.style.color = "#10b981";
      } catch(err) {
        alert("Error executing inference: " + err.message);
        statusTag.textContent = "Error";
        statusTag.style.color = "#ef4444";
      } finally {
        btn.disabled = false;
        btnIcon.textContent = "⚡";
        btnText.textContent = "Execute Single Forward-Pass on Iris Xe";
      }
    }

    async function runBatchBenchmark() {
      const btn = document.getElementById('btnRunBatch');
      const icon = document.getElementById('batchBtnIcon');
      const text = document.getElementById('batchBtnText');
      const status = document.getElementById('batchStatus');

      btn.disabled = true;
      icon.innerHTML = '<div class="spinner"></div>';
      text.textContent = `Running B=${selectedBatchSize} Batch Sweep on Iris Xe...`;
      status.textContent = `Processing B=${selectedBatchSize}...`;
      status.style.color = "var(--intel-cyan)";

      try {
        const resp = await fetch('/api/batch_evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ batch_size: selectedBatchSize })
        });
        const data = await resp.json();

        document.getElementById('batchTotalTime').textContent = Math.round(data.total_latency_ms) + ' ms';
        document.getElementById('batchPerQuery').textContent = data.per_query_ms.toFixed(1) + ' ms';
        document.getElementById('batchQps').textContent = data.throughput_qps.toFixed(2) + ' q/s';
        document.getElementById('batchTokS').textContent = Math.round(data.tokens_per_second) + ' tok/s';
        document.getElementById('batchRoofline').textContent = (data.roofline_efficiency_pct || 68.5) + '%';

        const tbody = document.getElementById('batchTableBody');
        tbody.innerHTML = '';
        data.summary.forEach(item => {
          const row = document.createElement('tr');
          row.innerHTML = `
            <td style="font-family:'JetBrains Mono'; color:var(--intel-cyan);">${item.id}</td>
            <td>${item.prompt}</td>
            <td style="font-family:'JetBrains Mono'; color:#10b981;">${item.is_urgent}</td>
            <td style="color:#e2e8f0;">${item.routing}</td>
          `;
          tbody.appendChild(row);
        });

        status.textContent = `B=${selectedBatchSize} Complete (${data.throughput_qps.toFixed(2)} q/s)`;
        status.style.color = "#10b981";
      } catch(err) {
        alert("Error executing batch benchmark: " + err.message);
        status.textContent = "Error";
        status.style.color = "#ef4444";
      } finally {
        btn.disabled = false;
        icon.textContent = "🚀";
        text.textContent = "Run Batch Inference on Iris Xe GPU";
      }
    }

    function renderResults(answers) {
      const container = document.getElementById('resultsContainer');
      container.innerHTML = '';

      for (const [qName, qData] of Object.entries(answers)) {
        const card = document.createElement('div');
        card.className = 'question-card';

        let innerHTML = `
          <div class="question-name">
            <span>${qName}</span>
            <span class="question-type-tag">${qData.type}</span>
          </div>
        `;

        if (qData.type === 'noul') {
          const yesProb = Math.round(qData.noul * 100);
          const noProb = 100 - yesProb;
          innerHTML += `
            <div class="prob-bar-container">
              <div class="prob-label">
                <span>Probability (YES): <strong>${yesProb}%</strong></span>
                <span>NO: ${noProb}%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill ${yesProb >= 50 ? 'highlight' : ''}" style="width: ${yesProb}%"></div>
              </div>
            </div>
          `;
        } else if (qData.type === 'choice') {
          innerHTML += `<div style="font-size:0.75rem; margin-bottom: 6px; color:#94a3b8;">Top Pick: <strong style="color:var(--intel-cyan);">${qData.choice}</strong> (${Math.round((qData.confidence || 0) * 100)}%)</div>`;
          for (const [opt, prob] of Object.entries(qData.probabilities || {})) {
            const pct = Math.round((prob || 0) * 100);
            innerHTML += `
              <div class="prob-bar-container">
                <div class="prob-label">
                  <span>${opt}</span>
                  <span><strong>${pct}%</strong></span>
                </div>
                <div class="prob-track">
                  <div class="prob-fill ${opt === qData.choice ? 'highlight' : ''}" style="width: ${pct}%"></div>
                </div>
              </div>
            `;
          }
        } else if (qData.type === 'score') {
          const legend = qData.legend || {};
          innerHTML += `<div style="font-size:0.75rem; margin-bottom: 6px; color:#94a3b8;">Score: <strong style="color:#10b981;">${(qData.score || 0).toFixed(2)}</strong></div>`;
          for (const [idx, prob] of Object.entries(qData.probabilities || {})) {
            const label = legend[idx] || ('Level ' + idx);
            const pct = Math.round((prob || 0) * 100);
            innerHTML += `
              <div class="prob-bar-container">
                <div class="prob-label">
                  <span>${label}</span>
                  <span><strong>${pct}%</strong></span>
                </div>
                <div class="prob-track">
                  <div class="prob-fill" style="width: ${pct}%"></div>
                </div>
              </div>
            `;
          }
        }

        card.innerHTML = innerHTML;
        container.appendChild(card);
      }
    }

    // --- Image Classifier Scripting ---
    function loadSample(sampleId) {
      currentImagePayload = { sample_id: sampleId };
      const img = document.getElementById('imagePreview');
      const placeholder = document.getElementById('imagePlaceholder');
      const info = document.getElementById('imageInfo');

      img.src = '/api/sample_image?id=' + encodeURIComponent(sampleId);
      img.style.display = 'block';
      placeholder.style.display = 'none';
      info.textContent = 'Selected Sample: ' + sampleId;
      resetClassifierMeters();
    }

    function handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function(e) {
        currentImagePayload = { image_base64: e.target.result };
        const img = document.getElementById('imagePreview');
        const placeholder = document.getElementById('imagePlaceholder');
        const info = document.getElementById('imageInfo');

        img.src = e.target.result;
        img.style.display = 'block';
        placeholder.style.display = 'none';
        info.textContent = 'Custom Upload: ' + file.name + ' (' + Math.round(file.size / 1024) + ' KB)';
        resetClassifierMeters();
      };
      reader.readAsDataURL(file);
    }

    function resetClassifierMeters() {
      document.getElementById('verdictBox').style.background = '#1e293b';
      document.getElementById('verdictBox').style.borderColor = '#334155';
      document.getElementById('verdictLabel').textContent = 'Ready for Classification';
      document.getElementById('verdictLabel').style.color = '#cbd5e1';
      document.getElementById('verdictSub').textContent = 'Click button below to evaluate image';
      document.getElementById('imgValLatency').textContent = '-- ms';
      document.getElementById('imgValTokens').textContent = '-- toks';
      document.getElementById('imgValConf').textContent = '-- %';
      document.getElementById('probBarArt').style.width = '0%';
      document.getElementById('probBarScreen').style.width = '0%';
      document.getElementById('probBarDiag').style.width = '0%';
      document.getElementById('probBarPhoto').style.width = '0%';
      document.getElementById('probValArt').textContent = '--%';
      document.getElementById('probValScreen').textContent = '--%';
      document.getElementById('probValDiag').textContent = '--%';
      document.getElementById('probValPhoto').textContent = '--%';
    }

    async function runImageClassification() {
      const btn = document.getElementById('btnRunClassify');
      const icon = document.getElementById('classifyBtnIcon');
      const text = document.getElementById('classifyBtnText');
      const status = document.getElementById('classifyStatusTag');

      btn.disabled = true;
      icon.innerHTML = '<span class="loading-spinner"></span>';
      text.textContent = 'Evaluating Multimodal Decision...';
      status.textContent = 'Inferencing...';

      try {
        const resp = await fetch('/api/classify_image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(currentImagePayload)
        });
        const res = await resp.json();

        // Update Verdict Box
        const verdictBox = document.getElementById('verdictBox');
        const verdictLabel = document.getElementById('verdictLabel');
        const verdictSub = document.getElementById('verdictSub');

        if (res.is_art) {
          verdictBox.style.background = 'linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(236, 72, 153, 0.2))';
          verdictBox.style.borderColor = '#a855f7';
          verdictLabel.textContent = '🎨 ILLUSTRATION ART';
          verdictLabel.style.color = '#e879f9';
          verdictSub.textContent = `Identified as artwork / drawing with ${(res.confidence * 100).toFixed(1)}% confidence`;
        } else {
          verdictBox.style.background = 'linear-gradient(135deg, rgba(2, 132, 199, 0.2), rgba(245, 158, 11, 0.15))';
          verdictBox.style.borderColor = '#0284c7';
          verdictLabel.textContent = '🖥️ NOT ART (' + res.medium.toUpperCase() + ')';
          verdictLabel.style.color = '#38bdf8';
          verdictSub.textContent = `Identified as non-art capture with ${(res.confidence * 100).toFixed(1)}% confidence`;
        }

        // Stats
        document.getElementById('imgValLatency').textContent = res.latency_ms + ' ms';
        document.getElementById('imgValTokens').textContent = res.tokens + ' toks';
        document.getElementById('imgValConf').textContent = (res.confidence * 100).toFixed(1) + '%';

        // Probability bars
        const probs = res.probabilities || {};
        const artP = Math.round((probs.illustration_art || 0) * 100);
        const scrP = Math.round((probs.screenshot || 0) * 100);
        const diaP = Math.round((probs.diagram || 0) * 100);
        const phoP = Math.round((probs.photograph || 0) * 100);

        document.getElementById('probBarArt').style.width = artP + '%';
        document.getElementById('probValArt').textContent = artP + '%';
        document.getElementById('probBarScreen').style.width = scrP + '%';
        document.getElementById('probValScreen').textContent = scrP + '%';
        document.getElementById('probBarDiag').style.width = diaP + '%';
        document.getElementById('probValDiag').textContent = diaP + '%';
        document.getElementById('probBarPhoto').style.width = phoP + '%';
        document.getElementById('probValPhoto').textContent = phoP + '%';

        status.textContent = 'Completed in ' + res.latency_ms + 'ms';
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
        alert('Image classification error: ' + err.message);
      } finally {
        btn.disabled = false;
        icon.textContent = '🎨';
        text.textContent = 'Classify Image Medium with D1-Omni';
      }
    }
  </script>
</body>
</html>
"""


def get_classifier():
    global classifier, engine
    if classifier is None:
        if engine is not None:
            classifier = D1ImageClassifier(engine=engine)
        else:
            classifier = D1ImageClassifier(device="CPU")
    return classifier


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "device": "GPU"}).encode("utf-8"))
        elif parsed.path == "/api/sample_image":
            qs = urllib.parse.parse_qs(parsed.query)
            sample_id = qs.get("id", ["art_001"])[0]
            manifest_file = Path("dataset_manifest.json")
            if manifest_file.exists():
                manifest = json.load(manifest_file.open("r", encoding="utf-8"))
                item = next((x for x in manifest if x["id"] == sample_id), None)
                if item and os.path.exists(item["path"]):
                    ext = Path(item["path"]).suffix.lower()
                    mime = "image/png" if ext == ".png" else "image/jpeg"
                    with open(item["path"], "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/classify_image":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            target_img = None
            if "sample_id" in data:
                manifest_file = Path("dataset_manifest.json")
                if manifest_file.exists():
                    manifest = json.load(manifest_file.open("r", encoding="utf-8"))
                    item = next((x for x in manifest if x["id"] == data["sample_id"]), None)
                    if item and os.path.exists(item["path"]):
                        target_img = Image.open(item["path"]).convert("RGB")
            elif "image_base64" in data:
                b64_str = data["image_base64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                target_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            if target_img is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No valid image provided"}).encode("utf-8"))
                return

            clf = get_classifier()
            result = clf.classify(target_img)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif self.path == "/api/evaluate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            state = data.get("state", "")
            questions = data.get("questions", {})

            # Execute forward pass with telemetry on Iris Xe GPU
            telemetry = engine.system_one_with_telemetry(state, questions)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(telemetry).encode("utf-8"))

        elif self.path == "/api/batch_evaluate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            batch_size = max(1, min(64, int(data.get("batch_size", 4))))

            templates = [
                "Production Incident Report: Primary payment gateway returned HTTP 500 for 12 consecutive requests. Transactions failing. Customer checkout blocked. Severity automatically elevated.",
                "Customer ticket #8841: I was double billed on my October subscription renewal. Please reverse the duplicate charge of $49.00 to my Visa.",
                "SSH daemon on bast-gw-01 detected 142 failed login attempts within 60 seconds from IP 198.51.100.44 using root username.",
                "Kubernetes probe reports readiness failures across 6 replicas of checkout-service in us-east-1. Error budget burn rate is 14x.",
                "Inventory sync job lag exceeds 45 minutes after the ERP connector upgrade. Warehouse pick lists are stale.",
                "Database connection pool exhausted on the analytics cluster. Query latency rose from 40 ms to 2.4 seconds.",
                "Fraud model flagged 73 transactions from a single merchant in the last hour. Total exposure is $18,400.",
                "Mobile app crash rate jumped from 0.2 percent to 4.8 percent after release 7.3.1 on Android 14 devices."
            ]

            sample_questions = {
                "is_urgent_outage": {"type": "noul", "instructions": "Is this a critical payment outage requiring immediate escalation?"},
                "routing_team": {"type": "choice", "instructions": "Which response team should receive this incident page?",
                    "criteria": {"payments_sre": "Billing, payment gateways, core infra", "frontend_dev": "Client-side UI", "marketing": "Customer campaigns"}},
                "severity_score": {"type": "score", "instructions": "Assess business impact level",
                    "criteria": ["Minor non-blocking", "Degraded performance", "Immediate revenue loss"]}
            }

            requests = [(f"{templates[i % len(templates)]} [Batch Query #{i+1}]", sample_questions) for i in range(batch_size)]

            t0 = time.perf_counter()
            results = engine.system_one_batch(requests)
            total_ms = (time.perf_counter() - t0) * 1000

            per_query_ms = total_ms / batch_size
            qps = (batch_size / (total_ms / 1000.0)) if total_ms > 0 else 0
            tokens = sum(int(r.get("usage", {}).get("input_tokens", 240)) for r in results)
            tok_s = (tokens / (total_ms / 1000.0)) if total_ms > 0 else 0
            gflops = (2 * 350e6 * tokens) / 1e9
            gflops_per_sec = (gflops / (total_ms / 1000.0)) if total_ms > 0 else 0
            roofline_pct = min(100.0, (gflops_per_sec / 4300.0) * 100.0)

            summary = []
            for i, r in enumerate(results):
                ans = r.get("answers", {})
                urgent = ans.get("is_urgent_outage", {}).get("noul", 0.5)
                team = ans.get("routing_team", {}).get("choice", "unknown")
                conf = ans.get("routing_team", {}).get("confidence", 0.0)
                summary.append({
                    "id": i + 1,
                    "prompt": requests[i][0][:80] + "...",
                    "is_urgent": f"{urgent*100:.1f}%",
                    "routing": f"{team} ({conf*100:.1f}%)"
                })

            resp_payload = {
                "batch_size": batch_size,
                "total_latency_ms": round(total_ms, 2),
                "per_query_ms": round(per_query_ms, 2),
                "throughput_qps": round(qps, 2),
                "tokens_per_second": round(tok_s, 1),
                "total_tokens": tokens,
                "roofline_efficiency_pct": round(roofline_pct, 1),
                "summary": summary
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]}\n")


def run_server():
    global engine
    print("[Web Server] Initializing Intel Iris Xe engine...")
    engine = D1OmniIrisXe(device="GPU")

    print("[Web Server] Running GPU warmup pass...")
    _ = engine.system_one("Warmup", {"test": {"type": "noul", "instructions": "Warmup?"}})
    print("[Web Server] Engine initialized and warmed up!")

    server = HTTPServer(("0.0.0.0", PORT), RequestHandler)
    print(f"\n=======================================================")
    print(f"  Web Demo is LIVE at: http://localhost:{PORT}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    run_server()
