"""
Web Demo Interface for Liquid AI d1-omni-600M running on Intel Iris Xe
----------------------------------------------------------------------
Serves a sleek, real-time web UI on http://localhost:7860.
Zero external web framework dependencies (built-in http.server + JSON API).
"""

import os
import sys
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
from d1_iris_xe import D1OmniIrisXe

PORT = 7860
engine = None

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
      max-width: 1100px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
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
      gap: 12px;
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
      font-size: 0.7rem;
      color: var(--text-muted);
      text-transform: uppercase;
    }
    .stat-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.1rem;
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
        <p>Accelerated Decision Engine running natively on Intel Iris Xe Graphics (Xe-LP 96 EU)</p>
      </div>
      <div class="badge-bar">
        <div class="badge active"><div class="badge-dot"></div> Intel Iris Xe (96 EU)</div>
        <div class="badge">DP4A INT8</div>
        <div class="badge">OpenVINO JIT</div>
      </div>
    </header>

    <div class="grid">
      <!-- Input Panel -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">1. Input State (Text / JSON)</span>
        </div>
        <div class="presets">
          <button class="btn-preset" onclick="loadPreset('payment')">💳 Payment 500 Outage</button>
          <button class="btn-preset" onclick="loadPreset('refund')">📦 Refund Request</button>
          <button class="btn-preset" onclick="loadPreset('security')">🚨 Security Alert</button>
          <button class="btn-preset" onclick="loadPreset('flight')">✈️ Flight Delay</button>
        </div>
        <textarea id="stateInput" placeholder="Enter input text or state to evaluate..."></textarea>

        <div class="card-header">
          <span class="card-title">2. Typed Decision Questions (JSON)</span>
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
          <span class="card-title">Hardware Decision Output</span>
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
            <div class="stat-label">Generated Tokens</div>
            <div class="stat-val" style="color: #10b981;">0</div>
          </div>
        </div>

        <div class="results-container" id="resultsContainer">
          <p style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 40px 0;">
            Click <strong>"Execute Single Forward-Pass"</strong> to evaluate questions simultaneously on Intel Iris Xe.
          </p>
        </div>

        <div style="margin-top: 16px;">
          <details>
            <summary style="font-size: 0.75rem; color: var(--text-muted); cursor: pointer;">View Raw System One Output</summary>
            <pre id="rawJson" style="margin-top: 8px;">Waiting for first run...</pre>
          </details>
        </div>
      </div>
    </div>
  </div>

  <script>
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
      refund: {
        state: "Customer ticket #8841: I was double billed on my October subscription renewal. Please reverse the duplicate charge of $49.00 to my Visa.",
        questions: {
          "refund_requested": {
            "type": "noul",
            "instructions": "Is the customer explicitly asking for monetary refund?"
          },
          "category": {
            "type": "choice",
            "instructions": "Classify ticket department",
            "criteria": {
              "billing": "Invoice disputes, double charges, refunds",
              "technical": "Software bugs and service errors",
              "sales": "Upgrades and plan changes"
            }
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
      flight: {
        state: "Passenger Smith on flight AA1290 from DFW to ORD: Flight cancelled due to mechanical failure. Stranded overnight.",
        questions: {
          "eligible_hotel": {
            "type": "noul",
            "instructions": "Is passenger entitled to complimentary hotel accommodations under airline policy?"
          },
          "urgency": {
            "type": "score",
            "instructions": "Customer compensation priority rating",
            "criteria": ["Routine", "High priority", "VIP immediate escalation"]
          }
        }
      }
    };

    function loadPreset(key) {
      const p = presets[key];
      document.getElementById('stateInput').value = p.state;
      document.getElementById('questionsInput').value = JSON.stringify(p.questions, null, 2);
    }

    // Load default
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

      const t0 = performance.now();
      try {
        const resp = await fetch('/api/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ state, questions })
        });
        const data = await resp.json();
        const clientLatency = Math.round(performance.now() - t0);
        const serverLatency = data.latency_ms ? Math.round(data.latency_ms) : clientLatency;

        document.getElementById('valLatency').textContent = serverLatency + ' ms';
        document.getElementById('valThroughput').textContent = (1000 / serverLatency).toFixed(1) + ' fps';
        document.getElementById('rawJson').textContent = JSON.stringify(data.result, null, 2);

        renderResults(data.result.answers);
        statusTag.textContent = "Complete (" + serverLatency + "ms)";
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
          innerHTML += `<div style="font-size:0.75rem; margin-bottom: 6px; color:#94a3b8;">Top Pick: <strong style="color:var(--intel-cyan);">${qData.choice}</strong> (${Math.round(qData.confidence * 100)}%)</div>`;
          for (const [opt, prob] of Object.entries(qData.probabilities || {})) {
            const pct = Math.round(prob * 100);
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
          innerHTML += `<div style="font-size:0.75rem; margin-bottom: 6px; color:#94a3b8;">Score: <strong style="color:#10b981;">${qData.score.toFixed(2)}</strong></div>`;
          for (const [idx, prob] of Object.entries(qData.probabilities || {})) {
            const label = legend[idx] || ('Level ' + idx);
            const pct = Math.round(prob * 100);
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
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "device": "GPU"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/evaluate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            state = data.get("state", "")
            questions = data.get("questions", {})

            # Execute forward pass on Iris Xe GPU
            t0 = time.perf_counter()
            result = engine.system_one(state, questions)
            latency_ms = (time.perf_counter() - t0) * 1000

            resp_payload = {
                "latency_ms": latency_ms,
                "result": result
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
