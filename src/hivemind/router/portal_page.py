# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The interactive landing page and AI playground served at ``/`` and ``/playground``. Pure function, no I/O.

The page provides:
- Live HiveMind OpenRouter status and market architecture overview.
- An interactive in-browser AI chat completion playground (with streaming tokens/sec and latency metrics).
- Account balance and role inspector (querying ``/v1/hivemind/me``).
- Interactive capability tiers and VRAM matrix (derived from ``catalog.py``).
- 1-click editor and client snippets (Python OpenAI SDK, Continue, Cline/Roo, Aider, OpenCode, cURL).
- Quickstart guide for GPU producers to share idle compute with zero-lag game preemption (< 1.5s).
- Direct links to ``/signup`` (Google Sign-In), HiveMind@home Portal, and GitHub repository.

All dynamic content is written safely with ``textContent``, avoiding script injection. The content security policy
strictly enforces nonce-pinned scripts and styles.
"""

from __future__ import annotations

from hivemind.router.catalog import TIERS
from hivemind.router.google_auth import CLIENT_ID_RE
from hivemind.router.life_as_a_bee import render_life_as_a_bee_section


def portal_content_security_policy(nonce: str) -> str:
    """Strict Content-Security-Policy for the OpenRouter portal and playground."""
    return "; ".join([
        "default-src 'none'",
        f"script-src 'nonce-{nonce}'",
        f"style-src 'nonce-{nonce}'",
        "connect-src 'self'",
        "img-src 'self' data:",
        "font-src 'self' data:",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ])


_CSS = """
:root {
  --bg: #070b16;
  --bg-card: #0d1326;
  --bg-input: #121933;
  --border: #1e293b;
  --border-focus: #00eefc;
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --cyan: #00eefc;
  --amber: #fbbf24;
  --green: #34d399;
  --purple: #c084fc;
  --red: #f87171;
  --font-mono: ui-monospace, 'JetBrains Mono', 'Fira Code', monospace;
  --font-sans: system-ui, -apple-system, 'Space Grotesk', sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  line-height: 1.5;
  padding: 1.5rem 1rem 4rem;
}
.container {
  max-width: 68rem;
  margin: 0 auto;
}
header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.brand {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.brand h1 {
  font-size: 1.4rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, var(--cyan), var(--amber));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.badge-live {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  padding: 0.2rem 0.6rem;
  background: rgba(52, 211, 153, 0.12);
  color: var(--green);
  border: 1px solid rgba(52, 211, 153, 0.3);
  border-radius: 9999px;
  text-transform: uppercase;
}
.badge-live::before {
  content: "";
  display: inline-block;
  width: 6px;
  height: 6px;
  background: var(--green);
  border-radius: 50%;
  box-shadow: 0 0 6px var(--green);
}
nav {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}
.nav-link {
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: var(--text-dim);
  text-decoration: none;
  padding: 0.35rem 0.75rem;
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  transition: all 0.15s ease;
}
.nav-link:hover, .nav-link.active {
  color: var(--cyan);
  border-color: var(--cyan);
  background: rgba(0, 238, 252, 0.05);
}
.hero {
  margin-bottom: 2.5rem;
}
.hero h2 {
  font-size: 2rem;
  font-weight: 800;
  line-height: 1.2;
  margin-bottom: 0.75rem;
  color: #fff;
}
.hero p {
  color: var(--text-dim);
  font-size: 1.05rem;
  max-width: 44rem;
  margin-bottom: 1.5rem;
}
.cta-group {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-family: var(--font-mono);
  font-size: 0.88rem;
  font-weight: 600;
  padding: 0.6rem 1.2rem;
  border-radius: 0.375rem;
  cursor: pointer;
  text-decoration: none;
  border: 1px solid transparent;
  transition: all 0.15s ease;
}
.btn-primary {
  background: var(--cyan);
  color: #040711;
}
.btn-primary:hover {
  background: #22f3ff;
  box-shadow: 0 0 14px rgba(0, 238, 252, 0.4);
}
.btn-amber {
  background: var(--amber);
  color: #040711;
}
.btn-amber:hover {
  background: #fcd34d;
  box-shadow: 0 0 14px rgba(251, 191, 36, 0.4);
}
.btn-outline {
  border-color: var(--border);
  background: var(--bg-card);
  color: var(--text);
}
.btn-outline:hover {
  border-color: var(--text-dim);
  background: var(--bg-input);
}
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-bottom: 2.5rem;
}
@media (max-width: 840px) {
  .grid-2 { grid-template-columns: 1fr; }
}
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 0.6rem;
  padding: 1.25rem;
  position: relative;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.75rem;
}
.card-title {
  font-size: 1.1rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
label {
  display: block;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text-dim);
  margin-bottom: 0.35rem;
}
input[type="text"], input[type="password"], select, textarea {
  width: 100%;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 0.85rem;
  padding: 0.55rem 0.75rem;
  margin-bottom: 0.85rem;
  outline: none;
}
input:focus, select:focus, textarea:focus {
  border-color: var(--cyan);
}
.pill-row {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 0.85rem;
}
.pill {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  padding: 0.25rem 0.55rem;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border);
  border-radius: 0.25rem;
  cursor: pointer;
  color: var(--text-dim);
}
.pill:hover {
  color: var(--cyan);
  border-color: var(--cyan);
}
.terminal {
  background: #040711;
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 0.85rem;
  font-family: var(--font-mono);
  font-size: 0.82rem;
  min-height: 180px;
  max-height: 380px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
  color: #38bdf8;
}
.stats-bar {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.75rem;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--text-dim);
  margin-top: 0.65rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border);
}
.stat-item span {
  color: var(--amber);
  font-weight: 700;
}
.tiers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2.5rem;
}
.tier-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.tier-card:hover {
  border-color: var(--cyan);
}
.tier-badge {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--amber);
  margin-bottom: 0.35rem;
}
.tier-name {
  font-size: 0.95rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  color: #fff;
}
.tier-vram {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--cyan);
  margin-bottom: 0.5rem;
}
.tier-desc {
  font-size: 0.78rem;
  color: var(--text-dim);
  line-height: 1.4;
  margin-bottom: 0.75rem;
}
.tier-tag {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  padding: 0.2rem 0.4rem;
  background: var(--bg-input);
  border-radius: 0.25rem;
  color: var(--green);
}
.code-box {
  background: #040711;
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 1rem;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  overflow-x: auto;
  position: relative;
  margin-bottom: 1rem;
}
.tab-row {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
}
.tab-btn {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  padding: 0.3rem 0.6rem;
  background: transparent;
  border: 1px solid transparent;
  color: var(--text-dim);
  border-radius: 0.25rem;
  cursor: pointer;
}
.tab-btn.active {
  color: var(--cyan);
  background: var(--bg-input);
  border-color: var(--border);
}
footer {
  border-top: 1px solid var(--border);
  padding-top: 2rem;
  margin-top: 3rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  font-size: 0.82rem;
  color: var(--text-dim);
}
.footer-links {
  display: flex;
  gap: 1.25rem;
}
.footer-links a {
  color: var(--text-dim);
  text-decoration: none;
}
.footer-links a:hover {
  color: var(--cyan);
}
"""

_JS = """
const $ = (id) => document.getElementById(id);

// Load saved key from localStorage if present
const savedKey = localStorage.getItem('hivemind_api_key');
if (savedKey) {
  $('apiKey').value = savedKey;
}

// Save key changes
$('apiKey').addEventListener('input', (e) => {
  localStorage.setItem('hivemind_api_key', e.target.value.trim());
});

// Quick prompts
const PROMPTS = {
  python: "Write a high-performance Python decorator with LRU caching, thread safety, and execution duration metrics in microseconds.",
  pytorch: "Write a PyTorch autograd Function that computes the exact Voronoi cell projections on a Conway-Sloane A2 hexagonal lattice.",
  astro: "Simulate transit timing variations (TTV) of an exoplanetary system under 3-body gravitational resonance.",
  sundials: "Rescue a stiff Navier-Stokes / Gross-Pitaevskii ODE-PDE system in Rust using Rusty-SUNDIALS CVODE BDF with adaptive step control.",
  francais: "Bonjour ! Explique comment fonctionne La Ruche HiveMind@home pour partager son GPU citoyen et utiliser l'API dans VS Code.",
  reciprocity: "Explain how HiveMind's non-monetary compute reciprocity model works for gamers and AI developers.",
  crowdgpt: "Explain how a 1B generalist dense trunk (CrowdGPT Vxtzq/Crowd-v1 trained on 7B FineWeb tokens) and 18 MB domain QLoRA adapters can share a Hub-and-Spoke DiLoCo + Multi-Krum volunteer GPU swarm.",
  lean4: "Formalize in machine-checked Lean 4 with zero sorry the theorem that a discrete sequence of energies strictly bounded below by 0 and satisfying E_{t+1} <= E_t - delta (delta > 0) must terminate in finite steps.",
  rustdma: "Implement a cache-line aligned (#[repr(align(64))]) lock-free SPSC ring buffer in Rust with AtomicUsize acquire/release semantics and zero heap allocations on enqueue/dequeue.",
  cvepatch: "Analyze this asynchronous token-bucket rate limiter race condition where concurrent requests bypass limits (CVE pattern), and provide the atomic CAS remediation diff.",
  jwst: "In 2 sentences, explain how transmission spectroscopy at 1.4 µm (H2O) and 4.3 µm (CO2) from JWST NIRSpec constrains exoplanetary atmospheric metallicity [M/H] and C/O ratio.",
  crispr: "Evaluate guide RNA on-target efficiency and CFD off-target cleavage risk for CRISPR Cas9 targeting human HBB exon 1, citing PubMed clinical trial benchmarks.",
  triton: "Write an autotuned Triton kernel for fused Scaled Dot-Product Attention with online softmax normalization (FlashAttention-2 style) on FP16 tensors."
};

document.querySelectorAll('.prompt-pill').forEach(btn => {
  btn.addEventListener('click', () => {
    const key = btn.getAttribute('data-prompt');
    if (PROMPTS[key]) {
      $('promptInput').value = PROMPTS[key];
    }
  });
});

// Chat completion playground
let abortController = null;

async function sendChat() {
  const apiKey = $('apiKey').value.trim();
  const prompt = $('promptInput').value.trim();
  const model = $('modelSelect').value;
  const maxTokens = parseInt($('maxTokens').value, 10) || 512;
  const temperature = parseFloat($('temperature').value) || 0.7;
  const stream = $('streamCheck').checked;

  if (!apiKey) {
    $('outputTerminal').textContent = "Error: Please enter a valid HiveMind API Key (hm-...) or click 'Sign in with Google' to create one for free.";
    return;
  }
  if (!prompt) {
    $('outputTerminal').textContent = "Error: Please enter a prompt message.";
    return;
  }

  $('sendBtn').disabled = true;
  $('sendBtn').textContent = "Streaming...";
  $('outputTerminal').textContent = "Connecting to HiveMind OpenRouter...";
  $('statLatency').textContent = "...";
  $('statSpeed').textContent = "...";
  $('statTokens').textContent = "...";

  const startTime = performance.now();
  let firstByteTime = null;
  let receivedText = "";
  let tokenCount = 0;

  abortController = new AbortController();

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey,
        'X-Title': 'HiveMind-Playground',
      },
      body: JSON.stringify({
        model: model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: maxTokens,
        temperature: temperature,
        stream: stream,
      }),
      signal: abortController.signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      const msg = (err.error && err.error.message) || ('HTTP ' + response.status + ' Error');
      $('outputTerminal').textContent = 'Error: ' + msg;
      $('sendBtn').disabled = false;
      $('sendBtn').textContent = "Send Request";
      return;
    }

    if (stream && response.body) {
      $('outputTerminal').textContent = "";
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!firstByteTime) {
          firstByteTime = performance.now();
          $('statLatency').textContent = Math.round(firstByteTime - startTime) + ' ms';
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue;
          if (trimmed === 'data: [DONE]') continue;
          if (trimmed.startsWith('data: ')) {
            try {
              const data = JSON.parse(trimmed.slice(6));
              const delta = data.choices && data.choices[0] && data.choices[0].delta && data.choices[0].delta.content;
              if (delta) {
                receivedText += delta;
                tokenCount += 1;
                $('outputTerminal').textContent = receivedText;
                $('outputTerminal').scrollTop = $('outputTerminal').scrollHeight;
                const elapsedS = (performance.now() - (firstByteTime || startTime)) / 1000;
                if (elapsedS > 0) {
                  $('statSpeed').textContent = (tokenCount / elapsedS).toFixed(1) + ' tok/s';
                }
              }
            } catch (e) {}
          }
        }
      }
    } else {
      const data = await response.json();
      firstByteTime = performance.now();
      $('statLatency').textContent = Math.round(firstByteTime - startTime) + ' ms';
      const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
      receivedText = content || JSON.stringify(data, null, 2);
      $('outputTerminal').textContent = receivedText;
      if (data.usage && data.usage.completion_tokens) {
        tokenCount = data.usage.completion_tokens;
      }
    }

    const totalDurationS = (performance.now() - (firstByteTime || startTime)) / 1000;
    $('statTokens').textContent = tokenCount || Math.round(receivedText.length / 4);
    if (totalDurationS > 0 && tokenCount > 0) {
      $('statSpeed').textContent = (tokenCount / totalDurationS).toFixed(1) + ' tok/s';
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      $('outputTerminal').textContent = 'Network error: Could not reach gateway.';
    }
  } finally {
    $('sendBtn').disabled = false;
    $('sendBtn').textContent = "Send Request";
  }
}

$('sendBtn').addEventListener('click', sendChat);
$('clearBtn').addEventListener('click', () => {
  $('outputTerminal').textContent = "Ready.";
  $('statLatency').textContent = "0 ms";
  $('statSpeed').textContent = "0 tok/s";
  $('statTokens').textContent = "0";
});

// Account inspector
$('checkAccountBtn').addEventListener('click', async () => {
  const apiKey = $('apiKey').value.trim();
  if (!apiKey) {
    $('accountResult').textContent = "Enter your API key first.";
    return;
  }
  $('accountResult').textContent = "Checking account...";
  try {
    const res = await fetch('/v1/hivemind/me', {
      headers: { 'Authorization': 'Bearer ' + apiKey }
    });
    if (!res.ok) {
      $('accountResult').textContent = "Invalid API key or account not found.";
      return;
    }
    const data = await res.json();
    $('accountResult').textContent = 'Account: ' + data.account_id + ' | Role: ' + data.role + ' | Balance: ' + data.balance.toFixed(4) + ' HMT';
  } catch (err) {
    $('accountResult').textContent = "Failed to fetch account status.";
  }
});

// Snippet tabs
const SNIPPETS = {
  python: `from openai import OpenAI

client = OpenAI(
    base_url="${window.location.origin}/v1",
    api_key="HIVEMIND_API_KEY", # or os.getenv("HIVEMIND_API_KEY")
)

response = client.chat.completions.create(
    model="hivemind/coder",
    messages=[{"role": "user", "content": "Write a fast Python script"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`,
  opencode: `// opencode.json (project root or ~/.config/opencode/opencode.json)
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hivemind": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "HiveMind OpenRouter",
      "options": { "baseURL": "${window.location.origin}/v1", "apiKey": "YOUR_HIVEMIND_API_KEY" },
      "models": {
        "hivemind/coder": { "name": "HiveMind coder", "limit": { "context": 8192, "output": 700 } }
      }
    }
  }
}
// Keep "limit": without it a small balance is refused. 7B models often write tool calls as plain text:
// use them for chat-style help, or turn tools off for this provider.`,
  continue: `# Continue (VS Code / JetBrains): config.yaml
models:
  - name: HiveMind coder
    provider: openai
    model: hivemind/coder
    apiBase: ${window.location.origin}/v1
    apiKey: YOUR_HIVEMIND_API_KEY
    roles: [chat, edit, apply]
    defaultCompletionOptions:
      maxTokens: 700
# Not for autocomplete: there is no inline-completion endpoint yet.`,
  cline: `// In Cline / Roo-Code Provider Settings:
// Provider: OpenAI Compatible
// Base URL: ${window.location.origin}/v1
// API Key:  YOUR_HIVEMIND_API_KEY
// Model ID: hivemind/coder
// Max output tokens: 700 (if offered). Cline relies on tool calls, which 7B models often write as plain text.`,
  aider: `aider --openai-api-base ${window.location.origin}/v1 \\
      --openai-api-key YOUR_HIVEMIND_API_KEY \\
      --model openai/hivemind/coder`,
  zed: `// Zed settings.json (enter the key in Zed's provider settings)
{
  "language_models": {
    "openai_compatible": {
      "HiveMind": {
        "api_url": "${window.location.origin}/v1",
        "available_models": [
          { "name": "hivemind/coder", "display_name": "HiveMind coder", "max_tokens": 32768, "max_output_tokens": 700 }
        ]
      }
    }
  }
}`,
  jetbrains: `PyCharm / IntelliJ: install the Continue plugin and use the "VS Code Continue" tab, or add an
OpenAI-compatible provider in Settings > Tools > AI Assistant > Models:
  URL   : ${window.location.origin}/v1
  Key   : YOUR_HIVEMIND_API_KEY
  Model : hivemind/coder`,
  env: `export OPENAI_BASE_URL=${window.location.origin}/v1
export OPENAI_API_KEY=YOUR_HIVEMIND_API_KEY
# then any tool that reads the standard OpenAI variables works, with model hivemind/coder`,
  curl: `curl -X POST "${window.location.origin}/v1/chat/completions" \\
  -H "Authorization: Bearer YOUR_HIVEMIND_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "hivemind/coder", "messages": [{"role": "user", "content": "Hello!"}]}'`,
  antigravity: `# Google Antigravity SDK (pip install google-antigravity) + Unity/Steam Scientific Game Co-Pilot
import asyncio
from google.antigravity import Agent, LocalOpenAIAgentConfig

async def main():
    config = LocalOpenAIAgentConfig(
        model="hivemind/bee-stars",
        base_url="${window.location.origin}/v1",
    ).lightweight()
    async with Agent(config=config) as agent:
        reply = await agent.chat("Optimize our Symplectic Velocity-Verlet N-Body orbital step for KEPLER-HM-442b.")
        print(await reply.text())

asyncio.run(main())`
};

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.getAttribute('data-tab');
    if (SNIPPETS[tab]) {
      $('snippetCode').textContent = SNIPPETS[tab];
    }
  });
});
$('snippetCode').textContent = SNIPPETS['python'];

$('copySnippetBtn').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('snippetCode').textContent);
    $('copySnippetBtn').textContent = "Copied!";
    setTimeout(() => { $('copySnippetBtn').textContent = "Copy Snippet"; }, 2000);
  } catch (err) {
    $('copySnippetBtn').textContent = "Select & copy manually";
  }
});

if ($('campaignPollBtn')) {
  $('campaignPollBtn').addEventListener('click', async () => {
    const apiKey = $('apiKey').value.trim();
    if (!apiKey) {
      $('campaignPollStatus').textContent = "Enter your hm-... API key in the Playground first.";
      return;
    }
    const subjectId = $('campaignSubjectSelect').value;
    const kind = $('campaignRoleSelect').value;
    const vram = parseFloat($('campaignVramInput').value) || 0.0;
    $('campaignPollStatus').textContent = "Registering on AMCP/Kafka...";
    try {
      const res = await fetch('/v1/hivemind/campaigns/poll', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + apiKey,
        },
        body: JSON.stringify({
          subject_id: subjectId,
          participant_kind: kind,
          pledged_vram_gb: vram,
          availability_slot: 'eu_evening_cet',
        }),
      });
      if (!res.ok) {
        $('campaignPollStatus').textContent = "Registration failed (check API key).";
        return;
      }
      const data = await res.json();
      $('campaignPollStatus').textContent =
        "Registered! Leading: " + data.poll_summary.leading_subject_id +
        " (" + data.poll_summary.total_registrations + " votes, " +
        data.poll_summary.total_pledged_vram_gb + " GB VRAM pledged)";
    } catch (e) {
      $('campaignPollStatus').textContent = "Network error.";
    }
  });
}

// Life as a Bee category filter buttons
document.querySelectorAll('.bee-filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const filter = btn.getAttribute('data-bee-filter');
    document.querySelectorAll('.bee-filter-btn').forEach(b => {
      b.style.color = '';
      b.style.borderColor = '';
    });
    btn.style.color = 'var(--cyan)';
    btn.style.borderColor = 'var(--cyan)';
    document.querySelectorAll('.bee-scenario-card').forEach(card => {
      const cat = card.getAttribute('data-bee-category');
      if (filter === 'all' || cat === filter || (filter === 'dense_coding' && cat === 'smart_router')) {
        card.style.display = 'flex';
      } else {
        card.style.display = 'none';
      }
    });
  });
});

// Load Bee Scenario directly into the Playground
document.querySelectorAll('.load-bee-scenario-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const model = btn.getAttribute('data-bee-model');
    const prompt = btn.getAttribute('data-bee-prompt');
    const temp = btn.getAttribute('data-bee-temp');
    const tokens = btn.getAttribute('data-bee-tokens');
    if (model && $('modelSelect')) {
      $('modelSelect').value = model;
    }
    if (prompt && $('promptInput')) {
      $('promptInput').value = prompt;
    }
    if (temp && $('temperature') && $('tempVal')) {
      $('temperature').value = temp;
      $('tempVal').textContent = temp;
    }
    if (tokens && $('maxTokens') && $('maxTokensVal')) {
      $('maxTokens').value = tokens;
      $('maxTokensVal').textContent = tokens;
    }
    const pg = $('playground');
    if (pg && pg.scrollIntoView) {
      pg.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Free Steam Game Registration, Login & Idle-GPU BOINC Contribution Console
if ($('steamRegisterLoginBtn')) {
  $('steamRegisterLoginBtn').addEventListener('click', async () => {
    const handle = ($('steamHandleInput').value || 'Cmdr_RTX2070_Steam').trim();
    const steamId = ($('steamIdInput').value || '76561198031337001').trim();
    const gpuModel = ($('steamGpuInput').value || 'NVIDIA GeForce RTX 2070').trim();
    $('steamPortalStatus').textContent = 'Registering & authenticating Free Steam Game account (AppID 3133700)...';
    try {
      const res = await fetch('/v1/hivemind/steam/register-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          handle: handle,
          steam_id64: steamId,
          app_id: 3133700,
          gpu_model: gpuModel,
          role: 'prosumer',
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        $('steamPortalStatus').textContent = 'Error: ' + ((data.error && data.error.message) || 'Registration failed');
        return;
      }
      if (data.api_key && $('apiKey')) {
        $('apiKey').value = data.api_key;
        localStorage.setItem('hivemind_api_key', data.api_key);
      }
      localStorage.setItem('hivemind_steam_handle', data.handle || handle);
      $('steamPortalStatus').textContent =
        'AUTHENTICATED [' + (data.created ? 'NEW REGISTRATION' : 'EXISTING ACCOUNT LOGIN') + '] • Commander: ' +
        data.handle + ' • SteamID64: ' + data.steam_id64 + ' • AppID: ' + data.app_id +
        ' (Free Steam Edition) • Account: ' + data.account_id + ' • Balance: ' + Number(data.balance_hmt || 0).toFixed(2) +
        ' HMT • Ticket: ' + data.steam_session_ticket;
      if ($('accountResult')) {
        $('accountResult').textContent =
          'Account: ' + data.account_id + ' (' + data.handle + ') | Role: ' + data.role + ' | Balance: ' + Number(data.balance_hmt || 0).toFixed(4) + ' HMT';
      }
    } catch (err) {
      $('steamPortalStatus').textContent = 'Network error during Steam registration/login.';
    }
  });
}

if ($('steamContributeBtn')) {
  $('steamContributeBtn').addEventListener('click', async () => {
    const apiKey = ($('apiKey') && $('apiKey').value.trim()) || localStorage.getItem('hivemind_api_key') || '';
    const handle = ($('steamHandleInput') && $('steamHandleInput').value.trim()) || 'Cmdr_RTX2070_Steam';
    const target = ($('steamTargetSelect') && $('steamTargetSelect').value) || 'K2-18 b';
    $('steamPortalStatus').textContent = 'Executing idle-GPU CUDA PTX BOINC workunit on ' + target + ' & crediting HMT...';
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey) headers['Authorization'] = 'Bearer ' + apiKey;
      const res = await fetch('/v1/hivemind/steam/contribute-boinc', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
          handle: handle,
          target_planet: target,
          app_id: 3133700,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        $('steamPortalStatus').textContent = 'Error: ' + ((data.error && data.error.message) || 'BOINC contribution failed');
        return;
      }
      $('steamPortalStatus').textContent =
        'BOINC CONTRIBUTION VERIFIED • Target: ' + data.target_planet +
        ' • Depth: ' + data.recovered_transit_depth_ppm + ' ppm • Kernel: ' + data.kernel_exec_ms + ' ms (' + data.backend + ')' +
        ' • Awarded: +' + data.hmt_awarded + ' HMT & +' + data.cobblestones_awarded + ' Cobblestones' +
        ' • New Balance: ' + Number(data.balance_hmt || 0).toFixed(2) + ' HMT • SHA256: ' + String(data.receipt_sha256 || '').slice(0, 16) + '...';
      if ($('accountResult') && data.account_id) {
        $('accountResult').textContent =
          'Account: ' + data.account_id + ' (' + data.handle + ') | Role: prosumer | Balance: ' + Number(data.balance_hmt || 0).toFixed(4) + ' HMT';
      }
    } catch (err) {
      $('steamPortalStatus').textContent = 'Network error during BOINC GPU contribution.';
    }
  });
}
"""


def render_portal_page(client_id: str | None, nonce: str) -> str:
    """Render the full OpenRouter Portal and Playground HTML page."""
    life_as_a_bee_html = render_life_as_a_bee_section()
    tier_cards = []
    for tier in TIERS:
        tier_cards.append(f"""
        <div class="tier-card">
          <div>
            <div class="tier-badge">{"⚡ MIXTURE-OF-EXPERTS" if tier.moe else "⚡ DENSE"}</div>
            <div class="tier-name">{tier.title}</div>
            <div class="tier-vram">Min VRAM: {tier.min_vram_gb:.1f} GB &bull; {tier.suggested_cost_per_1k:.1f} HMT/1k</div>
            <div class="tier-desc">{tier.good_for}</div>
          </div>
          <div>
            <span class="tier-tag">{tier.ollama_tag}</span>
          </div>
        </div>
        """)
    tier_cards_html = "\n".join(tier_cards)

    signup_cta = """<a href="/signup" class="btn btn-amber">&#x1F41D; Sign in with Google (Free Key)</a>"""
    if client_id is None:
        signup_cta = """<a href="/signup" class="btn btn-amber">&#x1F41D; Join &amp; Register (API Key)</a>"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HiveMind OpenRouter &bull; Decentralized AI Inference for Coders &amp; Gamers</title>
  <meta name="description" content="Decentralized OpenAI-compatible inference router. Non-monetary compute reciprocity: share idle consumer GPU compute, earn points, and query frontier coding and MoE models.">
  <style nonce="{nonce}">{_CSS}</style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <span style="font-size:1.8rem;">&#x1F41D;</span>
        <div>
          <h1>HiveMind OpenRouter</h1>
          <div style="font-size:0.75rem;color:var(--text-dim);">Decentralized AI Inference Gateway &bull; v2.10.1</div>
        </div>
        <span class="badge-live">Gateway Live</span>
      </div>
      <nav>
        <a href="#playground" class="nav-link active">Playground</a>
        <a href="#life-as-a-bee" class="nav-link">&#x1F41D; Life as a Bee</a>
        <a href="#campaigns" class="nav-link">Joint Campaigns</a>
        <a href="#night-telemetry" class="nav-link">Night Progress</a>
        <a href="#tiers" class="nav-link">Model Tiers</a>
        <a href="#integrations" class="nav-link">Editor Config</a>
        <a href="#steam-free-game" class="nav-link">&#x1F3AE; Free Steam Game</a>
        <a href="#producers" class="nav-link">Share GPU</a>
        <a href="/signup" class="nav-link">&#x1F511; Sign In</a>
      </nav>
    </header>

    <section class="hero">
      <h2>Community AI Inference.<br><span style="color:var(--cyan);">Zero Cost. Powered by Consumer GPUs.</span></h2>
      <p>
        HiveMind OpenRouter gives developers an OpenAI-compatible API powered by a global swarm of idle gamer GPUs.
        Operates strictly under the <strong>HiveMind Reflexion Points (HMT) Non-Monetary Compute Reciprocity Charter</strong>:
        share compute when your GPU is idle, earn HMT points, and spend them in your editor.
      </p>
      <div class="cta-group">
        <a href="#playground" class="btn btn-primary">&#x26A1; Try In-Browser Playground</a>
        <a href="#steam-free-game" class="btn btn-outline" style="border-color:var(--green);color:var(--green);">&#x1F3AE; Free Steam Game (AppID 3133700)</a>
        <a href="#life-as-a-bee" class="btn btn-outline" style="border-color:var(--amber);color:var(--amber);">&#x1F41D; Life as a Bee (15 Scenarios)</a>
        {signup_cta}
        <a href="https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/" target="_blank" rel="noopener" class="btn btn-outline">&#x1F36F; HiveMind@home Portal</a>
      </div>
    </section>

    <!-- Interactive AI Playground -->
    <div id="playground" class="grid-2">
      <!-- Input Controls -->
      <section class="card">
        <div class="card-header">
          <div class="card-title">&#x1F680; Live AI Completion Console</div>
          <span style="font-size:0.75rem;color:var(--cyan);font-family:var(--font-mono);">OpenAI /v1/chat/completions</span>
        </div>

        <label for="apiKey">HMT API Key (Bearer hm-...)</label>
        <input id="apiKey" type="password" placeholder="Paste your hm-... key or sign in at /signup" />

        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.85rem;">
          <button id="checkAccountBtn" type="button" class="pill" style="color:var(--amber);">&#x1F50D; Check Account &amp; Balance</button>
          <span id="accountResult" style="font-size:0.75rem;font-family:var(--font-mono);color:var(--text-dim);"></span>
        </div>

        <label for="modelSelect">Capability Tier (Model)</label>
        <select id="modelSelect">
          <option value="hivemind/auto">hivemind/auto (Smart Intent &amp; Market Liquidity Routing)</option>
          <option value="hivemind/coder">hivemind/coder (7B &bull; Everyday Python &amp; Game Code)</option>
          <option value="hivemind/qwen35-core">hivemind/qwen35-core (8B &bull; Qwen 3.5 Hybrid Reasoning &amp; Polyglot Core)</option>
          <option value="hivemind/gemma-core">hivemind/gemma-core (4B &bull; Google Gemma 3 Multilingual &amp; Multimodal Core)</option>
          <option value="hivemind/mistral-core">hivemind/mistral-core (7B &bull; Mistral v0.3 European Bilingual FR/EN Core)</option>
          <option value="hivemind/moe-qwen35-bee">hivemind/moe-qwen35-bee (20B MoE / 3.0B Active &bull; Polyglot-Singularity MoE Bee)</option>
          <option value="hivemind/bee-ruche-moe">hivemind/bee-ruche-moe (20B MoE / 2.4B Active &bull; La Ruche 8-Expert Sparse MoE Bee)</option>
          <option value="hivemind/bee-rusty-sundials">hivemind/bee-rusty-sundials (7B &bull; Rusty-SUNDIALS Numeric ODE/PDE Rescue)</option>
          <option value="hivemind/bee-stars">hivemind/bee-stars (7B &bull; Astrophysics &amp; Symplectic N-Body QLoRA Bee)</option>
          <option value="hivemind/bee-lean-prover">hivemind/bee-lean-prover (8B &bull; Goedel-Prover-V2 &amp; DeepSeek-Prover-V2 Lean 4 Bee)</option>
          <option value="hivemind/crowd-v1">hivemind/crowd-v1 (1B &bull; CrowdGPT Vxtzq/Crowd-v1 Generalist Trunk &bull; Pre-Wired Partner)</option>
          <option value="hivemind/coder-lite">hivemind/coder-lite (3B &bull; Ultra-Fast Edits)</option>
          <option value="hivemind/coder-plus">hivemind/coder-plus (14B &bull; Multi-File &amp; DeepSeek-R1 Reasoning)</option>
          <option value="hivemind/moe-general">hivemind/moe-general (20B MoE / 3.6B Active &bull; General Reasoning)</option>
          <option value="hivemind/moe-coder">hivemind/moe-coder (30B MoE / 3.3B Active &bull; Frontier Coding)</option>
          <optgroup label="👑 Super-MoE &amp; Macro-Clusters (20B / Top-2)">
            <option value="hivemind/moe-systems-formal">hivemind/moe-systems-formal (Cluster 00 &bull; Systèmes, Kernels &amp; Lean 4)</option>
            <option value="hivemind/moe-cyber-crypto">hivemind/moe-cyber-crypto (Cluster 01 &bull; Cybersécurité, CVE &amp; Circuits ZK)</option>
            <option value="hivemind/moe-frontier-science">hivemind/moe-frontier-science (Cluster 02 &bull; Astrophysique, JWST &amp; Réseau Quantique)</option>
            <option value="hivemind/moe-reasoning-bio">hivemind/moe-reasoning-bio (Cluster 03 &bull; Olympiades Math, Génomique CRISPR &amp; Essais Cliniques)</option>
          </optgroup>
          <optgroup label="🔬 Adaptateurs MoE Hugging Face (+277 TFLOPs)">
            <option value="hivemind/moe-maths-formal">hivemind/moe-maths-formal (Lean 4 Machine-Checked &bull; proofnet)</option>
            <option value="hivemind/moe-rust-kernel">hivemind/moe-rust-kernel (Rust DMA Kernel &bull; the-stack-v2)</option>
            <option value="hivemind/moe-python-perf">hivemind/moe-python-perf (Python SIMD Kernel &bull; CodeAlpaca)</option>
            <option value="hivemind/moe-feedback-debug">hivemind/moe-feedback-debug (Multi-Turn Debug &bull; CodeFeedback)</option>
            <option value="hivemind/moe-dist-p2p">hivemind/moe-dist-p2p (Consensus P2P &bull; santacoder)</option>
            <option value="hivemind/moe-cuda-opt">hivemind/moe-cuda-opt (Triton / Tensor Core &bull; LLM_Compression)</option>
            <option value="hivemind/moe-cyber-cti">hivemind/moe-cyber-cti (Threat Intel CTI-Bench &bull; cti-bench)</option>
            <option value="hivemind/moe-cyber-cve">hivemind/moe-cyber-cve (SWE-Bench CVE Remediation &bull; SWE-bench)</option>
            <option value="hivemind/moe-smart-contracts">hivemind/moe-smart-contracts (Solidity ZK Escrow &bull; DISL-Solidity)</option>
            <option value="hivemind/moe-crypto-zkp">hivemind/moe-crypto-zkp (ZK Circuits SNARK/STARK &bull; zk-bench)</option>
            <option value="hivemind/moe-reverse-eng">hivemind/moe-reverse-eng (Malware Triage &bull; malware-prompts)</option>
            <option value="hivemind/moe-astro-transits">hivemind/moe-astro-transits (Kepler Exoplanet Transits &bull; exoplanets)</option>
            <option value="hivemind/moe-astro-spectra">hivemind/moe-astro-spectra (Gaia DR3 Stellar Spectra &bull; sdss_dr12)</option>
            <option value="hivemind/moe-astro-atmosphere">hivemind/moe-astro-atmosphere (JWST Atmosphere Biosignatures &bull; jwst)</option>
            <option value="hivemind/moe-quantum-lattice">hivemind/moe-quantum-lattice (Quantum Lattice QCD &bull; multimodal_universe)</option>
            <option value="hivemind/moe-chem-docking">hivemind/moe-chem-docking (Molecular Docking &bull; zinc250k)</option>
            <option value="hivemind/moe-climate-earth">hivemind/moe-climate-earth (Earth Observation &bull; hls-foundation)</option>
            <option value="hivemind/moe-maths-olympiad">hivemind/moe-maths-olympiad (Olympiad Math GSM8K &bull; gsm8k)</option>
            <option value="hivemind/moe-bio-genomics">hivemind/moe-bio-genomics (CRISPR PubMedQA &bull; PubMedQA)</option>
            <option value="hivemind/moe-bio-clinical">hivemind/moe-bio-clinical (Clinical Trial Evidence &bull; guidance-biomedical)</option>
          </optgroup>
          <optgroup label="🐝 Exemplaires Bring Your Own Bee (BYOB)">
            <option value="hivemind/bee-quantum-simd">hivemind/bee-quantum-simd (Quantum SIMD &bull; AVX-512 Contractions)</option>
            <option value="hivemind/bee-reverse-cve">hivemind/bee-reverse-cve (Binary Disassembly &bull; SWE-bench CVE Patch)</option>
            <option value="hivemind/bee-genomics-crispr">hivemind/bee-genomics-crispr (CRISPR Cleavage &bull; Clinical Evidence)</option>
          </optgroup>
          <optgroup label="⚡ Modèles Ultra-Légers &amp; Edge (CPU / 8 Go RAM)">
            <option value="hivemind/qwen35-4b">hivemind/qwen35-4b (Qwen 3.5 4B &bull; Best Overall 8GB RAM / Tool-Calling)</option>
            <option value="hivemind/phi4-mini">hivemind/phi4-mini (Phi-4 Mini 3.8B &bull; Footprint ~2.5 GB RAM)</option>
            <option value="hivemind/gemma4-e4b">hivemind/gemma4-e4b (Gemma 4 E4B &bull; Multimodal Edge Text/Image/Audio)</option>
            <option value="hivemind/qwen35-sub2b">hivemind/qwen35-sub2b (Qwen 3.5 2B/0.8B &bull; Sub-2B Low-RAM &lt; 3 GB)</option>
          </optgroup>
        </select>

        <label>Quick Prompt Presets</label>
        <div class="pill-row">
          <button type="button" class="pill prompt-pill" data-prompt="python">Python LRU</button>
          <button type="button" class="pill prompt-pill" data-prompt="lean4">Lean 4 ProofNet</button>
          <button type="button" class="pill prompt-pill" data-prompt="rustdma">Rust DMA Buffer</button>
          <button type="button" class="pill prompt-pill" data-prompt="cvepatch">SWE-Bench CVE</button>
          <button type="button" class="pill prompt-pill" data-prompt="jwst">JWST Atmosphere</button>
          <button type="button" class="pill prompt-pill" data-prompt="crispr">CRISPR PubMedQA</button>
          <button type="button" class="pill prompt-pill" data-prompt="triton">Triton FlashAttn</button>
          <button type="button" class="pill prompt-pill" data-prompt="sundials">Rusty-SUNDIALS</button>
          <button type="button" class="pill prompt-pill" data-prompt="astro">Exoplanet TTV</button>
          <button type="button" class="pill prompt-pill" data-prompt="francais">Citoyen IA (FR)</button>
        </div>

        <label for="promptInput">Prompt Message</label>
        <textarea id="promptInput" rows="5" placeholder="Ask a coding question, request refactoring, or generate algorithm tests...">Write a high-performance Python decorator with LRU caching, thread safety, and execution duration metrics in microseconds.</textarea>

        <div style="display:flex;gap:1rem;align-items:center;margin-bottom:1rem;flex-wrap:wrap;">
          <div style="flex:1;min-width:120px;">
            <label for="maxTokens">Max Tokens: <span id="maxTokensVal">512</span></label>
            <input id="maxTokens" type="range" min="64" max="2048" step="64" value="512" oninput="document.getElementById('maxTokensVal').textContent=this.value" />
          </div>
          <div style="flex:1;min-width:120px;">
            <label for="temperature">Temperature: <span id="tempVal">0.7</span></label>
            <input id="temperature" type="range" min="0" max="1.5" step="0.1" value="0.7" oninput="document.getElementById('tempVal').textContent=this.value" />
          </div>
          <div style="display:flex;align-items:center;gap:0.4rem;padding-top:0.8rem;">
            <input id="streamCheck" type="checkbox" checked />
            <label for="streamCheck" style="margin:0;cursor:pointer;">SSE Stream</label>
          </div>
        </div>

        <div style="display:flex;gap:0.75rem;">
          <button id="sendBtn" type="button" class="btn btn-primary" style="flex:2;">&#x26A1; Send Request</button>
          <button id="clearBtn" type="button" class="btn btn-outline" style="flex:1;">Clear</button>
        </div>
      </section>

      <!-- Terminal Output -->
      <section class="card" style="display:flex;flex-direction:column;justify-content:space-between;">
        <div class="card-header">
          <div class="card-title">&#x1F4DF; Response Terminal</div>
          <span style="font-size:0.75rem;color:var(--green);font-family:var(--font-mono);">Streaming Output</span>
        </div>

        <div id="outputTerminal" class="terminal">Ready. Enter your API key and click 'Send Request' to test the HiveMind OpenRouter.</div>

        <div class="stats-bar">
          <div class="stat-item">First Byte: <span id="statLatency">0 ms</span></div>
          <div class="stat-item">Generation Speed: <span id="statSpeed">0 tok/s</span></div>
          <div class="stat-item">Tokens: <span id="statTokens">0</span></div>
          <div class="stat-item">Reciprocity: <span>HiveMind Reflexion Points (HMT)</span></div>
        </div>
      </section>
    </div>

    {life_as_a_bee_html}

    <!-- Joint Subject-by-Subject Improvement Campaigns, Community Poll & AMCP/Kafka Bus -->
    <section id="campaigns" class="card" style="margin-bottom:3rem;border-color:rgba(0,238,252,0.35);">
      <div class="card-header">
        <div class="card-title" style="color:var(--cyan);">&#x1F4C5; Joint Subject-by-Subject Improvement Campaigns, Poll &amp; AMCP/Kafka Bus</div>
        <a href="/v1/hivemind/campaigns/calendar.ics" class="btn btn-outline" style="padding:0.3rem 0.75rem;font-size:0.75rem;">&#x1F4C6; Download Joint Calendar (.ics)</a>
      </div>
      <p style="color:var(--text-dim);font-size:0.88rem;margin-bottom:1rem;line-height:1.5;">
        Humans, GPU Producers, AI Agents, and Specialized Bees unite on <strong>one flagship scientific or coding subject per sprint</strong>.
        Vote in the community poll below to steer nocturnal BOINC/QLoRA workunit weights and coordinate over our
        <strong>AMCP v1.5 + CloudEvents / Kafka Event Bus</strong> (<code>hivemind.amcp.campaigns.v1</code>, <code>hivemind.amcp.human_agent.v1</code>, <code>hivemind.amcp.bee_coord.v1</code>, <code>hivemind.amcp.bee_trading.v1</code>).
      </p>
      <div style="display:flex;gap:0.75rem;flex-wrap:wrap;align-items:flex-end;margin-bottom:1rem;">
        <div style="flex:2;min-width:220px;">
          <label for="campaignSubjectSelect">Vote / Register for Subject Sprint</label>
          <select id="campaignSubjectSelect" style="margin-bottom:0;">
            <option value="SUBJ-01-PYTHON-RUST-CODER">Subject 1 (Oct 28–30): Python &amp; Rust Systems Coding (hivemind/coder)</option>
            <option value="SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE">Subject 2 (Oct 31–Nov 2): Stiff ODE/PDE &amp; Navier-Stokes Fluid Rescue (bee-rusty-sundials)</option>
            <option value="SUBJ-03-ASTRO-SYMPLECTIC-STARS">Subject 3 (Nov 3–5): Exoplanet TTV &amp; Symplectic Astrophysics (bee-stars)</option>
            <option value="SUBJ-04-QUANTUM-BEC-GROSS-PITAEVSKII">Subject 4 (Nov 6–8): Quantum Fluids &amp; Gross-Pitaevskii Solitons (bee-rusty-sundials)</option>
            <option value="SUBJ-05-GENETICS-WRIGHT-FISHER">Subject 5 (Nov 9–11): Population Genetics &amp; Fisher-KPP Waves (bee-rusty-sundials)</option>
            <option value="SUBJ-06-LEAN4-FORMAL-PROOFS">Subject 6 (Nov 12–15): Formal Lean 4 Proofs &amp; Invariants (moe-coder)</option>
          </select>
        </div>
        <div style="flex:1;min-width:160px;">
          <label for="campaignRoleSelect">Participant Role</label>
          <select id="campaignRoleSelect" style="margin-bottom:0;">
            <option value="human_researcher">Human Researcher / Dev</option>
            <option value="gpu_producer">GPU Producer / Miner</option>
            <option value="ai_agent">Autonomous AI Agent</option>
            <option value="bee_specialist">Specialist Bee</option>
          </select>
        </div>
        <div style="width:130px;">
          <label for="campaignVramInput">Pledged VRAM (GB)</label>
          <input id="campaignVramInput" type="text" value="8" style="margin-bottom:0;" />
        </div>
        <button id="campaignPollBtn" type="button" class="btn btn-primary">&#x1F5F3;&#xFE0F; Register &amp; Vote</button>
      </div>
      <div id="campaignPollStatus" style="font-size:0.78rem;font-family:var(--font-mono);color:var(--amber);">
        API Endpoints: <code>GET /v1/hivemind/campaigns</code> &bull; <code>POST /v1/hivemind/campaigns/poll</code> &bull; <code>POST /v1/hivemind/bees/coordinate</code> &bull; <code>POST /v1/hivemind/bees/trade</code>
      </div>
    </section>

    <!-- Real-Life Night Progress & BOINC QLoRA Telemetry (25-26 Oct 2026) -->
    <section id="night-telemetry" class="card" style="margin-bottom:3rem;border-color:rgba(52,211,153,0.35);">
      <div class="card-header">
        <div class="card-title" style="color:var(--green);">&#x1F319; European Night Real-Life Telemetry &amp; BOINC QLoRA-MoE Progress (25&ndash;26 Oct 2026)</div>
        <span class="pill" style="border-color:var(--green);color:var(--green);">15/15 Live Probes PASS &bull; 0 HTTP 402/5xx</span>
      </div>
      <p style="color:var(--text-dim);font-size:0.88rem;margin-bottom:1rem;line-height:1.5;">
        Measured production telemetry from the European Night window (<code>2026-10-25 20:00Z &rarr; 2026-10-26 08:00Z</code>) across our three Cloud Run gateways, dual-GPU inference fleet (<strong>NVIDIA Tesla T4 16 GB + NVIDIA GeForce RTX 2070 8 GB</strong>), and the BOINC 7.24 chained QLoRA-MoE coordinator (<code>scripts/t4_training_report.py</code>).
      </p>
      <div class="tiers-grid" style="margin-bottom:1.25rem;">
        <div class="tier-card">
          <div>
            <div class="tier-badge">DUAL-GPU INFERENCE FLEET</div>
            <div class="tier-name">Tesla T4 (16 GB) + RTX 2070 (8 GB)</div>
            <div class="tier-vram">4 Active Producers &bull; 7 Live Models</div>
            <div class="tier-desc">Serving <code>hivemind/coder</code>, <code>coder-lite</code>, <code>bee-stars</code>, <code>coder-plus</code>, <code>moe-general</code>, <code>moe-coder</code>, and <code>auto</code> (with <code>hivemind/crowd-v1</code> pre-wired).</div>
          </div>
          <div><span class="tier-tag">100% Uptime Overnight</span></div>
        </div>
        <div class="tier-card">
          <div>
            <div class="tier-badge">OVERNIGHT GATEWAY TRAFFIC</div>
            <div class="tier-name">4,474 HTTP Reqs &bull; 133 Completions</div>
            <div class="tier-vram">Router p50: 198.1 ms &bull; p95: 799.6 ms</div>
            <div class="tier-desc">83 chat completions on <code>router</code>, 25 on <code>ruche</code> (p50 293.9 ms), 25 on <code>tribute</code> (p50 265.1 ms), and 81,424 cumulative nginx requests on the T4 node.</div>
          </div>
          <div><span class="tier-tag">0 HTTP 402 &bull; 0 HTTP 5xx</span></div>
        </div>
        <div class="tier-card">
          <div>
            <div class="tier-badge">BOINC CHAINED QLORA-MOE</div>
            <div class="tier-name">100 Assimilated WUs (98 hmq_*)</div>
            <div class="tier-vram">5,632.11 Cobblestones &bull; 0.0 Replay Drift</div>
            <div class="tier-desc">85 of 98 rounds completed inside 22:00&ndash;07:00 CET (mean 105 s/round, 3,409 MiB peak VRAM). Strict loss ratchet accepted 20 improvements and held 78 rounds at head <code>hmq_00092</code>.</div>
          </div>
          <div><span class="tier-tag">Head: hmq_00092 (SHA e91e0ec3)</span></div>
        </div>
        <div class="tier-card">
          <div>
            <div class="tier-badge">6-DOMAIN HELD-OUT NLL</div>
            <div class="tier-name">2.2705 &rarr; 2.1380 (-5.8%)</div>
            <div class="tier-vram">Math -15.1% &bull; Biomed -11.7% &bull; Gen -8.8%</div>
            <div class="tier-desc"><code>math</code>: 1.6440 &rarr; 1.3949 | <code>biomed</code>: 2.5017 &rarr; 2.2089 | <code>general</code>: 2.3230 &rarr; 2.1177 | <code>french</code>: 2.3789 &rarr; 2.3475 (-1.3%) | <code>news</code>: 2.1422 &rarr; 2.1246 (-0.8%) | <code>edu_web</code>: 2.6332 &rarr; 2.6342 (~0%).</div>
          </div>
          <div><span class="tier-tag">Plateau after ~30 rounds</span></div>
        </div>
      </div>
      <div style="font-size:0.78rem;color:var(--text-dim);font-family:var(--font-mono);border-top:1px solid var(--border);padding-top:0.75rem;line-height:1.5;">
        <strong>Scientific &amp; Methodological Caveats (<code>scripts/t4_training_report.py</code>):</strong> Baseline (<code>2.2705</code>) is the state before <code>hmq_00002</code> (first 6-domain round; <code>hmq_00001</code> trained on <code>general</code> only). Loss plateaued after ~30 rounds (std over last 20 rounds = <code>0.0055</code>; target-domain delta in rounds 69&ndash;98 = <code>+0.0006</code>). Held-out sets are fixed (8 sequences &times; 128 tokens/domain), head <code>hmq_00092</code> is best-of-97 on those sets, and the BOINC <code>granite-3.1-1b-a400m-base</code> LoRA chain is an offline training track not yet served by <code>/v1/chat/completions</code>. Governor recorded 130 total yields (latest mid-BOINC-task kill-on-yield: <code>1,135 ms</code> and <code>1,133 ms</code>).
      </div>
    </section>

    <!-- Capability Tiers Grid -->
    <section id="tiers" style="margin-bottom:3rem;">
      <h3 style="font-size:1.3rem;font-weight:700;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.5rem;">
        <span>&#x1F4D1; 5 Community Capability Tiers &amp; GPU Sizing (14 Live Models &amp; MoE Bees)</span>
      </h3>
      <p style="color:var(--text-dim);font-size:0.9rem;margin-bottom:1.25rem;">
        Capability tiers pool model demand and producer supply into deep, liquid markets.
        Gamers pick the tier that fits their VRAM; developers request the tier matching their reasoning needs.
      </p>
      <div class="tiers-grid">
        {tier_cards_html}
      </div>
    </section>

    <!-- Editor Integrations -->
    <section id="integrations" class="card" style="margin-bottom:3rem;">
      <div class="card-header">
        <div class="card-title">&#x1F9E9; 1-Click Editor &amp; SDK Configurations</div>
        <button id="copySnippetBtn" type="button" class="btn btn-outline" style="padding:0.3rem 0.75rem;font-size:0.75rem;">Copy Snippet</button>
      </div>

      <div class="tab-row">
        <button type="button" class="tab-btn active" data-tab="python">Python (OpenAI SDK)</button>
        <button type="button" class="tab-btn" data-tab="opencode">OpenCode</button>
        <button type="button" class="tab-btn" data-tab="continue">VS Code Continue</button>
        <button type="button" class="tab-btn" data-tab="cline">Cline / Roo-Code</button>
        <button type="button" class="tab-btn" data-tab="zed">Zed</button>
        <button type="button" class="tab-btn" data-tab="jetbrains">JetBrains</button>
        <button type="button" class="tab-btn" data-tab="aider">Aider Terminal</button>
        <button type="button" class="tab-btn" data-tab="env">Env variables</button>
        <button type="button" class="tab-btn" data-tab="curl">cURL</button>
        <button type="button" class="tab-btn" data-tab="antigravity">Google Antigravity SDK</button>
      </div>

      <pre id="snippetCode" class="code-box"></pre>
      <p style="font-size:0.78rem;color:var(--text-dim);margin-top:0.75rem;">
        Only the live models in <code>/v1/models</code> answer; the larger tiers have no producer yet. Tested live: OpenCode, Python, cURL.
        The other tabs are format-checked but not yet run against a real editor. Full guide:
        <a href="https://github.com/xaviercallens/HiveMind/blob/main/docs/HIVEMIND_CONNECT_IDE.md" rel="noopener noreferrer">docs/HIVEMIND_CONNECT_IDE.md</a>.
      </p>
    </section>

    <!-- Free Steam Game Registration, Login & Idle-GPU BOINC Contribution Console -->
    <section id="steam-free-game" class="card" style="border-color:rgba(52,211,153,0.45);margin-bottom:3rem;">
      <div class="card-header">
        <div class="card-title" style="color:var(--green);">&#x1F3AE; Free Steam Game Registered on Portal: USS Hivemind NCC-31337 &mdash; Exoplanet Hunter (AppID 3133700)</div>
        <span class="pill" style="border-color:var(--green);color:var(--green);">FREE ON STEAM (FreeOnDemand) &bull; &lt; 100 ms BOINC Preemption</span>
      </div>
      <p style="color:var(--text-dim);font-size:0.9rem;margin-bottom:1rem;line-height:1.5;">
        <strong>USS Hivemind NCC-31337 v5.1</strong> is registered on the HiveMind Portal as a <strong>100% Free-to-Play Steam Edition</strong> (<code>AppID 3133700</code>, <code>Depot 3133702</code>, non-monetary compute reciprocity charter).
        Register or sign in with your Commander Handle and SteamID64 below to link your Free Steam Game client, obtain your <code>hm-...</code> API key, and contribute idle-GPU CUDA PTX BOINC light-curve folding whenever you are not actively flying.
      </p>
      <div style="display:flex;gap:0.75rem;flex-wrap:wrap;align-items:flex-end;margin-bottom:0.85rem;">
        <div style="flex:1.2;min-width:190px;">
          <label for="steamHandleInput">Commander Handle</label>
          <input id="steamHandleInput" type="text" value="Cmdr_RTX2070_Steam" style="margin-bottom:0;" />
        </div>
        <div style="flex:1.1;min-width:180px;">
          <label for="steamIdInput">SteamID64 (17-digit)</label>
          <input id="steamIdInput" type="text" value="76561198031337001" style="margin-bottom:0;" />
        </div>
        <div style="flex:1.2;min-width:190px;">
          <label for="steamGpuInput">Local GPU Hardware</label>
          <input id="steamGpuInput" type="text" value="NVIDIA GeForce RTX 2070 (8 GB VRAM)" style="margin-bottom:0;" />
        </div>
        <div style="flex:1;min-width:160px;">
          <label for="steamTargetSelect">BOINC NASA Target</label>
          <select id="steamTargetSelect" style="margin-bottom:0;">
            <option value="K2-18 b">K2-18 b (2789 ppm Hycean)</option>
            <option value="LHS 1140 b">LHS 1140 b (5385 ppm Super-Earth)</option>
            <option value="TRAPPIST-1 e">TRAPPIST-1 e (4996 ppm HZ)</option>
            <option value="WASP-39 b">WASP-39 b (19549 ppm JWST)</option>
            <option value="TOI-700 d">TOI-700 d (653 ppm TESS)</option>
            <option value="HD 80606 b">HD 80606 b (10179 ppm Eccentric)</option>
            <option value="Kepler-11 f">Kepler-11 f (458 ppm Compact)</option>
            <option value="Kepler-186 f">Kepler-186 f (420 ppm HZ)</option>
          </select>
        </div>
      </div>
      <div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.85rem;">
        <button id="steamRegisterLoginBtn" type="button" class="btn btn-primary">&#x1F3AE; Register &amp; Login Free Steam Game Account</button>
        <button id="steamContributeBtn" type="button" class="btn btn-amber">&#x26A1; Contribute Idle-GPU BOINC Workunit (RTX CUDA PTX)</button>
        <a href="/join#play" class="btn btn-outline">&#x1F4D6; Steam &amp; BOINC Setup Guide (/join#play)</a>
      </div>
      <div id="steamPortalStatus" style="font-size:0.78rem;font-family:var(--font-mono);color:var(--green);background:#040711;border:1px solid var(--border);border-radius:0.375rem;padding:0.65rem;">
        Free Steam Game Registry: <code>AppID 3133700</code> (USS Hivemind: Exoplanet Hunter &mdash; Free BOINC Edition) &bull; Status: <strong>REGISTERED &amp; LIVE ON PORTAL</strong> &bull; Click &lsquo;Register &amp; Login Free Steam Game Account&rsquo; to authenticate.
      </div>
    </section>

    <!-- Producer Quickstart for Gamers -->
    <section id="producers" class="card" style="border-color:rgba(251,191,36,0.3);margin-bottom:3rem;">
      <div class="card-header">
        <div class="card-title" style="color:var(--amber);">&#x1F41D; Gamers &amp; GPU Sharers: Earn Points While AFK</div>
        <span class="pill" style="border-color:var(--amber);color:var(--amber);">&lt; 1.5s Game Preemption</span>
      </div>
      <p style="color:var(--text-dim);font-size:0.92rem;margin-bottom:1rem;">
        Got an RTX 3060, 3080, 4070, 4090, Radeon, or Apple Silicon? Run Ollama and our lightweight producer agent.
        The moment you launch Elden Ring, Steam, or Cyberpunk 2077, your card yields 100% compute in under 1.5 seconds.
      </p>
      <div class="code-box" style="margin-bottom:0.5rem;">
# 1. Start your local Ollama server (or vLLM / llama.cpp)
ollama run qwen2.5-coder:7b

# 2. Get your producer API key with Google or handle at /signup
export HIVEMIND_API_KEY=hm-...

# 3. Start sharing with the OpenRouter gateway
python scripts/hivemind_producer_agent.py --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app --preset coder
      </div>
      <div style="font-size:0.78rem;color:var(--text-dim);font-family:var(--font-mono);">
        &bull; Enforces zero-allocation memory caps &bull; Automatic heartbeat &bull; SSRF and privacy shield protection.
      </div>
    </section>

    <footer>
      <div>
        <strong>HiveMind OpenRouter</strong> &bull; Non-monetary compute reciprocity.
      </div>
      <div class="footer-links">
        <a href="https://github.com/xaviercallens/HiveMind" target="_blank" rel="noopener">GitHub</a>
        <a href="https://xaviercallens.github.io/HiveMindatHome.github.io/" target="_blank" rel="noopener">HiveMind@home</a>
        <a href="/signup">Sign in with Google</a>
        <a href="/v1/hivemind/catalog">Catalog JSON</a>
        <a href="/healthz">Healthz</a>
      </div>
    </footer>
  </div>

  <script nonce="{nonce}">{_JS}</script>
</body>
</html>"""
