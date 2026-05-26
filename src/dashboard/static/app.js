// Relay dashboard front-end. Polls /api/workers for telemetry and streams
// chat completions through /api/chat/completions.

const WORKER_POLL_INTERVAL_MS = 3000;

const state = {
  conversation: [],
  workers: [],
  selectedModel: null,
  abortController: null,
};

const el = {
  workers: document.getElementById("worker-list"),
  workerCount: document.getElementById("worker-count"),
  messages: document.getElementById("messages"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send-btn"),
  cancel: document.getElementById("cancel-btn"),
  clear: document.getElementById("clear-btn"),
  modelSelect: document.getElementById("model-select"),
  composerHint: document.getElementById("composer-hint"),
  detailBody: document.getElementById("detail-body"),
  detailCost: document.getElementById("detail-cost"),
  statWorkers: document.getElementById("stat-workers"),
  statHealthy: document.getElementById("stat-healthy"),
  statQw: document.getElementById("stat-qw"),
  statMw: document.getElementById("stat-mw"),
  statThrottled: document.getElementById("stat-throttled"),
  connDot: document.getElementById("conn-dot"),
  connStatus: document.getElementById("conn-status"),
  coordinatorUrl: document.getElementById("coordinator-url"),
  footerCoordinator: document.getElementById("footer-coordinator"),
};

document.addEventListener("DOMContentLoaded", () => {
  init();
});

async function init() {
  el.send.addEventListener("click", onSend);
  el.cancel.addEventListener("click", onCancel);
  el.clear.addEventListener("click", clearConversation);
  el.modelSelect.addEventListener("change", () => {
    state.selectedModel = el.modelSelect.value;
  });
  el.prompt.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      onSend();
    }
  });

  await loadConfig();
  await refreshWorkers();
  setInterval(refreshWorkers, WORKER_POLL_INTERVAL_MS);
}

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    el.coordinatorUrl.textContent = cfg.coordinator_url;
    el.footerCoordinator.textContent = `coord ${cfg.coordinator_url}`;
  } catch (e) {
    el.coordinatorUrl.textContent = "—";
  }
}

async function refreshWorkers() {
  try {
    const res = await fetch("/api/workers");
    if (!res.ok) {
      setConnection("danger", `coordinator ${res.status}`);
      return;
    }
    const workers = await res.json();
    state.workers = workers;
    renderWorkers();
    updateModelSelect();
    updateStats();
    setConnection("healthy", "connected");
  } catch (e) {
    setConnection("danger", "unreachable");
  }
}

function setConnection(level, label) {
  el.connDot.className = `conn-dot ${level}`;
  el.connStatus.textContent = label;
}

function updateStats() {
  const workers = state.workers;
  el.statWorkers.textContent = workers.length || "0";
  const healthy = workers.filter((w) => isWorkerHealthy(w)).length;
  el.statHealthy.textContent = `${healthy}/${workers.length || 0}`;
  if (workers.length === 0) {
    el.statQw.textContent = "—";
    el.statMw.textContent = "—";
    el.statThrottled.textContent = "—";
    return;
  }
  const avgQw =
    workers.reduce((sum, w) => sum + (w.telemetry?.qw || 0), 0) / workers.length;
  const avgMw =
    workers.reduce((sum, w) => sum + (w.telemetry?.mw || 0), 0) / workers.length;
  const throttled = workers.filter((w) => (w.telemetry?.theta_w || 0) > 0).length;
  el.statQw.textContent = avgQw.toFixed(2);
  el.statMw.textContent = `${(avgMw * 100).toFixed(0)}%`;
  el.statThrottled.textContent = `${throttled}/${workers.length}`;
}

function isWorkerHealthy(worker) {
  if (!isInferenceReady(worker)) return false;
  const tele = worker.telemetry || {};
  if ((tele.theta_w || 0) > 0) return false;
  if ((tele.mw || 0) > 0.95) return false;
  return true;
}

function isInferenceReady(worker) {
  if (worker.healthy === false) return false;
  const engine = worker.health?.engine || worker.health?.body?.engine || worker.engine;
  if (!engine) return worker.healthy === true;
  return engine.status === true;
}

function renderWorkers() {
  el.workerCount.textContent = String(state.workers.length);
  if (state.workers.length === 0) {
    el.workers.innerHTML =
      '<div class="empty">No workers registered. Start a worker with <code>relay start</code>.</div>';
    return;
  }
  el.workers.innerHTML = "";
  for (const worker of state.workers) {
    el.workers.appendChild(renderWorkerCard(worker));
  }
}

function renderWorkerCard(worker) {
  const tele = worker.telemetry || {};
  const card = document.createElement("div");
  card.className = "worker-card";

  const inferenceReady = isInferenceReady(worker);
  const stateLevel = !inferenceReady
    ? "danger"
    : (tele.theta_w || 0) > 0
    ? "danger"
    : (tele.mw || 0) > 0.85
    ? "warn"
    : "healthy";
  const stateLabel = !inferenceReady ? "unhealthy" : stateLevel;

  const head = document.createElement("div");
  head.className = "worker-card-head";
  head.innerHTML = `
    <span class="worker-id">${escapeHtml(worker.node_id)}</span>
    <span class="worker-state ${stateLevel}">${stateLabel}</span>
  `;
  card.appendChild(head);

  const meta = document.createElement("div");
  meta.className = "worker-meta";
  meta.textContent = worker.address || "(no address)";
  card.appendChild(meta);

  if (!inferenceReady) {
    const detail = document.createElement("div");
    detail.className = "worker-health-detail";
    detail.textContent = healthDetail(worker);
    card.appendChild(detail);
  }

  const bars = document.createElement("div");
  bars.className = "worker-bars";
  bars.appendChild(makeBar("q_w", tele.qw || 0, 8));
  bars.appendChild(makeBar("m_w", (tele.mw || 0) * 100, 100, "%"));
  bars.appendChild(makeBar("j_w", tele.jw || 0, 50, "ms"));
  bars.appendChild(makeBar("θ_w", tele.theta_w || 0, 1));
  card.appendChild(bars);

  return card;
}

function healthDetail(worker) {
  const engine = worker.health?.engine || worker.health?.body?.engine || {};
  return engine.detail || worker.health?.detail || "inference engine unavailable";
}

function makeBar(label, value, max, suffix = "") {
  const ratio = Math.max(0, Math.min(1, max > 0 ? value / max : 0));
  const fillClass = ratio > 0.9 ? "danger" : ratio > 0.7 ? "warn" : "";
  const displayValue =
    suffix === "%"
      ? `${value.toFixed(0)}%`
      : suffix === "ms"
      ? `${value.toFixed(1)}ms`
      : Number.isInteger(value)
      ? `${value}`
      : value.toFixed(2);
  const node = document.createElement("div");
  node.className = "bar";
  node.innerHTML = `
    <div class="bar-label">
      <span>${label}</span>
      <span class="value">${displayValue}</span>
    </div>
    <div class="bar-track"><div class="bar-fill ${fillClass}" style="width:${ratio * 100}%"></div></div>
  `;
  return node;
}

function updateModelSelect() {
  const models = new Set();
  for (const worker of state.workers) {
    for (const model of worker.models || []) {
      if (model && model.id) models.add(model.id);
    }
  }
  const previous = el.modelSelect.value || state.selectedModel;
  el.modelSelect.innerHTML = "";

  // "auto" — coordinator's scheduler picks the worker regardless of model id.
  const autoOption = document.createElement("option");
  autoOption.value = "auto";
  autoOption.textContent = "auto (coordinator picks)";
  el.modelSelect.appendChild(autoOption);

  for (const id of models) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = id;
    el.modelSelect.appendChild(option);
  }

  // Restore prior choice if it's still valid, else default to auto.
  if (previous && (previous === "auto" || models.has(previous))) {
    el.modelSelect.value = previous;
  } else {
    el.modelSelect.value = "auto";
  }
  state.selectedModel = el.modelSelect.value;
}

function clearConversation() {
  state.conversation = [];
  el.messages.innerHTML =
    '<div class="empty messages-empty"><div class="empty-title">Cleared.</div><div class="empty-sub">Send a message to route a new request.</div></div>';
  el.detailBody.innerHTML = '<div class="empty">No request yet.</div>';
  el.detailCost.textContent = "—";
}

async function onSend() {
  const content = el.prompt.value.trim();
  if (!content || state.abortController) return;
  const model = state.selectedModel || el.modelSelect.value;
  if (!model) {
    el.composerHint.textContent = "No model available. Start a worker first.";
    return;
  }

  el.prompt.value = "";
  state.conversation.push({ role: "user", content });
  ensureMessagesContainer();
  appendMessageBubble("user", content);

  const assistantBubble = appendMessageBubble("assistant", "");
  const startedAt = performance.now();
  state.abortController = new AbortController();
  toggleSending(true);
  el.composerHint.textContent = "streaming…";

  // Omit the model field when "auto" so the scheduler picks across all workers.
  const requestBody = { messages: state.conversation, stream: true };
  if (model && model !== "auto") {
    requestBody.model = model;
  }

  try {
    const { content: assistantText, headers } = await streamCompletion(
      requestBody,
      (chunk) => {
        assistantBubble.body.textContent += chunk;
        scrollMessagesToBottom();
      },
      state.abortController.signal,
    );
    state.conversation.push({ role: "assistant", content: assistantText });

    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    decorateAssistantBubble(assistantBubble, headers, elapsedSeconds, assistantText);
    renderDecisionDetail(headers, elapsedSeconds, assistantText);
    el.composerHint.textContent = `Done in ${elapsedSeconds.toFixed(2)}s`;
  } catch (e) {
    if (e.name === "AbortError") {
      assistantBubble.body.textContent += "\n[cancelled]";
      el.composerHint.textContent = "cancelled";
    } else {
      assistantBubble.body.textContent = `error: ${e.message || e}`;
      el.composerHint.textContent = "error";
    }
  } finally {
    state.abortController = null;
    toggleSending(false);
  }
}

function onCancel() {
  if (state.abortController) {
    state.abortController.abort();
  }
}

function toggleSending(active) {
  el.send.disabled = active;
  el.cancel.disabled = !active;
}

async function streamCompletion(body, onDelta, signal) {
  const response = await fetch("/api/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errText.slice(0, 200)}`);
  }
  const headers = {
    worker: response.headers.get("x-relay-worker"),
    cost: response.headers.get("x-relay-cost"),
    matchedTokens: response.headers.get("x-relay-matched-tokens"),
    promptTokens: response.headers.get("x-relay-prompt-tokens"),
    overlap: response.headers.get("x-relay-overlap"),
    attempts: response.headers.get("x-relay-attempts"),
  };

  if (!response.body) {
    const text = await response.text();
    onDelta(text);
    return { content: text, headers };
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let content = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      try {
        const json = JSON.parse(data);
        const delta = json?.choices?.[0]?.delta?.content;
        if (delta) {
          content += delta;
          onDelta(delta);
        }
      } catch (_) {
        // Ignore non-JSON SSE lines.
      }
    }
  }
  return { content, headers };
}

function ensureMessagesContainer() {
  const empty = el.messages.querySelector(".messages-empty");
  if (empty) empty.remove();
}

function appendMessageBubble(role, initialText) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  wrap.innerHTML = `
    <span class="message-role">${role}</span>
    <div class="message-body"></div>
    <div class="message-badge"></div>
  `;
  const body = wrap.querySelector(".message-body");
  const badge = wrap.querySelector(".message-badge");
  body.textContent = initialText;
  el.messages.appendChild(wrap);
  scrollMessagesToBottom();
  return { wrap, body, badge };
}

function decorateAssistantBubble(bubble, headers, elapsedSeconds, text) {
  if (!bubble.badge) return;
  bubble.badge.innerHTML = "";
  if (headers.worker) {
    bubble.badge.appendChild(makePill(`worker ${headers.worker}`, "worker"));
  }
  if (headers.cost) {
    bubble.badge.appendChild(makePill(`cost ${parseFloat(headers.cost).toFixed(3)}`));
  }
  if (headers.matchedTokens) {
    bubble.badge.appendChild(makePill(`match ${headers.matchedTokens}t`));
  }
  const attempts = parseInt(headers.attempts || "1", 10);
  if (attempts > 1) {
    // Only call out retries — single-attempt is the boring default.
    bubble.badge.appendChild(makePill(`${attempts} tries`));
  }
  bubble.badge.appendChild(makePill(`${elapsedSeconds.toFixed(2)}s`));
  const tokens = Math.max(1, Math.round(text.length / 4));
  bubble.badge.appendChild(makePill(`~${(tokens / Math.max(0.01, elapsedSeconds)).toFixed(1)} tok/s`));
}

function makePill(label, variant = "") {
  const pill = document.createElement("span");
  pill.className = `pill ${variant}`.trim();
  pill.textContent = label;
  return pill;
}

function renderDecisionDetail(headers, elapsedSeconds, text) {
  if (!headers.worker) {
    el.detailBody.innerHTML = '<div class="empty">No headers returned.</div>';
    el.detailCost.textContent = "—";
    return;
  }
  el.detailCost.textContent = headers.cost
    ? parseFloat(headers.cost).toFixed(3)
    : "—";
  const promptTokens = headers.promptTokens || "—";
  const matchedTokens = headers.matchedTokens || "0";
  const overlap = headers.overlap
    ? `${(parseFloat(headers.overlap) * 100).toFixed(0)}%`
    : "—";

  const approxTokens = Math.max(1, Math.round(text.length / 4));
  const tokRate = (approxTokens / Math.max(0.01, elapsedSeconds)).toFixed(1);

  el.detailBody.innerHTML = `
    <div class="detail-section">
      <div class="detail-section-title">Selected Worker</div>
      <dl class="detail-kv">
        <dt>node_id</dt><dd class="accent">${escapeHtml(headers.worker)}</dd>
        <dt>cost</dt><dd>${headers.cost ? parseFloat(headers.cost).toFixed(4) : "—"}</dd>
      </dl>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Prefix Cache</div>
      <dl class="detail-kv">
        <dt>prompt tokens</dt><dd>${promptTokens}</dd>
        <dt>matched tokens</dt><dd>${matchedTokens}</dd>
        <dt>overlap</dt><dd>${overlap}</dd>
      </dl>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Timing</div>
      <dl class="detail-kv">
        <dt>elapsed</dt><dd>${elapsedSeconds.toFixed(2)}s</dd>
        <dt>tok/s (approx)</dt><dd>${tokRate}</dd>
      </dl>
    </div>
  `;
}

function scrollMessagesToBottom() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
