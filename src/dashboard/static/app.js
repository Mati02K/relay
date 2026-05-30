// Relay dashboard front-end.
//
// Three views switchable from the top bar:
//   - map      hub-and-spoke SVG topology of coordinator + workers
//   - chat     test chat UI (streams through the coordinator)
//   - settings live scheduler weight tuning (5 sliders + Apply)
//
// State is global because the dashboard is small; switching views is just
// hiding/showing top-level <main> elements. The map view polls /api/workers
// every WORKER_POLL_INTERVAL_MS; the chat view does its own bookkeeping;
// the settings view only fetches when first opened.

const WORKER_POLL_INTERVAL_MS = 3000;
const PREFILL_HISTORY_LENGTH = 60;

const state = {
  view: "map",
  conversation: [],
  workers: [],
  selectedModel: null,
  abortController: null,
  weights: null,
  weightsBaseline: null,
  weightsDirty: false,
  workerWeights: null,
  workerWeightsBaseline: null,
  mode: null,
  modeBaseline: null,
  drawerWorkerId: null,
  prefillHistory: new Map(),
};

const el = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  wireGlobalControls();
  wireChat();
  wireSettings();
  wireDrawer();
  loadConfig();
  refreshWorkers();
  setInterval(refreshWorkers, WORKER_POLL_INTERVAL_MS);
});

function cacheElements() {
  el.body = document.body;
  el.views = {
    map: document.getElementById("view-map"),
    chat: document.getElementById("view-chat"),
    settings: document.getElementById("view-settings"),
  };
  el.brandBtn = document.getElementById("brand-btn");
  el.testBtn = document.getElementById("test-btn");
  el.settingsBtn = document.getElementById("settings-btn");
  el.connDot = document.getElementById("conn-dot");
  el.connStatus = document.getElementById("conn-status");
  el.coordinatorUrl = document.getElementById("coordinator-url");
  el.footerCoordinator = document.getElementById("footer-coordinator");
  el.footerHint = document.getElementById("footer-hint");

  // Map view
  el.mapSvg = document.getElementById("map-svg");
  el.mapCanvas = document.getElementById("map-canvas");
  el.mapEmpty = document.getElementById("map-empty");
  el.statWorkers = document.getElementById("stat-workers");
  el.statHealthy = document.getElementById("stat-healthy");
  el.statThermal = document.getElementById("stat-thermal");
  el.statJitter = document.getElementById("stat-jitter");

  // Chat view
  el.messages = document.getElementById("messages");
  el.prompt = document.getElementById("prompt");
  el.send = document.getElementById("send-btn");
  el.cancel = document.getElementById("cancel-btn");
  el.clear = document.getElementById("clear-btn");
  el.modelSelect = document.getElementById("model-select");
  el.composerHint = document.getElementById("composer-hint");
  el.detailBody = document.getElementById("detail-body");
  el.detailCost = document.getElementById("detail-cost");

  // Settings view
  el.weightsGrid = document.getElementById("weights-grid");
  el.workerTableBody = document.getElementById("worker-table-body");
  el.applyBtn = document.getElementById("apply-btn");
  el.resetBtn = document.getElementById("reset-btn");
  el.applyStatus = document.getElementById("apply-status");
  el.modeToggle = document.getElementById("mode-toggle");
  el.modeButtons = el.modeToggle ? Array.from(el.modeToggle.querySelectorAll(".mode-btn")) : [];

  // Drawer
  el.drawer = document.getElementById("worker-drawer");
  el.drawerTitle = document.getElementById("drawer-title");
  el.drawerSub = document.getElementById("drawer-sub");
  el.drawerBody = document.getElementById("drawer-body");
  el.drawerClose = document.getElementById("drawer-close");
}

// ============================ View switching ============================

function wireGlobalControls() {
  el.brandBtn.addEventListener("click", () => switchView("map"));
  el.testBtn.addEventListener("click", () => switchView("chat"));
  el.settingsBtn.addEventListener("click", () => switchView("settings"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!el.drawer.hidden) {
        closeDrawer();
      } else if (state.view !== "map") {
        switchView("map");
      }
    }
  });
}

function switchView(name) {
  if (!(name in el.views)) return;
  state.view = name;
  el.body.dataset.view = name;
  for (const [key, node] of Object.entries(el.views)) {
    node.hidden = key !== name;
  }
  // Highlight the active nav button.
  el.testBtn.classList.toggle("active", name === "chat");
  el.settingsBtn.classList.toggle("active", name === "settings");
  // Footer hint is only relevant in the chat view.
  el.footerHint.hidden = name !== "chat";
  if (name === "settings") {
    if (state.weights === null) fetchWeights();
    fetchWorkerWeights();
    fetchSchedulerMode();
  }
  if (name === "map") {
    // Re-render the map so layout fills the now-visible canvas.
    renderMap();
  }
}

// ============================ Config / connection ============================

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

function setConnection(level, label) {
  el.connDot.className = `conn-dot ${level}`;
  el.connStatus.textContent = label;
}

// ============================ Worker polling + stats ============================

async function refreshWorkers() {
  try {
    const res = await fetch("/api/workers");
    if (!res.ok) {
      setConnection("danger", `coordinator ${res.status}`);
      return;
    }
    state.workers = await res.json();
    setConnection("healthy", "connected");
    recordPrefillHistory();
    updateStats();
    updateModelSelect();
    renderMap();
    refreshDrawerIfOpen();
    if (state.view === "settings" && state.workerWeights !== null) {
      renderWorkerTable();
    }
  } catch (e) {
    setConnection("danger", "unreachable");
  }
}

function updateStats() {
  const workers = state.workers;
  el.statWorkers.textContent = workers.length || "0";
  const healthy = workers.filter(workerInferenceReady).length;
  el.statHealthy.textContent = `${healthy}/${workers.length || 0}`;
  if (workers.length === 0) {
    el.statThermal.textContent = "—";
    el.statJitter.textContent = "—";
    return;
  }
  const thermals = workers.map((w) => Number(w.telemetry?.theta_w || 0));
  const jitters = workers.map((w) => Number(w.telemetry?.jw || 0));
  el.statThermal.textContent = avg(thermals).toFixed(2);
  el.statJitter.textContent = `${avg(jitters).toFixed(1)}ms`;
}

function avg(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function workerInferenceReady(worker) {
  const engine = worker.health?.engine || worker.health?.body?.engine || {};
  if (engine.status === false) return false;
  return worker.healthy !== false;
}

function workerState(worker) {
  if (!workerInferenceReady(worker)) return "danger";
  const tele = worker.telemetry || {};
  const thermal = Number(tele.thermal?.pressure ?? tele.theta_w ?? 0);
  if (thermal >= 0.65) return "danger";
  if (thermal >= 0.35) return "warn";
  const mw = Number(tele.mw || 0);
  if (mw >= 0.85) return "warn";
  return "healthy";
}

// ============================ Map view ============================

function renderMap() {
  const svg = el.mapSvg;
  const canvas = el.mapCanvas;
  // Make SVG fill the canvas; recompute every render so resizes work.
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(400, rect.width);
  const h = Math.max(280, rect.height);
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);

  const workers = state.workers;
  // Use style.display directly — the CSS `display: flex` on .map-empty would
  // override the [hidden] attribute's implicit `display: none` otherwise.
  el.mapEmpty.style.display = workers.length > 0 ? "none" : "flex";

  // Hub-and-spoke layout: coordinator at center, workers around the rim.
  const cx = w / 2;
  const cy = h / 2;
  const ringRadius = Math.min(w, h) * 0.36;

  const edges = [];
  const workerNodes = [];
  workers.forEach((worker, index) => {
    const angle = workers.length === 1
      ? -Math.PI / 2
      : (index / workers.length) * Math.PI * 2 - Math.PI / 2;
    const x = cx + Math.cos(angle) * ringRadius;
    const y = cy + Math.sin(angle) * ringRadius;
    edges.push({ x1: cx, y1: cy, x2: x, y2: y, level: workerState(worker) });
    workerNodes.push({ worker, x, y, level: workerState(worker) });
  });

  svg.innerHTML = "";
  svg.appendChild(svgDefs());

  // edges first so they sit behind nodes
  for (const edge of edges) {
    svg.appendChild(svgEdge(edge));
  }

  // coordinator hub
  svg.appendChild(svgCoordinator(cx, cy));

  // workers
  for (const node of workerNodes) {
    svg.appendChild(svgWorker(node));
  }
}

function svgDefs() {
  const defs = svgEl("defs");
  defs.innerHTML = `
    <radialGradient id="hubGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(245,176,33,0.65)"/>
      <stop offset="55%" stop-color="rgba(245,176,33,0.18)"/>
      <stop offset="100%" stop-color="rgba(245,176,33,0)"/>
    </radialGradient>
    <radialGradient id="workerGlowHealthy" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(74,222,128,0.55)"/>
      <stop offset="100%" stop-color="rgba(74,222,128,0)"/>
    </radialGradient>
    <radialGradient id="workerGlowWarn" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(251,191,36,0.55)"/>
      <stop offset="100%" stop-color="rgba(251,191,36,0)"/>
    </radialGradient>
    <radialGradient id="workerGlowDanger" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="rgba(239,68,68,0.55)"/>
      <stop offset="100%" stop-color="rgba(239,68,68,0)"/>
    </radialGradient>
  `;
  return defs;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function svgEdge({ x1, y1, x2, y2, level }) {
  const group = svgEl("g", { class: `map-edge level-${level}` });
  group.appendChild(svgEl("line", {
    x1, y1, x2, y2,
    "stroke-linecap": "round",
  }));
  // Animated pulse dot travelling from worker to coordinator.
  const pulse = svgEl("circle", { r: 2.5, class: "map-pulse" });
  const dx = x1 - x2;
  const dy = y1 - y2;
  const len = Math.hypot(dx, dy);
  const period = Math.max(1.6, len / 220);
  const animateX = svgEl("animate", {
    attributeName: "cx",
    from: x2, to: x1,
    dur: `${period}s`,
    repeatCount: "indefinite",
  });
  const animateY = svgEl("animate", {
    attributeName: "cy",
    from: y2, to: y1,
    dur: `${period}s`,
    repeatCount: "indefinite",
  });
  pulse.appendChild(animateX);
  pulse.appendChild(animateY);
  group.appendChild(pulse);
  return group;
}

function svgCoordinator(cx, cy) {
  const group = svgEl("g", { class: "map-node coordinator-node", role: "button", tabindex: "0" });
  group.setAttribute("aria-label", "Coordinator — click to open settings");
  group.appendChild(svgEl("circle", { cx, cy, r: 64, fill: "url(#hubGlow)" }));
  group.appendChild(svgEl("circle", { cx, cy, r: 36, class: "coord-disc" }));
  const label = svgEl("text", {
    x: cx, y: cy + 4,
    "text-anchor": "middle",
    class: "coord-label",
  });
  label.textContent = "COORD";
  group.appendChild(label);
  const sub = svgEl("text", {
    x: cx, y: cy + 60,
    "text-anchor": "middle",
    class: "coord-sub",
  });
  sub.textContent = "click → settings";
  group.appendChild(sub);
  group.addEventListener("click", () => switchView("settings"));
  group.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      switchView("settings");
    }
  });
  return group;
}

function svgWorker(node) {
  const { worker, x, y, level } = node;
  const group = svgEl("g", { class: `map-node worker-node level-${level}`, role: "button", tabindex: "0" });
  group.setAttribute("aria-label", `Worker ${worker.node_id} — click for details`);
  group.appendChild(svgEl("circle", { cx: x, cy: y, r: 38, fill: `url(#workerGlow${cap(level)})` }));
  group.appendChild(svgEl("circle", { cx: x, cy: y, r: 22, class: "worker-disc" }));
  const label = svgEl("text", {
    x, y: y + 4,
    "text-anchor": "middle",
    class: "worker-letter",
  });
  label.textContent = (worker.node_id || "?").charAt(0).toUpperCase();
  group.appendChild(label);
  const idText = svgEl("text", {
    x, y: y + 42,
    "text-anchor": "middle",
    class: "worker-id",
  });
  idText.textContent = worker.node_id;
  group.appendChild(idText);
  group.addEventListener("click", () => openDrawer(worker.node_id));
  group.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openDrawer(worker.node_id);
    }
  });
  return group;
}

function cap(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ============================ Prefill speed history ============================

function recordPrefillHistory() {
  const liveIds = new Set();
  for (const worker of state.workers) {
    liveIds.add(worker.node_id);
    const value = Number(worker.telemetry?.sprefill_tokens_per_sec ?? 0);
    let history = state.prefillHistory.get(worker.node_id);
    if (!history) {
      history = [];
      state.prefillHistory.set(worker.node_id, history);
    }
    history.push(value);
    if (history.length > PREFILL_HISTORY_LENGTH) {
      history.splice(0, history.length - PREFILL_HISTORY_LENGTH);
    }
  }
  for (const nodeId of state.prefillHistory.keys()) {
    if (!liveIds.has(nodeId)) state.prefillHistory.delete(nodeId);
  }
}

function renderPrefillSparkline(nodeId) {
  const history = state.prefillHistory.get(nodeId) || [];
  if (history.length < 2) {
    return '<div class="sparkline-empty">collecting samples…</div>';
  }
  const w = 220;
  const h = 56;
  const padY = 6;
  const maxObserved = Math.max(...history);
  const minObserved = Math.min(...history);
  // Pad the visible range so flat lines don't pin to the top edge, and small
  // fluctuations on small absolute values still register visually.
  const rangeFloor = Math.max(maxObserved * 0.15, 5);
  const yMax = maxObserved + rangeFloor;
  const yMin = Math.max(0, minObserved - rangeFloor);
  const span = Math.max(yMax - yMin, 1);
  const stepX = w / (PREFILL_HISTORY_LENGTH - 1);
  const offsetX = w - (history.length - 1) * stepX;
  const coords = history.map((value, idx) => {
    const x = offsetX + idx * stepX;
    const y = h - padY - ((value - yMin) / span) * (h - 2 * padY);
    return { x, y };
  });
  const linePoints = coords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const fillPoints = `${coords[0].x.toFixed(1)},${(h - padY).toFixed(1)} ${linePoints} ${coords[coords.length - 1].x.toFixed(1)},${(h - padY).toFixed(1)}`;
  return `
    <svg class="sparkline" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
      <polygon class="sparkline-fill" points="${fillPoints}" />
      <polyline class="sparkline-line" points="${linePoints}" />
    </svg>
  `;
}

// ============================ Worker detail drawer ============================

function wireDrawer() {
  el.drawerClose.addEventListener("click", closeDrawer);
}

function openDrawer(nodeId) {
  state.drawerWorkerId = nodeId;
  refreshDrawerIfOpen();
  el.drawer.hidden = false;
  requestAnimationFrame(() => el.drawer.classList.add("open"));
}

function closeDrawer() {
  state.drawerWorkerId = null;
  el.drawer.classList.remove("open");
  setTimeout(() => { el.drawer.hidden = true; }, 220);
}

function refreshDrawerIfOpen() {
  if (!state.drawerWorkerId) return;
  const worker = state.workers.find((w) => w.node_id === state.drawerWorkerId);
  if (!worker) {
    el.drawerTitle.textContent = state.drawerWorkerId;
    el.drawerSub.textContent = "Worker no longer registered.";
    el.drawerBody.innerHTML = '<div class="empty">This worker has left the cluster.</div>';
    return;
  }
  const tele = worker.telemetry || {};
  const thermal = tele.thermal || {};
  const models = (worker.models || []).map((m) => m.id || m.filename || "?").join(", ") || "—";
  const engine = worker.health?.engine?.engine || worker.health?.body?.engine?.engine || "—";
  const engineDetail = worker.health?.engine?.detail || worker.health?.body?.engine?.detail || "—";
  const level = workerState(worker);

  el.drawerTitle.textContent = worker.node_id;
  el.drawerSub.textContent = worker.address || "—";
  el.drawerBody.innerHTML = `
    <div class="drawer-section">
      <div class="drawer-section-title">Health</div>
      <dl class="drawer-kv">
        <dt>state</dt><dd class="state-${level}">${level}</dd>
        <dt>engine</dt><dd>${escapeHtml(engine)} (${escapeHtml(engineDetail)})</dd>
        <dt>http</dt><dd>${escapeHtml(String(worker.health?.status_code ?? "—"))}</dd>
        <dt>worker weight</dt><dd>${Number(worker.weight ?? 0).toFixed(2)}</dd>
      </dl>
    </div>
    <div class="drawer-section">
      <div class="drawer-section-title">Telemetry</div>
      <dl class="drawer-kv">
        <dt>queue (q_w)</dt><dd>${Number(tele.qw ?? 0).toFixed(0)}</dd>
        <dt>memory (m_w)</dt><dd>${(Number(tele.mw ?? 0) * 100).toFixed(0)}%</dd>
        <dt>jitter (j_w)</dt><dd>${Number(tele.jw ?? 0).toFixed(1)} ms</dd>
        <dt>thermal (θ_w)</dt><dd>${Number(tele.theta_w ?? 0).toFixed(2)}</dd>
        <dt>thermal state</dt><dd class="state-${level}">${escapeHtml(thermal.state || "—")}</dd>
        <dt>cpu pressure</dt><dd>${Number(thermal.cpu_pressure ?? 0).toFixed(2)}</dd>
        <dt>gpu pressure</dt><dd>${Number(thermal.gpu_pressure ?? 0).toFixed(2)}</dd>
      </dl>
    </div>
    <div class="drawer-section">
      <div class="drawer-section-title">Prefill speed</div>
      <dl class="drawer-kv">
        <dt>current</dt><dd>${Number(tele.sprefill_tokens_per_sec ?? 0).toFixed(1)} tok/s</dd>
      </dl>
      ${renderPrefillSparkline(worker.node_id)}
    </div>
    <div class="drawer-section">
      <div class="drawer-section-title">Models</div>
      <div class="drawer-models">${escapeHtml(models)}</div>
    </div>
  `;
}

// ============================ Settings view ============================

// Compact knob row: one dial per scheduler cost term. `max` distinguishes
// the base 5 (in [0, 1]) from the `nu` RouteLLM term (in [0, 5]).
const WEIGHT_FIELDS = [
  {
    key: "queue",
    label: "queue",
    max: 1,
    step: 0.01,
    hint: "q_w — punishes deep queues. Higher spreads load away from busy workers.",
  },
  {
    key: "prefix_miss",
    label: "cache",
    max: 1,
    step: 0.01,
    hint: "prefix_miss (1 − overlap) — rewards KV-cache reuse on long, repetitive prompts.",
  },
  {
    key: "memory",
    label: "memory",
    max: 1,
    step: 0.01,
    hint: "m_w — punishes KV/RAM pressure. Higher avoids OOM-prone workers.",
  },
  {
    key: "jitter",
    label: "jitter",
    max: 1,
    step: 0.01,
    hint: "j_w / j_max — punishes flaky/high-latency network paths.",
  },
  {
    key: "thermal",
    label: "thermal",
    max: 1,
    step: 0.01,
    hint: "θ_w — punishes thermally-throttled workers. 0 ignores temperature.",
  },
  {
    key: "nu",
    label: "ν · quality",
    max: 5,
    step: 0.1,
    hint: "RouteLLM: ν × complexity × (1 − quality). 0 disables the chart entirely (no skill filter, no quality term). 2–3 is a good default with the heuristic classifier.",
  },
];

function wireSettings() {
  el.applyBtn.addEventListener("click", applyAllWeights);
  el.resetBtn.addEventListener("click", resetAllDirty);
  for (const btn of el.modeButtons) {
    btn.addEventListener("click", () => {
      const next = btn.dataset.mode;
      if (!next || next === state.mode) return;
      state.mode = next;
      renderMode();
      renderWeights();
      renderWorkerTable();
      markWeightsDirty(detectDirty());
    });
  }
}

function resetAllDirty() {
  // Reset mode first so the downstream renders pick up the correct
  // locked/unlocked state in one pass instead of flashing through stale
  // locked knobs before they unlock.
  if (state.modeBaseline !== null) {
    state.mode = state.modeBaseline;
  }
  if (state.weightsBaseline) {
    state.weights = { ...state.weightsBaseline };
  }
  if (state.workerWeightsBaseline) {
    state.workerWeights = { ...state.workerWeightsBaseline };
  }
  renderMode();
  renderWeights();
  renderWorkerTable();
  setApplyStatus("reset to last applied values.", "neutral");
  markWeightsDirty(false);
}

async function fetchWeights() {
  setApplyStatus("loading…", "neutral");
  try {
    const res = await fetch("/api/scheduler/weights");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const weights = await res.json();
    state.weights = weights;
    state.weightsBaseline = { ...weights };
    renderWeights();
    markWeightsDirty(false);
    setApplyStatus("", "neutral");
  } catch (e) {
    setApplyStatus(`Failed to load weights: ${e.message || e}`, "error");
  }
}

function renderWeights() {
  if (!state.weights) {
    el.weightsGrid.innerHTML = '<div class="empty">Loading scheduler weights…</div>';
    return;
  }
  const locked = state.mode === "round_robin";
  el.weightsGrid.classList.toggle("locked", locked);
  el.weightsGrid.innerHTML = "";
  for (const field of WEIGHT_FIELDS) {
    const value = Number(state.weights[field.key] ?? 0);
    const baseline = Number(state.weightsBaseline?.[field.key] ?? value);
    const knob = document.createElement("div");
    knob.className = "knob";
    if (value !== baseline) knob.classList.add("dirty");
    if (locked) knob.classList.add("locked");
    knob.innerHTML = `
      <span class="knob-key">${escapeHtml(field.label)}</span>
      <div class="knob-dial" data-bind="${field.key}-dial">
        <svg viewBox="0 0 60 60">
          <circle class="knob-dial-track" cx="30" cy="30" r="24" />
          <circle class="knob-dial-fill"  cx="30" cy="30" r="24" stroke-dasharray="151" stroke-dashoffset="${dialOffset(value, field.max)}" />
        </svg>
        <span class="knob-dial-value" data-bind="${field.key}-num">${formatWeight(value, field.max)}</span>
      </div>
      <input type="range" min="0" max="${field.max}" step="${field.step}" value="${value}" data-bind="${field.key}-range" ${locked ? "disabled" : ""} />
      <div class="knob-tooltip">${escapeHtml(locked ? "round-robin mode — cost weights ignored" : field.hint)}</div>
    `;
    el.weightsGrid.appendChild(knob);
    const range = knob.querySelector(`input[data-bind='${field.key}-range']`);
    const numEl = knob.querySelector(`[data-bind='${field.key}-num']`);
    const fillEl = knob.querySelector(".knob-dial-fill");
    range.addEventListener("input", () => {
      if (locked) return;
      const v = clampRange(parseFloat(range.value), 0, field.max);
      state.weights[field.key] = v;
      numEl.textContent = formatWeight(v, field.max);
      fillEl.setAttribute("stroke-dashoffset", dialOffset(v, field.max));
      const baselineNow = Number(state.weightsBaseline?.[field.key] ?? v);
      knob.classList.toggle("dirty", v !== baselineNow);
      markWeightsDirty(detectDirty());
    });
  }
}

function renderMode() {
  if (!el.modeButtons.length) return;
  const active = state.mode || "cost";
  for (const btn of el.modeButtons) {
    const isActive = btn.dataset.mode === active;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  }
}

async function fetchSchedulerMode() {
  try {
    const res = await fetch("/api/scheduler/mode");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json();
    const mode = typeof body.mode === "string" ? body.mode : "cost";
    state.mode = mode;
    state.modeBaseline = mode;
    renderMode();
    renderWeights();
    renderWorkerTable();
    markWeightsDirty(detectDirty());
  } catch (e) {
    setApplyStatus(`Failed to load scheduler mode: ${e.message || e}`, "error");
  }
}

// SVG arc fill: circle r=24 has circumference 2π·24 ≈ 151. Drive
// `stroke-dashoffset` from 151 (empty) down to 0 (full ring).
function dialOffset(value, max) {
  const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  return (151 * (1 - frac)).toFixed(2);
}

function formatWeight(value, max) {
  return max > 1 ? value.toFixed(1) : value.toFixed(2);
}

function detectDirty() {
  const weightsDirty =
    !!state.weights &&
    !!state.weightsBaseline &&
    WEIGHT_FIELDS.some(
      (f) => (state.weights[f.key] ?? 0) !== (state.weightsBaseline[f.key] ?? 0),
    );
  const workerWeightDirty =
    !!state.workerWeights &&
    !!state.workerWeightsBaseline &&
    diffMaps(state.workerWeights, state.workerWeightsBaseline);
  const modeDirty =
    state.mode !== null && state.modeBaseline !== null && state.mode !== state.modeBaseline;
  return weightsDirty || workerWeightDirty || modeDirty;
}

function diffMaps(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) {
    if ((a[key] ?? null) !== (b[key] ?? null)) return true;
  }
  return false;
}

function markWeightsDirty(dirty) {
  state.weightsDirty = dirty;
  el.applyBtn.disabled = !dirty;
}

async function fetchWorkerWeights() {
  try {
    const res = await fetch("/api/scheduler/worker_weights");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const weights = await res.json();
    state.workerWeights = { ...weights };
    state.workerWeightsBaseline = { ...weights };
    renderWorkerTable();
    markWeightsDirty(detectDirty());
  } catch (e) {
    setApplyStatus(`Failed to load worker weights: ${e.message || e}`, "error");
  }
}

function renderWorkerTable() {
  if (!el.workerTableBody) return;
  const locked = state.mode === "round_robin";
  el.workerTableBody.classList.toggle("locked", locked);
  const workers = state.workers || [];
  if (workers.length === 0) {
    el.workerTableBody.innerHTML = '<tr class="wt-empty"><td colspan="4">No workers registered.</td></tr>';
    return;
  }
  if (state.workerWeights === null) {
    el.workerTableBody.innerHTML = '<tr class="wt-empty"><td colspan="4">Loading worker overrides…</td></tr>';
    return;
  }
  el.workerTableBody.innerHTML = "";
  for (const worker of workers) {
    el.workerTableBody.appendChild(renderWorkerRow(worker, locked));
  }
}

function renderWorkerRow(worker, locked = false) {
  const nodeId = worker.node_id;
  const stateLevel = workerState(worker);
  const modelId = ((worker.models && worker.models[0]) || {}).id || "—";

  const weightBaseline = Number(worker.weight ?? 0);
  const weightOverride = state.workerWeights[nodeId];
  const weightEffective = weightOverride !== undefined && weightOverride !== null
    ? Number(weightOverride)
    : weightBaseline;

  const tr = document.createElement("tr");
  tr.dataset.node = nodeId;

  const baselineDirty = state.workerWeightsBaseline[nodeId];
  const dirty = (weightOverride ?? null) !== (baselineDirty ?? null);
  if (dirty) tr.classList.add("row-dirty");
  if (locked) tr.classList.add("row-locked");

  const resetDisabled = !dirty || locked;
  tr.innerHTML = `
    <td class="wt-id">
      <div class="wt-id-cell">
        <span class="wt-id-name">${escapeHtml(nodeId)}</span>
        <span class="wt-id-state state-${stateLevel}">${stateLevel}</span>
      </div>
    </td>
    <td class="wt-model"><span class="wt-model-cell" title="${escapeHtml(modelId)}">${escapeHtml(modelId)}</span></td>
    <td class="wt-weight">
      <div class="mini-slider">
        <input type="range" min="-1" max="1" step="0.01" value="${weightEffective}" data-bind="weight-range" ${locked ? "disabled" : ""} />
        <span class="mini-value ${weightOverride !== undefined && weightOverride !== null ? "overridden" : "baseline"}" data-bind="weight-num">${weightEffective.toFixed(2)}</span>
      </div>
    </td>
    <td class="wt-actions">
      <button type="button" class="wt-reset" data-bind="reset" ${resetDisabled ? "disabled" : ""}>reset</button>
    </td>
  `;

  if (locked) return tr;

  // Wire weight slider.
  const weightRange = tr.querySelector("input[data-bind='weight-range']");
  const weightNum = tr.querySelector("[data-bind='weight-num']");
  weightRange.addEventListener("input", () => {
    const v = clampRange(parseFloat(weightRange.value), -1, 1);
    if (Math.abs(v) < 1e-6) {
      delete state.workerWeights[nodeId];
      weightNum.textContent = weightBaseline.toFixed(2);
      weightNum.className = "mini-value baseline";
    } else {
      state.workerWeights[nodeId] = v;
      weightNum.textContent = v.toFixed(2);
      weightNum.className = "mini-value overridden";
    }
    updateRowDirty(tr, nodeId);
    markWeightsDirty(detectDirty());
  });

  // Per-row reset button.
  const resetBtn = tr.querySelector("[data-bind='reset']");
  resetBtn.addEventListener("click", () => {
    delete state.workerWeights[nodeId];
    renderWorkerTable();
    markWeightsDirty(detectDirty());
  });

  return tr;
}

function updateRowDirty(tr, nodeId) {
  const weightBaseline = state.workerWeightsBaseline[nodeId];
  const currentWeight = state.workerWeights[nodeId];
  const dirty = (currentWeight ?? null) !== (weightBaseline ?? null);
  tr.classList.toggle("row-dirty", dirty);
  const resetBtn = tr.querySelector("[data-bind='reset']");
  if (resetBtn) resetBtn.disabled = !dirty;
}

function clampRange(v, lo, hi) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(lo, Math.min(hi, v));
}

async function applyAllWeights() {
  if (!state.weightsDirty) return;
  setApplyStatus("applying…", "neutral");
  el.applyBtn.disabled = true;
  try {
    const [w1, w2, m] = await Promise.all([
      applySchedulerWeights(),
      applyWorkerWeightOverrides(),
      applySchedulerMode(),
    ]);
    if (w1) {
      state.weights = w1;
      state.weightsBaseline = { ...w1 };
      renderWeights();
    }
    if (w2 !== null) {
      state.workerWeights = { ...w2 };
      state.workerWeightsBaseline = { ...w2 };
    }
    if (m !== null) {
      state.mode = m;
      state.modeBaseline = m;
      renderMode();
    }
    renderWorkerTable();
    markWeightsDirty(false);
    setApplyStatus("applied. Scheduler now using new values.", "success");
  } catch (e) {
    setApplyStatus(`Apply failed: ${e.message || e}`, "error");
    markWeightsDirty(true);
  }
}

async function applySchedulerMode() {
  if (state.mode === null || state.modeBaseline === null) return null;
  if (state.mode === state.modeBaseline) return null;
  const res = await fetch("/api/scheduler/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: state.mode }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text.slice(0, 200) || `HTTP ${res.status}`);
  }
  const body = await res.json();
  return typeof body.mode === "string" ? body.mode : null;
}

async function applySchedulerWeights() {
  if (!state.weights || !state.weightsBaseline) return null;
  const dirty = WEIGHT_FIELDS.some(
    (f) => (state.weights[f.key] ?? 0) !== (state.weightsBaseline[f.key] ?? 0),
  );
  if (!dirty) return null;
  const res = await fetch("/api/scheduler/weights", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.weights),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text.slice(0, 200) || `HTTP ${res.status}`);
  }
  return await res.json();
}

async function applyWorkerWeightOverrides() {
  if (!state.workerWeights || !state.workerWeightsBaseline) return null;
  const payload = {};
  const keys = new Set([
    ...Object.keys(state.workerWeights),
    ...Object.keys(state.workerWeightsBaseline),
  ]);
  let dirty = false;
  for (const key of keys) {
    const current = state.workerWeights[key];
    const baseline = state.workerWeightsBaseline[key];
    if ((current ?? null) === (baseline ?? null)) continue;
    payload[key] = current === undefined ? null : current;
    dirty = true;
  }
  if (!dirty) return null;
  const res = await fetch("/api/scheduler/worker_weights", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text.slice(0, 200) || `HTTP ${res.status}`);
  }
  return await res.json();
}

function setApplyStatus(text, level) {
  el.applyStatus.textContent = text;
  el.applyStatus.className = `apply-status apply-${level}`;
  el.applyStatus.hidden = !text;
}

function clamp01(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

// ============================ Chat view ============================

function wireChat() {
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

  const requestBody = { messages: state.conversation, stream: true };
  if (model && model !== "auto") requestBody.model = model;

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
  if (state.abortController) state.abortController.abort();
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
      } catch (_) {}
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
  if (headers.worker) bubble.badge.appendChild(makePill(`worker ${headers.worker}`, "worker"));
  if (headers.cost) bubble.badge.appendChild(makePill(`cost ${parseFloat(headers.cost).toFixed(3)}`));
  if (headers.matchedTokens) bubble.badge.appendChild(makePill(`match ${headers.matchedTokens}t`));
  const attempts = parseInt(headers.attempts || "1", 10);
  if (attempts > 1) bubble.badge.appendChild(makePill(`${attempts} tries`));
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
  el.detailCost.textContent = headers.cost ? parseFloat(headers.cost).toFixed(3) : "—";
  const promptTokens = headers.promptTokens || "—";
  const matchedTokens = headers.matchedTokens || "0";
  const overlap = headers.overlap ? `${(parseFloat(headers.overlap) * 100).toFixed(0)}%` : "—";
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
        <dt>prompt tokens</dt><dd>${escapeHtml(String(promptTokens))}</dd>
        <dt>matched tokens</dt><dd>${escapeHtml(String(matchedTokens))}</dd>
        <dt>overlap</dt><dd>${escapeHtml(overlap)}</dd>
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
