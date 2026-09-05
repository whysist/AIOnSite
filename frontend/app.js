"use strict";

/**
 * AIOnSite frontend -- a plain static page (no build step, no framework)
 * that talks directly to the real FastAPI backend (src/api/app.py).
 *
 * Every value rendered here comes from an actual API response. Running a
 * task uses the streaming flow: POST /tasks/stream starts it in the
 * background and returns immediately, then this page opens a live
 * EventSource against GET /tasks/{id}/events and renders each real
 * pipeline step (classification, plan, node starts/completions, tool
 * calls, cache reuse, verification) as the backend actually records it --
 * nothing here is simulated or pre-scripted. Once the stream's terminal
 * "stream_end" event arrives, the full detailed view (pipeline, audit,
 * verification) is loaded the same way a past run is loaded.
 */

const LS_API_BASE = "aionsite.apiBase";
const LS_HISTORY = "aionsite.history";

const els = {
  apiBase: document.getElementById("apiBase"),
  healthDot: document.getElementById("healthDot"),
  statusCard: document.getElementById("statusCard"),
  stEnv: document.getElementById("stEnv"),
  stProvider: document.getElementById("stProvider"),
  stSovereign: document.getElementById("stSovereign"),
  stTools: document.getElementById("stTools"),
  refreshBtn: document.getElementById("refreshBtn"),
  historyList: document.getElementById("historyList"),
  globalError: document.getElementById("globalError"),
  taskInput: document.getElementById("taskInput"),
  confidentialInput: document.getElementById("confidentialInput"),
  runBtn: document.getElementById("runBtn"),
  liveCard: document.getElementById("liveCard"),
  livePill: document.getElementById("livePill"),
  liveList: document.getElementById("liveList"),
  resultCard: document.getElementById("resultCard"),
  resultExecId: document.getElementById("resultExecId"),
  resultBanners: document.getElementById("resultBanners"),
  resultMetrics: document.getElementById("resultMetrics"),
  finalAnswer: document.getElementById("finalAnswer"),
  verificationBlock: document.getElementById("verificationBlock"),
  pipelineCard: document.getElementById("pipelineCard"),
  nodeList: document.getElementById("nodeList"),
  auditCard: document.getElementById("auditCard"),
  auditList: document.getElementById("auditList"),
};

let activeExecId = null;
let runTimer = null;

// ---------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------

function apiBase() {
  return els.apiBase.value.trim().replace(/\/+$/, "") || "http://localhost:8000";
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmtMs(ms) {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function fmtTime(unixSeconds) {
  if (!unixSeconds) return "—";
  try {
    return new Date(unixSeconds * 1000).toLocaleTimeString();
  } catch {
    return String(unixSeconds);
  }
}

async function api(path, options) {
  const url = `${apiBase()}${path}`;
  let resp;
  try {
    resp = await fetch(url, {
      headers: { "content-type": "application/json" },
      ...options,
    });
  } catch (err) {
    throw new Error(
      `Could not reach ${url} -- is the backend running (python -m src.main --serve) ` +
      `and is the API base URL correct? (${err.message})`
    );
  }
  const text = await resp.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    // non-JSON body; fall through with body=null and raw text below
  }
  if (!resp.ok) {
    const detail = body?.message || body?.detail || text || resp.statusText;
    throw new Error(`${resp.status} ${resp.statusText}: ${detail}`);
  }
  return body;
}

function showError(message) {
  els.globalError.innerHTML = `<div class="error-box">${esc(message)}</div>`;
}

function clearError() {
  els.globalError.innerHTML = "";
}

function statusPillClass(status) {
  const s = String(status || "").toLowerCase();
  if (["completed", "satisfied", "ok", "true"].includes(s)) return "ok";
  if (["failed", "bad", "blocked", "false"].includes(s)) return "bad";
  if (["needs_review", "retrying", "degraded", "incomplete"].includes(s)) return "warn";
  if (["running", "pending"].includes(s)) return "info";
  return "mute";
}

// ---------------------------------------------------------------------
// persistence: API base + local run history (labels only -- the backend
// remains the single source of truth for everything else)
// ---------------------------------------------------------------------

function loadPersistedApiBase() {
  const saved = localStorage.getItem(LS_API_BASE);
  if (saved) els.apiBase.value = saved;
}

function persistApiBase() {
  localStorage.setItem(LS_API_BASE, apiBase());
}

function loadLocalHistory() {
  try {
    return JSON.parse(localStorage.getItem(LS_HISTORY) || "[]");
  } catch {
    return [];
  }
}

function rememberRun(execId, taskText) {
  const history = loadLocalHistory().filter((h) => h.id !== execId);
  history.unshift({ id: execId, task: taskText, ts: Date.now() });
  localStorage.setItem(LS_HISTORY, JSON.stringify(history.slice(0, 50)));
}

// ---------------------------------------------------------------------
// health + history sidebar
// ---------------------------------------------------------------------

async function refreshHealth() {
  try {
    const h = await api("/health");
    els.healthDot.className = "dot up";
    els.statusCard.querySelector("strong").innerHTML =
      '<span class="dot up"></span>Backend reachable';
    els.stEnv.textContent = h.environment ?? "—";
    els.stProvider.textContent = h.llm_provider ?? "—";
    els.stSovereign.textContent = h.sovereign_mode ? "on" : "off";
    els.stTools.textContent = Array.isArray(h.tools) ? h.tools.length : "—";
    clearError();
    return true;
  } catch (err) {
    els.healthDot.className = "dot down";
    els.statusCard.querySelector("strong").innerHTML =
      '<span class="dot down"></span>Backend unreachable';
    els.stEnv.textContent = els.stProvider.textContent = "—";
    els.stSovereign.textContent = els.stTools.textContent = "—";
    showError(err.message);
    return false;
  }
}

async function refreshHistoryList() {
  // Merge what the backend actually knows about (source of truth for
  // which execution ids exist) with locally-remembered task text (the
  // backend's GET /tasks only returns bare ids).
  let serverIds = [];
  try {
    serverIds = await api("/tasks");
  } catch {
    serverIds = null; // backend unreachable -- fall back to local-only view
  }
  const local = loadLocalHistory();
  const localById = new Map(local.map((h) => [h.id, h]));

  let ids;
  if (serverIds) {
    ids = [...serverIds].reverse(); // most recent last from the API -> show newest first
    // include any locally-remembered runs the (in-memory, per-process)
    // backend no longer knows about, clearly marked, so history survives
    // a backend restart instead of silently vanishing.
    for (const h of local) if (!ids.includes(h.id)) ids.push(h.id);
  } else {
    ids = local.map((h) => h.id);
  }

  if (ids.length === 0) {
    els.historyList.innerHTML = '<div class="history-empty">No runs yet this session.</div>';
    return;
  }

  els.historyList.innerHTML = ids
    .map((id) => {
      const meta = localById.get(id);
      const known = serverIds ? serverIds.includes(id) : true;
      const label = meta?.task ? esc(meta.task.slice(0, 60)) : known ? "(task text unknown)" : "(no longer on backend)";
      const active = id === activeExecId ? " active" : "";
      return `<button class="history-item${active}" data-exec-id="${esc(id)}">
        <span class="hid">${esc(id)}</span>
        <span class="htask">${label}</span>
      </button>`;
    })
    .join("");

  els.historyList.querySelectorAll(".history-item").forEach((btn) => {
    btn.addEventListener("click", () => loadExecution(btn.dataset.execId));
  });
}

// ---------------------------------------------------------------------
// running a task
// ---------------------------------------------------------------------

let activeEventSource = null;
let liveStepCount = 0;

function setRunning(isRunning, lastStepLabel) {
  els.runBtn.disabled = isRunning;
  if (isRunning) {
    if (!runTimer) {
      const started = Date.now();
      runTimer = setInterval(() => {
        const secs = Math.round((Date.now() - started) / 1000);
        const label = els.runBtn.dataset.lastStep || "starting...";
        els.runBtn.innerHTML = `<span class="spinner"></span> ${esc(label)} (${secs}s)`;
      }, 500);
    }
    if (lastStepLabel) els.runBtn.dataset.lastStep = lastStepLabel;
  } else {
    clearInterval(runTimer);
    runTimer = null;
    delete els.runBtn.dataset.lastStep;
    els.runBtn.textContent = "Run task";
  }
}

// ---------------------------------------------------------------------
// live step rendering -- one line per real backend event, as it happens
// ---------------------------------------------------------------------

function resetLivePanel() {
  liveStepCount = 0;
  els.liveCard.style.display = "";
  els.livePill.className = "pill info";
  els.livePill.textContent = "running";
  els.liveList.innerHTML = "";
}

function appendLiveEvent(ev) {
  liveStepCount += 1;
  const row = document.createElement("div");
  row.className = "audit-item";
  const time = ev.timestamp ? fmtTime(ev.timestamp) : "";
  row.innerHTML = `
    <span class="audit-time">${esc(time)}</span>
    <span class="pill ${statusPillClass(ev.status)}">${esc(ev.status || "ok")}</span>
    <span>
      <span class="audit-event">${esc(ev.event_type)}</span>
      <span class="audit-meta">${ev.component ? " &middot; " + esc(ev.component) : ""}${ev.message ? " &mdash; " + esc(ev.message) : ""}</span>
    </span>`;
  els.liveList.appendChild(row);
  els.liveList.scrollTop = els.liveList.scrollHeight;
  setRunning(true, `${ev.event_type}${ev.message ? ": " + ev.message : ""}`.slice(0, 60));
}

function closeActiveStream() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
}

async function runTask() {
  const task = els.taskInput.value.trim();
  if (!task) {
    showError("Enter a task before running it.");
    return;
  }
  clearError();
  closeActiveStream();
  setRunning(true, "starting...");
  resetLivePanel();
  els.resultCard.style.display = "none";
  els.pipelineCard.style.display = "none";
  els.auditCard.style.display = "none";

  let created;
  try {
    created = await api("/tasks/stream", {
      method: "POST",
      body: JSON.stringify({ task, confidential: els.confidentialInput.checked }),
    });
  } catch (err) {
    showError(`Task run failed: ${err.message}`);
    setRunning(false);
    return;
  }

  rememberRun(created.execution_id, task);
  activeExecId = created.execution_id;
  await refreshHistoryList();

  const streamUrl = `${apiBase()}/tasks/${encodeURIComponent(created.execution_id)}/events`;
  const es = new EventSource(streamUrl);
  activeEventSource = es;

  es.onmessage = async (msg) => {
    let payload;
    try {
      payload = JSON.parse(msg.data);
    } catch {
      return;
    }
    if (payload.event_type === "stream_end") {
      closeActiveStream();
      els.livePill.className = `pill ${statusPillClass(payload.status)}`;
      els.livePill.textContent = payload.status || "done";
      setRunning(false);
      try {
        await loadExecution(created.execution_id);
      } catch (err) {
        showError(`Run finished but could not load the full result: ${err.message}`);
      }
      await refreshHistoryList();
      return;
    }
    appendLiveEvent(payload);
  };

  es.onerror = () => {
    // The browser's EventSource retries transient drops on its own; if the
    // backend process itself is gone, further retries will just keep
    // failing silently in the background, which is preferable to leaving
    // the "running" button spinner up forever without any error shown.
    if (es.readyState === EventSource.CLOSED) {
      setRunning(false);
      showError("Live event stream closed unexpectedly -- is the backend still running?");
    }
  };
}

// ---------------------------------------------------------------------
// loading & rendering a past (or just-finished) execution
// ---------------------------------------------------------------------

async function loadExecution(execId) {
  clearError();
  activeExecId = execId;
  document.querySelectorAll(".history-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.execId === execId);
  });

  try {
    const [execution, pipeline, audit] = await Promise.all([
      api(`/tasks/${encodeURIComponent(execId)}`),
      api(`/tasks/${encodeURIComponent(execId)}/pipeline`),
      api(`/tasks/${encodeURIComponent(execId)}/audit`),
    ]);
    renderResult(execution);
    renderPipeline(pipeline);
    renderAudit(audit);
  } catch (err) {
    showError(`Could not load execution ${execId}: ${err.message}`);
  }
}

function renderResult(ex) {
  els.resultCard.style.display = "";
  els.resultExecId.textContent = ex.execution_id;

  const banners = [];
  if (ex.error) {
    banners.push(`<div class="banner bad"><strong>Error:</strong> ${esc(ex.error)}</div>`);
  }
  if (ex.final_answer_degraded) {
    banners.push(
      '<div class="banner warn">This answer is <strong>provisional</strong> -- ' +
      "required evidence or requirements were not fully satisfied. See Verification below.</div>"
    );
  }
  els.resultBanners.innerHTML = banners.join("");

  const taskStatus = ex.task_status ?? ex.status;
  els.resultMetrics.innerHTML = `
    <div class="metric"><div class="label">Execution status</div>
      <div class="value"><span class="pill ${statusPillClass(ex.status)}">${esc(ex.status)}</span></div></div>
    <div class="metric"><div class="label">Task status</div>
      <div class="value"><span class="pill ${statusPillClass(taskStatus)}">${esc(taskStatus)}</span></div></div>
    <div class="metric"><div class="label">Duration</div>
      <div class="value" style="font-size:18px;">${fmtMs(ex.duration_ms)}</div></div>
    <div class="metric"><div class="label">Sovereign / confidential</div>
      <div class="value" style="font-size:15px;">${ex.sovereign_mode ? "sovereign" : "open"}${ex.confidential ? " + confidential" : ""}</div></div>
  `;

  els.finalAnswer.textContent = ex.final_answer || "(no final answer was produced)";

  document.getElementById("generatedDocs")?.remove();
  const generated = collectGeneratedDocuments(ex.node_results);
  if (generated.length) {
    els.finalAnswer.insertAdjacentHTML(
      "afterend",
      `<div class="row" style="margin-top:10px;flex-wrap:wrap;gap:8px;" id="generatedDocs">` +
        generated
          .map(
            (d) => `<a class="btn" href="${esc(d.url)}" target="_blank" rel="noopener">
              &#128196; Download ${esc(d.filename)}
            </a>`
          )
          .join("") +
        `</div>`
    );
  }

  const v = ex.final_verification;
  if (v) {
    const issues = (v.issues || [])
      .map((i) => `<li><span class="pill ${statusPillClass(i.severity)}">${esc(i.severity)}</span> ${esc(i.code)}: ${esc(i.message)}</li>`)
      .join("");
    const missingReq = (v.missing_requirements || [])
      .map((m) => `<li>${esc(m.description)} <span class="subtle" style="margin:0;">(${esc(m.reason)})</span></li>`)
      .join("");
    els.verificationBlock.innerHTML = `
      <div class="field"><label>Verification</label></div>
      <div class="row" style="margin:6px 0 10px;">
        <span class="pill ${v.passed ? "ok" : "bad"}">${v.passed ? "passed" : "failed"}</span>
        <span class="pill mute">score ${Number(v.score).toFixed(2)}</span>
        <span class="pill info">${esc(v.recommendation)}</span>
        <span class="pill mute">${esc(v.checker)}</span>
      </div>
      ${issues ? `<ul class="issue-list">${issues}</ul>` : ""}
      ${missingReq ? `<div class="subtle" style="margin-top:8px;">Missing requirements:</div><ul class="issue-list">${missingReq}</ul>` : ""}
    `;
  } else {
    els.verificationBlock.innerHTML = '<div class="subtle" style="margin:0;">No verification result.</div>';
  }
}

function renderPipeline(p) {
  const nodes = p.pipeline?.nodes || [];
  if (nodes.length === 0) {
    els.pipelineCard.style.display = "";
    els.nodeList.innerHTML = '<div class="empty-state">No pipeline recorded for this execution.</div>';
    return;
  }
  els.pipelineCard.style.display = "";
  els.nodeList.innerHTML = nodes.map((n) => renderNode(n)).join("");

  els.nodeList.querySelectorAll(".node-head").forEach((head) => {
    head.addEventListener("click", () => head.closest(".node").classList.toggle("expanded"));
  });
}

function collectGeneratedDocuments(nodeResults) {
  const docs = [];
  for (const result of Object.values(nodeResults || {})) {
    for (const t of result?.tool_results || []) {
      if (t.tool === "generate_word_document" && t.ok && t.output?.filename) {
        docs.push({
          filename: t.output.filename,
          url: `${apiBase()}/files/${encodeURIComponent(t.output.filename)}`,
        });
      }
    }
  }
  return docs;
}

function downloadLinkForToolResult(t) {
  if (t.tool !== "generate_word_document" || !t.ok || !t.output?.filename) return "";
  const url = `${apiBase()}/files/${encodeURIComponent(t.output.filename)}`;
  return `<div class="row" style="margin-top:6px;">
    <a class="btn secondary" href="${esc(url)}" target="_blank" rel="noopener" style="padding:6px 12px;font-size:12px;">
      &#128196; Download ${esc(t.output.filename)}
    </a>
  </div>`;
}

function renderNode(n) {
  const result = n.result;
  const toolResults = result?.tool_results || [];
  const toolsHtml = toolResults
    .map(
      (t) => `<div class="tool-call">
        <div class="tool-head">
          <span class="tool-name">${esc(t.tool)}</span>
          <span class="pill ${t.ok ? "ok" : "bad"}">${t.ok ? "ok" : "failed"}</span>
          ${t.error_kind ? `<span class="pill mute">${esc(t.error_kind)}</span>` : ""}
          ${t.cached ? '<span class="pill info">cached</span>' : ""}
          <span class="subtle" style="margin:0;">${fmtMs(t.duration_ms)}</span>
        </div>
        ${t.error ? `<div style="color:#a3372f;">${esc(t.error)}</div>` : ""}
        <pre>${esc(JSON.stringify({ input: t.input, output: t.output }, null, 2))}</pre>
        ${downloadLinkForToolResult(t)}
      </div>`
    )
    .join("");

  return `<div class="node" data-node-id="${esc(n.id)}">
    <div class="node-head">
      <span class="node-caret">&#9656;</span>
      <span class="node-id">${esc(n.id)}</span>
      <span class="node-name">${esc(n.name || n.description || n.id)}</span>
      <span class="pill mute">${esc(n.type)}</span>
      <span class="pill mute">${esc(n.criticality || "required")}</span>
      <span class="pill ${statusPillClass(n.outcome || n.status)}">${esc(n.outcome || n.status)}</span>
      <span class="node-meta">${esc(n.provider || "—")} / ${esc(n.model || "—")} &middot; ${fmtMs(n.duration_ms)}</span>
    </div>
    <div class="node-body">
      <div class="node-deps">
        agent: <code>${esc(n.agent || "—")}</code>
        &nbsp;&middot;&nbsp; depends on: ${n.depends_on?.length ? n.depends_on.map((d) => `<code>${esc(d)}</code>`).join(", ") : "(none)"}
        &nbsp;&middot;&nbsp; attempts: ${n.attempts ?? 1}
        ${n.tools?.length ? `&nbsp;&middot;&nbsp; tools available: ${n.tools.map((t) => `<code>${esc(t)}</code>`).join(", ")}` : ""}
      </div>
      ${n.description ? `<div class="subtle" style="margin:0 0 8px;">${esc(n.description)}</div>` : ""}
      ${n.error ? `<div class="banner bad">${esc(n.error)}</div>` : ""}
      ${result?.output ? `<div class="answer-text">${esc(result.output)}</div>` : '<div class="subtle" style="margin:0;">(no output)</div>'}
      ${toolsHtml}
    </div>
  </div>`;
}

function renderAudit(a) {
  const events = a.events || [];
  if (events.length === 0) {
    els.auditCard.style.display = "";
    els.auditList.innerHTML = '<div class="empty-state">No audit events recorded.</div>';
    return;
  }
  els.auditCard.style.display = "";
  els.auditList.innerHTML = events
    .map(
      (e) => `<div class="audit-item">
        <span class="audit-time">${fmtTime(e.timestamp)}</span>
        <span class="pill ${statusPillClass(e.status)}">${esc(e.status)}</span>
        <span>
          <span class="audit-event">${esc(e.event_type)}</span>
          <span class="audit-meta"> &middot; ${esc(e.component)}${e.message ? " &mdash; " + esc(e.message) : ""}</span>
        </span>
      </div>`
    )
    .join("");
}

// ---------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------

els.runBtn.addEventListener("click", runTask);
els.refreshBtn.addEventListener("click", async () => {
  await refreshHealth();
  await refreshHistoryList();
});
els.apiBase.addEventListener("change", async () => {
  persistApiBase();
  await refreshHealth();
  await refreshHistoryList();
});
els.taskInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runTask();
});

(async function init() {
  loadPersistedApiBase();
  await refreshHealth();
  await refreshHistoryList();
})();
