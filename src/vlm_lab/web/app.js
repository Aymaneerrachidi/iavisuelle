const API = "/api/v1";
const state = { models: [], runs: [], selectedRun: null, poller: null };

class ApiError extends Error {
  constructor(message, status, code) { super(message); this.status = status; this.code = code; }
}

async function api(path, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API}${path}`, { ...options, signal: controller.signal });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(body.error || `Erreur HTTP ${response.status}`, response.status, body.code);
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new ApiError("La requête a dépassé le délai autorisé.", 408, "TIMEOUT");
    throw error;
  } finally { clearTimeout(timer); }
}

const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const fmt = (value, unit = "") => value == null ? "—" : `${Number(value).toLocaleString("fr-FR", {maximumFractionDigits: 2})}${unit}`;

async function bootstrap() {
  try {
    await api("/health"); document.querySelector("#health").classList.add("online");
    state.models = await api("/models"); renderModelControls(); await loadRuns();
  } catch (error) { document.querySelector("#health").title = error.message; }
}

function renderModelControls() {
  document.querySelector("#caption-model").innerHTML = state.models.map(m => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`).join("");
  document.querySelector("#model-checks").innerHTML = state.models.map((m, i) => `<label><input type="checkbox" name="model_ids" value="${escapeHtml(m.id)}" ${i < 2 ? "checked" : ""}/><span>${escapeHtml(m.label)}<small>${m.parameters_millions} M params</small></span></label>`).join("");
}

const imageInput = document.querySelector("#image-input");
const dropzone = document.querySelector("#dropzone");
imageInput.addEventListener("change", previewImage);
["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", event => { if (event.dataTransfer.files.length) { imageInput.files = event.dataTransfer.files; previewImage(); } });
function previewImage() { const file = imageInput.files[0]; if (!file) return; const preview = document.querySelector("#preview"); preview.src = URL.createObjectURL(file); preview.hidden = false; document.querySelector("#drop-copy").hidden = true; }

document.querySelector("#caption-form").addEventListener("submit", async event => {
  event.preventDefault(); const form = event.currentTarget; const button = form.querySelector("button"); const errorBox = document.querySelector("#caption-error");
  errorBox.textContent = ""; button.disabled = true; button.textContent = "Analyse en cours…";
  try {
    const result = await api("/caption", { method: "POST", body: new FormData(form) }, 120000);
    document.querySelector("#caption-empty").hidden = true; document.querySelector("#caption-result").hidden = false;
    document.querySelector("#caption-text").textContent = result.caption; document.querySelector("#stat-latency").textContent = fmt(result.latency_ms, " ms");
    document.querySelector("#stat-vram").textContent = fmt(result.peak_vram_mb, " Mo"); document.querySelector("#stat-memory").textContent = fmt(result.process_memory_mb, " Mo");
  } catch (error) { errorBox.textContent = error.message; }
  finally { button.disabled = false; button.textContent = "Générer la légende"; }
});

document.querySelector("#experiment-form").addEventListener("submit", async event => {
  event.preventDefault(); const form = event.currentTarget; const button = form.querySelector("button"); const errorBox = document.querySelector("#experiment-error");
  const values = new FormData(form); const payload = {
    name: values.get("name"), dataset_manifest: values.get("dataset_manifest"),
    model_ids: values.getAll("model_ids"), precisions: values.getAll("precisions"), sample_limit: Number(values.get("sample_limit")),
    options: { max_new_tokens: Number(values.get("max_new_tokens")), num_beams: Number(values.get("num_beams")), temperature: 1 }
  };
  if (!payload.model_ids.length || !payload.precisions.length) { errorBox.textContent = "Sélectionnez au moins un modèle et une précision."; return; }
  button.disabled = true; errorBox.textContent = "";
  try { const run = await api("/experiments", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)}); await loadRuns(); selectRun(run.id); document.querySelector("#runs").scrollIntoView(); }
  catch(error) { errorBox.textContent = error.message; }
  finally { button.disabled = false; }
});

async function loadRuns() {
  try { state.runs = await api("/experiments"); renderRuns(); if (state.selectedRun) await selectRun(state.selectedRun, false); }
  catch(error) { document.querySelector("#run-list").innerHTML = `<p class="error-box">${escapeHtml(error.message)}</p>`; }
}

function renderRuns() {
  const list = document.querySelector("#run-list"); if (!state.runs.length) { list.innerHTML = '<p class="empty-list">Aucune expérience. Lancez votre première matrice ci-dessus.</p>'; return; }
  list.innerHTML = state.runs.map(run => { const ratio = run.progress_total ? 100 * run.progress_current / run.progress_total : 0; return `<button class="run-card ${state.selectedRun===run.id?"active":""}" data-id="${run.id}"><header><b>${escapeHtml(run.name)}</b><span class="status ${run.status}">${escapeHtml(run.status)}</span></header><p>${run.progress_current} / ${run.progress_total} inférences · ${new Date(run.created_at).toLocaleString("fr-FR")}</p><div class="progress"><i style="width:${ratio}%"></i></div></button>`; }).join("");
  list.querySelectorAll(".run-card").forEach(card => card.addEventListener("click", () => selectRun(card.dataset.id)));
}

async function selectRun(id, rerender = true) {
  state.selectedRun = id; if (rerender) renderRuns(); const detail = document.querySelector("#run-detail"); detail.innerHTML = '<p class="panel-label">Chargement…</p>';
  try {
    const run = await api(`/experiments/${id}`); const [metrics, predictions] = await Promise.all([api(`/experiments/${id}/metrics`), api(`/experiments/${id}/predictions?limit=6`)]);
    const running = ["queued","running"].includes(run.status); const ratio = run.progress_total ? Math.round(100*run.progress_current/run.progress_total) : 0;
    detail.innerHTML = `<p class="panel-label">Détail d’expérience</p><div class="detail-head"><div><h3>${escapeHtml(run.name)}</h3><p><span class="status ${run.status}">${escapeHtml(run.status)}</span> · ${ratio}%</p></div><div class="detail-actions">${running?'<button id="cancel-run" class="icon-button">Annuler</button>':''}<a class="icon-button" href="${API}/experiments/${id}/export.csv">CSV ↓</a></div></div>${run.error?`<p class="error-box">${escapeHtml(run.error)}</p>`:""}${renderMetrics(metrics)}${renderPredictions(predictions.items)}`;
    const cancel = detail.querySelector("#cancel-run"); if (cancel) cancel.addEventListener("click", async () => { try { await api(`/experiments/${id}/cancel`, {method:"POST"}); await loadRuns(); } catch(error) { alert(error.message); } });
    if (running) startPolling(); else stopPolling();
  } catch(error) { detail.innerHTML = `<p class="error-box">${escapeHtml(error.message)}</p>`; }
}

function renderMetrics(metrics) {
  if (!metrics.length) return '<div class="empty-state compact"><p>Les métriques apparaîtront dès la première prédiction.</p></div>';
  return `<table class="metrics-table"><thead><tr><th>Configuration</th><th>BLEU-4</th><th>ROUGE-L</th><th>Latence</th><th>VRAM</th></tr></thead><tbody>${metrics.map(m=>`<tr><td>${escapeHtml(m.model_id)} / ${escapeHtml(m.precision)}</td><td>${fmt(m.bleu4)}</td><td>${fmt(m.rouge_l)}</td><td>${fmt(m.mean_latency_ms," ms")}</td><td>${fmt(m.peak_vram_mb," Mo")}</td></tr>`).join("")}</tbody></table>`;
}
function renderPredictions(items) { if (!items.length) return ""; return `<div class="prediction-list">${items.map(item=>`<article class="prediction"><small>${escapeHtml(item.image_id)} · ${escapeHtml(item.model_id)} / ${escapeHtml(item.precision)}</small><p>${escapeHtml(item.caption)}</p></article>`).join("")}</div>`; }
function startPolling(){ if(!state.poller) state.poller=setInterval(loadRuns,3000); } function stopPolling(){ clearInterval(state.poller);state.poller=null; }
document.querySelector("#refresh-runs").addEventListener("click", loadRuns);
bootstrap();

