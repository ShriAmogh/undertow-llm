/**
 * gateway-sdk Dashboard — charts.js
 * Comprehensive Frontend Application for LLM Observability & Reliability.
 * Handles tab navigation, real-time polling, Chart.js charts, interactive log filtering,
 * distributed trace waterfall rendering, canary A/B analytics, and alert logging.
 */

// ── Chart.js Global Defaults ─────────────────────────────────────────────────
Chart.defaults.color = '#64748b';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;

const C = {
  cyan:   '#06b6d4',
  green:  '#10b981',
  blue:   '#3b82f6',
  amber:  '#f59e0b',
  red:    '#f43f5e',
  purple: '#8b5cf6',
  indigo: '#6366f1',
  grid:   'rgba(255,255,255,0.04)',
};

// Global Store
let cachedLogs = [];
let cachedTraces = [];
let cachedAlerts = [];
let cachedCanary = [];

// ── Initialize Overview Charts ───────────────────────────────────────────────
const hitRateCtx = document.getElementById('hitRateChart').getContext('2d');
const hitRateChart = new Chart(hitRateCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Hit Rate %',
      data: [],
      borderColor: C.green,
      backgroundColor: 'rgba(16,185,129,0.08)',
      fill: true,
      tension: 0.4,
      pointRadius: 3,
      pointBackgroundColor: C.green,
      borderWidth: 2,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
    scales: {
      x: { grid: { color: C.grid }, ticks: { maxTicksLimit: 8 } },
      y: { grid: { color: C.grid }, min: 0, max: 100, ticks: { callback: v => v + '%' } },
    },
  },
});

const latencyCtx = document.getElementById('latencyChart').getContext('2d');
const latencyChart = new Chart(latencyCtx, {
  type: 'bar',
  data: {
    labels: [],
    datasets: [{
      label: 'Avg Latency (ms)',
      data: [],
      backgroundColor: 'rgba(59,130,246,0.35)',
      borderColor: C.blue,
      borderWidth: 1.5,
      borderRadius: 4,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
    scales: {
      x: { grid: { color: C.grid }, ticks: { maxTicksLimit: 8 } },
      y: { grid: { color: C.grid }, min: 0, ticks: { callback: v => v + ' ms' } },
    },
  },
});

// ── Initialize Canary Comparison Charts ──────────────────────────────────────
const canaryLatCtx = document.getElementById('canaryLatencyChart').getContext('2d');
const canaryLatChart = new Chart(canaryLatCtx, {
  type: 'bar',
  data: { labels: [], datasets: [{ label: 'Avg Latency (ms)', data: [], backgroundColor: [] }] },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { grid: { color: C.grid } }, y: { grid: { color: C.grid }, ticks: { callback: v => v + ' ms' } } },
  },
});

const canaryCostCtx = document.getElementById('canaryCostChart').getContext('2d');
const canaryCostChart = new Chart(canaryCostCtx, {
  type: 'bar',
  data: { labels: [], datasets: [{ label: 'Total Cost (USD)', data: [], backgroundColor: [] }] },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { grid: { color: C.grid } }, y: { grid: { color: C.grid }, ticks: { callback: v => '$' + v } } },
  },
});

// ── Tab Switching Logic ──────────────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

  const targetBtn = document.querySelector(`.nav-btn[data-tab="${tabId}"]`);
  const targetTab = document.getElementById(`tab-${tabId}`);

  if (targetBtn) targetBtn.classList.add('active');
  if (targetTab) targetTab.classList.add('active');
}

// ── Formatters & Helpers ────────────────────────────────────────────────────
function fmtTime(unixTs) {
  if (!unixTs) return '—';
  return new Date(unixTs * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function fmtDateTime(unixTs) {
  if (!unixTs) return '—';
  const d = new Date(unixTs * 1000);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' +
         d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
function fmtLatency(ms) {
  if (ms == null) return '—';
  return ms < 1000 ? ms.toFixed(0) + ' ms' : (ms / 1000).toFixed(2) + ' s';
}
function fmtCost(usd) {
  if (usd == null) return '—';
  return usd === 0 ? '$0.00' : '$' + usd.toFixed(6);
}

// ── Refresh API Calls ────────────────────────────────────────────────────────
async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    const s   = await res.json();

    document.getElementById('totalCalls').textContent    = (s.total_calls || 0).toLocaleString();
    document.getElementById('hitRate').textContent       = (s.hit_rate_pct || 0).toFixed(1) + '%';
    document.getElementById('hitSub').textContent        = `${s.cache_hits || 0} hits / ${s.cache_misses || 0} misses`;
    document.getElementById('avgLatency').textContent    = fmtLatency(s.avg_latency_ms);
    document.getElementById('totalCost').textContent     = '$' + (s.total_cost_usd || 0).toFixed(4);
    
    // Estimate cost saved via semantic caching (~avg cost per miss * hits)
    const avgCostPerMiss = s.cache_misses > 0 ? (s.total_cost_usd / s.cache_misses) : 0.0001;
    const estSavings = (s.cache_hits || 0) * avgCostPerMiss;
    document.getElementById('costSavedSub').textContent = `Saved ~$${estSavings.toFixed(4)} via cache`;

    document.getElementById('errorRate').textContent     = (s.error_rate_pct || 0).toFixed(1) + '%';
    document.getElementById('errorSub').textContent      = `${s.error_count || 0} errors`;
    document.getElementById('cacheEntries').textContent  = (s.cache_entry_count || 0).toLocaleString();
    if (s.backend_label) {
      const cacheSubEl = document.getElementById('cacheSub');
      if (cacheSubEl) cacheSubEl.textContent = s.backend_label;
    }
  } catch (e) {
    console.warn('Stats fetch error:', e);
  }
}

async function refreshCharts() {
  try {
    const res = await fetch('/api/trends?bucket_minutes=5&buckets=24');
    const { trends } = await res.json();

    if (!trends || !trends.length) return;

    const labels    = trends.map(t => fmtTime(t.timestamp));
    const hitRates  = trends.map(t =>
      t.total_calls > 0 ? parseFloat(((t.cache_hits / t.total_calls) * 100).toFixed(1)) : 0
    );
    const latencies = trends.map(t => t.avg_latency_ms || 0);

    hitRateChart.data.labels = labels;
    hitRateChart.data.datasets[0].data = hitRates;
    hitRateChart.update('none');

    latencyChart.data.labels = labels;
    latencyChart.data.datasets[0].data = latencies;
    latencyChart.update('none');
  } catch (e) {
    console.warn('Trends fetch error:', e);
  }
}

async function refreshLogs() {
  try {
    const res = await fetch('/api/logs?limit=100');
    const data = await res.json();
    cachedLogs = data.logs || [];
    filterLogs();
  } catch (e) {
    console.warn('Logs fetch error:', e);
  }
}

function filterLogs() {
  const searchTerm = (document.getElementById('logSearchInput')?.value || '').toLowerCase();
  const statusVal = document.getElementById('statusFilter')?.value || 'all';

  const filtered = cachedLogs.filter(row => {
    const matchesSearch = !searchTerm ||
      (row.prompt || '').toLowerCase().includes(searchTerm) ||
      (row.response || '').toLowerCase().includes(searchTerm) ||
      (row.function_name || '').toLowerCase().includes(searchTerm);

    let matchesStatus = true;
    if (statusVal === 'hit') matchesStatus = row.cache_hit === 1;
    if (statusVal === 'miss') matchesStatus = row.cache_hit === 0 && !row.error;
    if (statusVal === 'error') matchesStatus = !!row.error;

    return matchesSearch && matchesStatus;
  });

  renderLogTables(filtered);
}

function renderLogTables(logs) {
  // 1. Overview Table (Top 6)
  const overviewBody = document.getElementById('overviewLogBody');
  if (overviewBody) {
    if (!logs.length) {
      overviewBody.innerHTML = '<tr><td colspan="7" class="loading">No recent request logs found.</td></tr>';
    } else {
      overviewBody.innerHTML = logs.slice(0, 6).map(row => createLogRowHTML(row, false)).join('');
    }
  }

  // 2. Full Request Logs Table
  const fullBody = document.getElementById('fullLogBody');
  if (fullBody) {
    if (!logs.length) {
      fullBody.innerHTML = '<tr><td colspan="9" class="loading">No request logs matching filters.</td></tr>';
    } else {
      fullBody.innerHTML = logs.map(row => createLogRowHTML(row, true)).join('');
    }
  }
}

function createLogRowHTML(row, isFullTable) {
  const cacheBadge = row.error
    ? '<span class="badge badge--error">ERROR</span>'
    : row.cache_hit
      ? '<span class="badge badge--hit">HIT</span>'
      : '<span class="badge badge--miss">MISS</span>';

  const variantBadge = row.variant
    ? `<span class="badge ${row.variant === 'canary' ? 'badge--canary' : 'badge--primary'}">${row.variant}</span>`
    : '<span style="color:var(--text-muted);font-size:11px">—</span>';

  const latency = row.cache_hit ? '<span style="color:var(--green)">cached</span>' : fmtLatency(row.latency_ms);
  
  const rawPrompt = row.prompt || '';
  const rawResponse = row.response || row.error || '—';

  const promptPreview = rawPrompt.length > 100 ? rawPrompt.substring(0, 100) + '…' : rawPrompt;
  const responsePreview = rawResponse.length > 100 ? rawResponse.substring(0, 100) + '…' : rawResponse;

  const rowJson = JSON.stringify(row).replace(/'/g, "&apos;").replace(/"/g, "&quot;");

  return `<tr>
    <td>${fmtDateTime(row.timestamp)}</td>
    <td class="code-text">${row.function_name}</td>
    <td>${cacheBadge}</td>
    <td style="font-variant-numeric:tabular-nums">${latency}</td>
    <td style="font-family:var(--mono);font-size:11px">${fmtCost(row.cost_estimate)}</td>
    ${isFullTable ? `<td>${variantBadge}</td>` : ''}
    <td class="cell-preview" title="Click Inspect for full view">${promptPreview}</td>
    ${isFullTable ? `<td class="cell-preview" title="Click Inspect for full view">${responsePreview}</td>` : ''}
    <td>
      <button class="btn-secondary" onclick='openModal(${rowJson})'>Inspect</button>
    </td>
  </tr>`;
}

// ── Distributed Traces ───────────────────────────────────────────────────────
async function refreshTraces() {
  try {
    const res = await fetch('/api/traces?limit=50');
    const data = await res.json();
    cachedTraces = data.traces || [];

    const badge = document.getElementById('traceBadge');
    if (badge) badge.textContent = cachedTraces.length;

    renderTraces();
  } catch (e) {
    console.warn('Traces fetch error:', e);
  }
}

function renderTraces() {
  const container = document.getElementById('tracesListContainer');
  if (!container) return;

  const searchTerm = (document.getElementById('traceSearchInput')?.value || '').toLowerCase();
  const filtered = cachedTraces.filter(t => 
    !searchTerm ||
    t.trace_id.toLowerCase().includes(searchTerm) ||
    t.functions.some(f => f.toLowerCase().includes(searchTerm))
  );

  if (!filtered.length) {
    container.innerHTML = '<div class="loading">No distributed traces found.</div>';
    return;
  }

  container.innerHTML = filtered.map(trace => {
    const totalLatency = trace.spans.reduce((acc, s) => acc + (s.latency_ms || 0), 0);
    const maxSpanLatency = Math.max(...trace.spans.map(s => s.latency_ms || 1), 1);

    const spansHTML = trace.spans.map((span, idx) => {
      const barWidthPct = Math.max(5, Math.min(100, ((span.latency_ms || 1) / maxSpanLatency) * 100));
      const spanJson = JSON.stringify(span).replace(/'/g, "&apos;").replace(/"/g, "&quot;");

      const spanPrompt = span.prompt || '';
      const spanResponse = span.response || span.error || '—';
      const promptPrev = spanPrompt.length > 100 ? spanPrompt.substring(0, 100) + '…' : spanPrompt;
      const respPrev = spanResponse.length > 100 ? spanResponse.substring(0, 100) + '…' : spanResponse;

      return `<div class="span-row">
        <div class="span-row-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:11px; font-weight:700; color:var(--text-muted)">Span #${idx + 1}</span>
            <span class="span-fn">${span.function_name}</span>
            ${span.cache_hit ? '<span class="badge badge--hit">HIT</span>' : ''}
            ${span.error ? '<span class="badge badge--error">ERROR</span>' : ''}
          </div>
          <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:11px; font-family:var(--mono); color:var(--text-dim)">${fmtLatency(span.latency_ms)}</span>
            <button class="btn-secondary" style="padding:2px 8px; font-size:10.5px" onclick='openModal(${spanJson})'>Inspect</button>
          </div>
        </div>
        <div class="span-waterfall-track">
          <div class="span-waterfall-bar" style="width: ${barWidthPct}%"></div>
        </div>
        <div class="cell-preview" style="font-size:11px; color:var(--text-dim); margin-top:4px;">
          <strong>Prompt:</strong> ${promptPrev}
        </div>
        <div class="cell-preview" style="font-size:11px; color:var(--text); margin-top:2px;">
          <strong>Response:</strong> ${respPrev}
        </div>
      </div>`;
    }).join('');

    return `<div class="trace-card">
      <div class="trace-header">
        <div class="trace-id-group">
          <span style="font-size:14px">🔀</span>
          <span class="trace-id">${trace.trace_id}</span>
          <span class="badge badge--ok">${trace.span_count} Spans</span>
        </div>
        <div style="display:flex; align-items:center; gap:14px">
          <span style="font-size:11.5px; color:var(--text-muted)">First call: ${fmtDateTime(trace.first_call_ts)}</span>
          <span style="font-size:12px; font-weight:700; color:var(--cyan)">Total: ${fmtLatency(totalLatency)}</span>
        </div>
      </div>
      <div class="trace-spans-body">
        ${spansHTML}
      </div>
    </div>`;
  }).join('');
}

// ── Canary Analytics ─────────────────────────────────────────────────────────
async function refreshCanary() {
  try {
    const res = await fetch('/api/canary');
    const data = await res.json();
    cachedCanary = data.variants || [];

    renderCanary();
  } catch (e) {
    console.warn('Canary fetch error:', e);
  }
}

function renderCanary() {
  const grid = document.getElementById('canaryVariantCards');
  if (!grid) return;

  if (!cachedCanary.length) {
    grid.innerHTML = '<div class="loading">No canary data logged yet. Execute canary functions with canary={"fn": ...} tag.</div>';
    return;
  }

  // Variant Cards
  grid.innerHTML = cachedCanary.map(v => {
    const name = v.variant || 'Untagged';
    const isCanary = name === 'canary';
    const isPrimary = name === 'primary';
    const badgeClass = isCanary ? 'badge--canary' : isPrimary ? 'badge--primary' : 'badge--ok';

    return `<div class="variant-card">
      <div class="variant-card-header">
        <span class="variant-name">${name} Variant</span>
        <span class="badge ${badgeClass}">${name}</span>
      </div>
      <div class="variant-metrics">
        <div class="variant-metric-box">
          <div class="v-label">Total Calls</div>
          <div class="v-val">${(v.total_calls || 0).toLocaleString()}</div>
        </div>
        <div class="variant-metric-box">
          <div class="v-label">Avg Latency</div>
          <div class="v-val" style="color:var(--cyan)">${fmtLatency(v.avg_latency_ms)}</div>
        </div>
        <div class="variant-metric-box">
          <div class="v-label">Total Cost</div>
          <div class="v-val" style="color:var(--amber)">${fmtCost(v.total_cost_usd)}</div>
        </div>
        <div class="variant-metric-box">
          <div class="v-label">Cache Hits</div>
          <div class="v-val" style="color:var(--green)">${v.cache_hits || 0}</div>
        </div>
      </div>
    </div>`;
  }).join('');

  // Charts
  const labels = cachedCanary.map(v => (v.variant || 'Untagged').toUpperCase());
  const latencies = cachedCanary.map(v => v.avg_latency_ms || 0);
  const costs = cachedCanary.map(v => parseFloat((v.total_cost_usd || 0).toFixed(6)));
  const colors = cachedCanary.map(v => v.variant === 'canary' ? C.purple : v.variant === 'primary' ? C.amber : C.blue);

  canaryLatChart.data.labels = labels;
  canaryLatChart.data.datasets[0].data = latencies;
  canaryLatChart.data.datasets[0].backgroundColor = colors;
  canaryLatChart.update('none');

  canaryCostChart.data.labels = labels;
  canaryCostChart.data.datasets[0].data = costs;
  canaryCostChart.data.datasets[0].backgroundColor = colors;
  canaryCostChart.update('none');
}

// ── Modal Payload Inspector ──────────────────────────────────────────────────

function openModal(row) {
  document.getElementById('modalFunctionName').textContent = row.function_name || 'LLM Function';
  document.getElementById('modalTimestamp').textContent = fmtDateTime(row.timestamp);
  
  const cacheStatusEl = document.getElementById('modalCacheStatus');
  if (row.error) {
    cacheStatusEl.innerHTML = '<span class="badge badge--error">ERROR</span>';
  } else if (row.cache_hit) {
    cacheStatusEl.innerHTML = '<span class="badge badge--hit">HIT</span>';
  } else {
    cacheStatusEl.innerHTML = '<span class="badge badge--miss">MISS</span>';
  }

  document.getElementById('modalLatency').textContent = row.cache_hit ? 'cached' : fmtLatency(row.latency_ms);
  document.getElementById('modalCost').textContent = fmtCost(row.cost_estimate);
  document.getElementById('modalVariant').textContent = row.variant || 'Untagged';
  document.getElementById('modalTraceId').textContent = row.trace_id || 'N/A';
  document.getElementById('modalSpanId').textContent = row.span_id || 'N/A';

  document.getElementById('modalPromptText').textContent = row.prompt || '(Empty prompt)';
  document.getElementById('modalResponseText').textContent = row.response || '(Empty response)';

  const errBlock = document.getElementById('modalErrorContainer');
  const errText = document.getElementById('modalErrorText');
  if (row.error) {
    errBlock.style.display = 'block';
    errText.textContent = row.error;
  } else {
    errBlock.style.display = 'none';
  }

  document.getElementById('payloadModal').classList.add('active');
}

function closeModal(event) {
  if (event && event.target !== document.getElementById('payloadModal') && !event.target.classList.contains('modal-close')) {
    return;
  }
  document.getElementById('payloadModal').classList.remove('active');
}

function copyModalText(elementId) {
  const text = document.getElementById(elementId).textContent;
  navigator.clipboard.writeText(text).then(() => {
    alert('Copied payload to clipboard!');
  }).catch(e => console.error('Copy error:', e));
}

// ── Main Loop ────────────────────────────────────────────────────────────────
function updateTimestamp() {
  const el = document.getElementById('lastRefresh');
  if (el) el.textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

async function refreshAll() {
  await Promise.all([
    refreshStats(),
    refreshCharts(),
    refreshLogs(),
    refreshTraces(),
    refreshCanary(),
  ]);
  updateTimestamp();
}

// Initial load + poll every 5s
refreshAll();
setInterval(refreshAll, 5000);
