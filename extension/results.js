const { DEFAULT_API_URL, DEMO_KEY } = AON;

const LABELS = {
  REAL: 'Human Made', AI_GENERATED: 'AI Generated',
  UNCERTAIN_LEANING_AI: 'Uncertain - Leaning AI', UNCERTAIN_LEANING_REAL: 'Uncertain - Leaning Real',
  UNCERTAIN_NEUTRAL: 'Uncertain',
};
const SUBS = {
  REAL: 'Born of mortal hands.', AI_GENERATED: "A machine's cold forgery.",
  UNCERTAIN_LEANING_AI: 'Methinks the lady doth protest too much.',
  UNCERTAIN_LEANING_REAL: 'There are more things in heaven and earth.',
  UNCERTAIN_NEUTRAL: 'Neither here nor there.',
};
const HERALDRY = { REAL: 'Vert', AI_GENERATED: 'Gules', UNCERTAIN_LEANING_AI: 'Or', UNCERTAIN_LEANING_REAL: 'Or', UNCERTAIN_NEUTRAL: 'Or' };
const EPILOGUES = {
  AI_GENERATED: ["What's done cannot be undone.", 'All that glitters is not gold.'],
  REAL: ["All's well that ends well.", 'Truth will come to light.'],
  UNCERTAIN_LEANING_AI: ['The rest is silence.', 'Something is rotten in the state of Denmark.'],
  UNCERTAIN_LEANING_REAL: ['There are more things in heaven and earth.', 'Neither here nor there.'],
  UNCERTAIN_NEUTRAL: ['The rest is silence.', 'What light through yonder window breaks?'],
};

const $ = (id) => document.getElementById(id);
const definite = (p) => p === 'REAL' || p === 'AI_GENERATED';
const cls = (p) => (p === 'REAL' ? 'real' : p === 'AI_GENERATED' ? 'ai' : 'uncertain');
const certainty = (p, c) => (p === 'REAL' ? 1 - c : c);   // API confidence is P(AI)
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const id = new URLSearchParams(location.search).get('id');
const KEY = `analysis:${id}`;
let settled = false;
let waiting = null;
let current = null;           // the result being shown, for feedback
let voted = false;

$('btnClose').addEventListener('click', () => window.close());
$('btnCloseErr').addEventListener('click', () => window.close());
$('btnCancelAccess').addEventListener('click', () => window.close());
$('btnAllowAll').addEventListener('click', () => allowAndRun(requestAllSites()));
$('btnAllowOne').addEventListener('click', () => allowAndRun(requestSite(pending.imageUrl)));
window.addEventListener('pagehide', () => chrome.storage.session.remove(KEY));

/* ── Loading ── */
function updateWaiting() {
  if (!waiting || settled) return;
  const secs = Math.floor((Date.now() - waiting.startedAt) / 1000);
  const base = waiting.mode === 'explain' ? 'The plot thickens...' : 'Consulting the muse...';
  $('loadingText').textContent = secs < 8
    ? base
    : `Waking the server. Free hosting sleeps, so this can take up to a minute. (${secs}s)`;
}
const ticker = setInterval(updateWaiting, 1000);

/* ── State from the background worker ── */
let pending = null;            // the record waiting for site access
let timeoutTimer = null;

function handle(rec) {
  if (!rec || settled) return;
  $('modeTag').textContent = rec.mode === 'explain' ? 'The Mousetrap' : 'Fast';
  if (rec.status === 'needs-access') { showAccess(rec); return; }
  if (rec.status === 'loading') {
    waiting = rec;
    showLoading();
    updateWaiting();
    // The clock only starts once the analysis does, not while you decide on access.
    timeoutTimer ??= setTimeout(() => showError('The analysis timed out. The server may be unavailable.'), 100000);
    return;
  }
  if (rec.status === 'error') showError(rec.error);
  else if (rec.status === 'success') showResult(rec);
}

if (!id) showError('This window was opened without an analysis.');
else {
  chrome.storage.session.get(KEY).then((d) => handle(d[KEY]));
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'session' && changes[KEY]) handle(changes[KEY].newValue);
  });
}

/* ── Site access ── */
function showAccess(rec) {
  pending = rec;
  let host = 'another site';
  try { host = new URL(rec.imageUrl).hostname; } catch { /* keep the generic name */ }
  $('accessMsg').textContent = `This image is on ${host}.`;
  $('btnAllowOne').textContent = `Only images from ${host}`;
  $('view-loading').style.display = 'none';
  $('view-access').style.display = 'flex';
}

function showLoading() {
  $('view-access').style.display = 'none';
  $('view-loading').style.display = 'flex';
}

// `request` is the permission prompt, already opened by the click that called this.
async function allowAndRun(request) {
  if (!(await request)) {
    showError('Access was not allowed, so the image could not be read.');
    return;
  }
  showLoading();
  chrome.runtime.sendMessage({ type: 'aon-run', id });
}

function showError(message) {
  if (settled) return;
  settled = true; clearInterval(ticker); clearTimeout(timeoutTimer);
  $('view-loading').style.display = 'none';
  $('view-access').style.display = 'none';
  $('view-error').style.display = 'flex';
  $('errMsg').textContent = `A tragedy of errors: ${message}`;
}

/* ── Result ── */
function showResult({ imageUrl, filename, result, apiUrl, mode }) {
  settled = true; clearInterval(ticker); clearTimeout(timeoutTimer);
  $('view-loading').style.display = 'none';
  $('view-result').style.display = 'block';

  const { prediction: pred, confidence, processing_time_ms: ms, heatmap_base64: heat, cached } = result;
  current = { prediction: pred, confidence, filename: filename || 'image.jpg' };

  const verdict = $('verdict');
  verdict.textContent = LABELS[pred] || pred;
  verdict.className = `verdict ${cls(pred)}`;
  $('heraldry').textContent = (HERALDRY[pred] || 'Or') +
    (definite(pred) ? ` · ${(certainty(pred, confidence) * 100).toFixed(1)}%` : '');
  $('verdictSub').textContent = SUBS[pred] || '';

  const meta = $('meta');
  meta.textContent = '';
  const parts = [filename || 'image.jpg', null, `${ms} ms`];
  if (cached) parts.push(null, 'Previously, in the Globe...');
  parts[1] = mode === 'explain' ? 'The Mousetrap' : 'Fast';
  const nodes = [el('span', '', parts[0]), el('span', 'serif', parts[1]), el('span', '', parts[2])];
  if (cached) nodes.push(el('span', 'serif', parts[4]));
  nodes.forEach((n, i) => { if (i) meta.append(el('span', 'sep', '·')); meta.append(n); });

  $('confScore').textContent = `${(confidence * 100).toFixed(1)}% machine-made`;
  setTimeout(() => { $('confMarker').style.left = `${confidence * 100}%`; }, 60);

  renderImages(imageUrl, heat);

  const lines = EPILOGUES[pred] || ['The rest is silence.'];
  $('epilogue').textContent = lines[Math.floor(Math.random() * lines.length)];

  const link = $('btnPlayground');
  const base = (apiUrl || DEFAULT_API_URL).replace(/\/$/, '');
  link.href = /^https?:/.test(imageUrl || '') ? `${base}/?imageUrl=${encodeURIComponent(imageUrl)}` : base;

  renderFeedback();
}

// The extension may fetch cross-origin images, so load the original as a blob (no CORS or hotlink trouble).
async function loadOriginal(img, url) {
  try { img.src = URL.createObjectURL(await (await fetch(url)).blob()); }
  catch { img.alt = 'Preview unavailable'; }
}

function renderImages(imageUrl, heatmap) {
  const area = $('imagesArea');
  area.textContent = '';
  if (!imageUrl) return;

  const wrap = el('div', 'images-wrap');
  wrap.append(el('div', 'sec', heatmap ? 'The Mousetrap' : 'Analyzed image'));
  const grid = el('div', heatmap ? 'images-grid' : '');

  const original = el('div', heatmap ? 'img-box' : 'img-box single');
  if (heatmap) original.append(el('span', 'img-label', 'Original'));
  const img = el('img');
  img.alt = 'Original image';
  img.referrerPolicy = 'no-referrer';
  original.append(img);
  loadOriginal(img, imageUrl);
  grid.append(original);

  if (heatmap) {
    const hm = el('div', 'img-box');
    hm.append(el('span', 'img-label heat', 'Illumination'));
    const hmImg = el('img');
    hmImg.alt = 'GradCAM heatmap';
    hmImg.src = `data:image/png;base64,${heatmap}`;
    hm.append(hmImg);
    grid.append(hm);
  }
  wrap.append(grid);
  area.append(wrap);
}

/* ── Feedback ── */
function deriveUserCorrection(prediction, vote) {
  if (vote === 'agree') return definite(prediction) ? prediction : 'UNKNOWN';
  if (prediction === 'AI_GENERATED') return 'REAL';
  if (prediction === 'REAL') return 'AI_GENERATED';
  if (prediction === 'UNCERTAIN_LEANING_AI') return 'REAL';
  if (prediction === 'UNCERTAIN_LEANING_REAL') return 'AI_GENERATED';
  return 'UNKNOWN';
}

function renderFeedback(state) {
  const box = $('feedback');
  box.textContent = '';
  if (state) { box.append(el('div', 'feedback-noted', state === 'failed' ? 'Could not be recorded.' : 'Noted.')); return; }

  const line = el('div', 'feedback-line');
  const up = el('button', 'feedback-btn up', '↑');
  const down = el('button', 'feedback-btn down', '↓');
  up.setAttribute('aria-label', 'Correct verdict'); up.title = 'Correct (C)';
  down.setAttribute('aria-label', 'Wrong verdict'); down.title = 'Wrong (W)';
  up.addEventListener('click', () => submitFeedback('agree'));
  down.addEventListener('click', () => submitFeedback('disagree'));
  line.append(el('span', '', 'Was this right?'), up, down);
  box.append(line);
}

async function submitFeedback(vote) {
  if (!current || voted) return;
  voted = true;
  renderFeedback(vote);                                   // optimistic
  try {
    const { apiKey, apiUrl } = await chrome.storage.sync.get({ apiKey: DEMO_KEY, apiUrl: DEFAULT_API_URL });
    const res = await fetch(`${apiUrl.replace(/\/$/, '')}/v1/feedback/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
      body: JSON.stringify({
        filename: current.filename, model_prediction: current.prediction, confidence: current.confidence,
        user_correction: deriveUserCorrection(current.prediction, vote), client_source: 'extension_v1',
      }),
    });
    if (!res.ok) throw new Error('rejected');
  } catch { renderFeedback('failed'); }
}

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === 'c') submitFeedback('agree');
  if (k === 'w') submitFeedback('disagree');
});