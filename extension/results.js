/**
 * results.js — Analysis results window
 *
 * Polls chrome.storage.session every 400ms until background.js
 * has stored either a success or error result, then renders it.
 *
 * Why polling instead of chrome.storage.onChanged?
 * onChanged fires when the value changes. If this window opens
 * after the result is already stored, the event never fires.
 * Polling handles both cases cleanly.
 */

const POLL_INTERVAL = 400;   // ms between checks
const POLL_TIMEOUT = 60000; // give up after 60s
const startedAt = Date.now();

const LABEL_MAP = {
    REAL: 'Human Made',
    AI_GENERATED: 'AI Generated',
    UNCERTAIN_LEANING_AI: 'Uncertain — Leaning AI',
    UNCERTAIN_LEANING_REAL: 'Uncertain — Leaning Real',
    UNCERTAIN_NEUTRAL: 'Uncertain — No Verdict',
};

function verdictClass(prediction) {
    if (!prediction) return '';
    if (prediction.includes('REAL')) return 'real';
    if (prediction.includes('AI_GENERATED')) return 'ai';
    return 'uncertain';
}

// ── Poll for results ─────────────────────────────────────────────────────────

const pollTimer = setInterval(async () => {
    // Timeout guard
    if (Date.now() - startedAt > POLL_TIMEOUT) {
        clearInterval(pollTimer);
        showError('Analysis timed out. The API may be unavailable.');
        return;
    }

    const data = await chrome.storage.session.get('pendingAnalysis');
    const analysis = data.pendingAnalysis;

    if (!analysis) return; // Not set yet — keep waiting

    if (analysis.status === 'loading') {
        // Update loading message with elapsed time
        const secs = Math.floor((Date.now() - (analysis.startedAt || startedAt)) / 1000);
        document.getElementById('loading-msg').textContent =
            secs < 3 ? 'Fetching image…' :
                secs < 10 ? 'Running analysis…' :
                    `Still working… (${secs}s)`;
        return;
    }

    clearInterval(pollTimer);

    if (analysis.status === 'error') { showError(analysis.error); return; }
    if (analysis.status === 'success') { showResult(analysis); return; }

}, POLL_INTERVAL);

// ── Render: error ────────────────────────────────────────────────────────────

function showError(message) {
    document.getElementById('view-loading').style.display = 'none';
    const errView = document.getElementById('view-error');
    errView.style.display = 'flex';
    document.getElementById('error-msg').textContent = message;
}

// ── Render: success ──────────────────────────────────────────────────────────

function showResult({ imageUrl, filename, result, apiUrl }) {
    document.getElementById('view-loading').style.display = 'none';
    const resView = document.getElementById('view-results');
    resView.style.display = 'flex';

    const { prediction, confidence, processing_time_ms, heatmap_base64, cached } = result;
    const cls = verdictClass(prediction);

    // Verdict label
    const label = document.getElementById('verdict-label');
    label.textContent = LABEL_MAP[prediction] || prediction;
    label.className = `verdict-label ${cls}`;

    // Cached tag
    if (cached) document.getElementById('cached-tag').style.display = 'inline-block';

    // Filename + timing
    document.getElementById('filename').textContent = filename || result.filename || '';
    document.getElementById('timing').textContent = `${processing_time_ms} ms`;

    // Confidence bar — delayed so CSS transition plays
    document.getElementById('conf-score').textContent = `${(confidence * 100).toFixed(1)}%`;
    const fill = document.getElementById('bar-fill');
    fill.className = `bar-fill ${cls}`;
    setTimeout(() => { fill.style.width = `${confidence * 100}%`; }, 50);

    // Images
    const container = document.getElementById('images-container');

    if (heatmap_base64 && imageUrl) {
        // Side-by-side: original + heatmap
        container.innerHTML = `
      <div class="images-row">
        <div class="img-box">
          <div class="img-label">Original</div>
          <img src="${imageUrl}" alt="Original" crossorigin="anonymous" />
        </div>
        <div class="img-box">
          <div class="img-label" style="color: var(--ai)">GradCAM</div>
          <img src="data:image/png;base64,${heatmap_base64}" alt="Heatmap" />
        </div>
      </div>
    `;
    } else if (imageUrl) {
        // Fast predict — just the original
        container.innerHTML = `
      <div class="image-single">
        <img src="${imageUrl}" alt="Analyzed image" crossorigin="anonymous" />
      </div>
    `;
    }

    // Playground deep-link
    // Passes the image URL so the playground can auto-load it
    const playgroundUrl = apiUrl
        ? `${apiUrl}/?imageUrl=${encodeURIComponent(imageUrl)}`
        : `http://localhost:8000/?imageUrl=${encodeURIComponent(imageUrl)}`;

    document.getElementById('btn-playground').href = playgroundUrl;
}