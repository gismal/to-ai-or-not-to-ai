const LABELS = {
    REAL: 'Human Made',
    AI_GENERATED: 'AI Generated',
    UNCERTAIN_LEANING_AI: 'Uncertain, Leaning AI',
    UNCERTAIN_LEANING_REAL: 'Uncertain, Leaning Real',
    UNCERTAIN_NEUTRAL: 'Uncertain',
};

const SHAKESPEARE_SUBS = {
    REAL: "Born of mortal hands.",
    AI_GENERATED: "A machine's cold forgery.",
    UNCERTAIN_LEANING_AI: "Methinks the lady doth protest too much.",
    UNCERTAIN_LEANING_REAL: "There are more things in heaven and earth.",
    UNCERTAIN_NEUTRAL: "Neither here nor there.",
};

const HERALDRY = {
    REAL: 'Vert',
    AI_GENERATED: 'Gules',
    UNCERTAIN_LEANING_AI: 'Or',
    UNCERTAIN_LEANING_REAL: 'Or',
    UNCERTAIN_NEUTRAL: 'Or',
};

const EPILOGUES = {
    AI_GENERATED: ["What's done cannot be undone.", "All that glitters is not gold."],
    REAL: ["All's well that ends well.", "Truth will come to light."],
    UNCERTAIN_LEANING_AI: ["The rest is silence.", "Something is rotten in the state of Denmark."],
    UNCERTAIN_LEANING_REAL: ["There are more things in heaven and earth.", "Neither here nor there."],
    UNCERTAIN_NEUTRAL: ["The rest is silence.", "What light through yonder window breaks?"],
};

function vcls(p) {
    if (!p) return '';
    if (p === 'REAL') return 'real';
    if (p === 'AI_GENERATED') return 'ai';
    return 'uncertain';
}

function getEpilogue(prediction) {
    const lines = EPILOGUES[prediction] || ['The rest is silence.'];
    return lines[Math.floor(Math.random() * lines.length)];
}

function $(id) { return document.getElementById(id); }

// ── Poll chrome.storage.session until result arrives ──
const POLL_MS = 400;
const TIMEOUT_MS = 70_000;
const startedAt = Date.now();

const timer = setInterval(async () => {
    if (Date.now() - startedAt > TIMEOUT_MS) {
        clearInterval(timer);
        showError('The analysis timed out. The API may be unavailable.');
        return;
    }

    const data = await chrome.storage.session.get('pendingAnalysis');
    const analysis = data.pendingAnalysis;
    if (!analysis) return;

    // Update loading text while waiting
    if (analysis.status === 'loading') {
        const secs = Math.floor((Date.now() - (analysis.startedAt || startedAt)) / 1000);
        let msg = '';

        if (secs < 3) {
            msg = analysis.mode === 'explain' ? 'The plot thickens...' : 'Consulting the muse...';
        } else if (secs < 12) {
            msg = analysis.mode === 'explain' ? `The Mousetrap is set... (${secs}s)` : `Deliberating... (${secs}s)`;
        } else {
            // RENDER COLD START WARNING
            msg = `Waking up the server from sleep (Render free tier). This may take up to 50 seconds... (${secs}s)`;
        }

        $('loadingText').textContent = msg;
        $('modeTag').textContent = analysis.mode === 'explain' ? 'The Mousetrap' : 'Fast';
        return;
    }

    clearInterval(timer);

    if (analysis.status === 'error') { showError(analysis.error); return; }
    if (analysis.status === 'success') { showResult(analysis); return; }

}, POLL_MS);

// ── Error view ──
function showError(message) {
    $('view-loading').style.display = 'none';
    const v = $('view-error');
    v.style.display = 'flex';
    $('errMsg').textContent = `A tragedy of errors: ${message}`;
}

// ── Result view ──
function showResult({ imageUrl, filename, result, apiUrl, mode }) {
    $('view-loading').style.display = 'none';
    $('view-result').style.display = 'block';

    const { prediction, confidence, processing_time_ms, heatmap_base64, cached } = result;
    const cls = vcls(prediction);

    // Topbar mode tag
    $('modeTag').textContent = mode === 'explain' ? 'The Mousetrap' : 'Fast';

    // Verdict
    const verdictEl = $('verdict');
    verdictEl.textContent = LABELS[prediction] || prediction;
    verdictEl.className = `broadside-verdict ${cls}`;

    // Heraldry line
    $('heraldry').textContent =
        `${HERALDRY[prediction] || 'Or'} · ${(confidence * 100).toFixed(1)}%`;

    // Shakespeare sub
    $('verdictSub').textContent = SHAKESPEARE_SUBS[prediction] || '';

    // Meta row
    const cachedFragment = cached
        ? `<span class="sep">·</span><span class="serif-detail">Previously, in the Globe...</span>`
        : '';
    $('meta').innerHTML =
        `<span>${filename || 'image.jpg'}</span>` +
        `<span class="sep">·</span>` +
        `<span>${processing_time_ms} ms</span>` +
        cachedFragment;

    // Confidence bar
    $('confScore').textContent = `${(confidence * 100).toFixed(1)}%`;
    const fill = $('confFill');
    fill.className = `conf-fill ${cls}`;
    setTimeout(() => { fill.style.width = `${confidence * 100}%`; }, 60);

    // Images
    const area = $('imagesArea');
    if (heatmap_base64 && imageUrl) {
        area.innerHTML = `
      <div class="images-wrap">
        <div class="sec">The Mousetrap</div>
        <div class="images-grid">
          <div class="img-box">
            <span class="img-box-label">Original</span>
            <img src="${imageUrl}" alt="Original" crossorigin="anonymous" />
          </div>
          <div class="img-box">
            <span class="img-box-label" style="color:var(--uncertain)">Illumination</span>
            <img src="data:image/png;base64,${heatmap_base64}" alt="GradCAM heatmap" />
          </div>
        </div>
      </div>
    `;
    } else if (imageUrl) {
        area.innerHTML = `
      <div class="images-wrap">
        <div class="sec">Analyzed image</div>
        <div class="img-single">
          <img src="${imageUrl}" alt="Analyzed image" crossorigin="anonymous" />
        </div>
      </div>
    `;
    }

    // Epilogue
    $('epilogue').textContent = getEpilogue(prediction);

    // Playground deep-link
    const base = (apiUrl || 'https://to-ai-or-not-to-ai.onrender.com').replace(/\/$/, '');
    $('btnPlayground').href = imageUrl
        ? `${base}/?imageUrl=${encodeURIComponent(imageUrl)}`
        : base;
}