// ── Config ──
const MIN_PX = 150;   // ignore images smaller than this
const MAX_PER_PAGE = 6;     // max images to badge per page load
const DELAY_MS = 2500;  // gap between API calls (rate limit safety)

// Labels for the badge — kept short to fit the pill
const BADGE_LABELS = {
    REAL: 'Human Made',
    AI_GENERATED: 'AI Generated',
    UNCERTAIN_LEANING_AI: 'Uncertain',
    UNCERTAIN_LEANING_REAL: 'Uncertain',
    UNCERTAIN_NEUTRAL: 'Uncertain',
};

const COLORS = {
    REAL: '#27ae60',
    AI_GENERATED: '#c0392b',
    UNCERTAIN_LEANING_AI: '#d4a017',
    UNCERTAIN_LEANING_REAL: '#d4a017',
    UNCERTAIN_NEUTRAL: '#d4a017',
};

const analyzed = new Set();
const queue = [];
let running = false;

// ── Badge element ──
function createBadge(prediction, confidence) {
    const badge = document.createElement('div');
    const color = COLORS[prediction] || '#d4a017';
    const label = BADGE_LABELS[prediction] || prediction;
    const pct = (confidence * 100).toFixed(0);

    badge.dataset.aonBadge = 'true';
    badge.textContent = `${label} · ${pct}%`;

    // Inline styles — content scripts can't rely on page CSS
    Object.assign(badge.style, {
        position: 'absolute',
        top: '8px',
        right: '8px',
        zIndex: '2147483647',
        background: color,
        color: '#fff',
        padding: '3px 9px',
        borderRadius: '99px',
        fontFamily: 'ui-monospace, "Cascadia Code", "Source Code Pro", Menlo, monospace',
        fontSize: '10px',
        fontWeight: '700',
        letterSpacing: '0.04em',
        lineHeight: '1.4',
        pointerEvents: 'none',
        boxShadow: '0 2px 8px rgba(0,0,0,0.45)',
        opacity: '0',
        transition: 'opacity 0.35s ease',
        whiteSpace: 'nowrap',
    });

    // Fade in after paint
    requestAnimationFrame(() => {
        requestAnimationFrame(() => { badge.style.opacity = '0.93'; });
    });

    return badge;
}

function attachBadge(img, prediction, confidence) {
    // Skip if already badged
    const parent = img.parentElement;
    if (!parent || parent.querySelector('[data-aon-badge]')) return;

    // Badge needs a positioned ancestor
    const pos = getComputedStyle(parent).position;
    if (pos === 'static') parent.style.position = 'relative';

    parent.appendChild(createBadge(prediction, confidence));
}

// ── Single image analysis ──
async function analyzeImage(img, apiKey, apiUrl) {
    const src = img.currentSrc || img.src;
    if (!src || analyzed.has(src)) return;
    analyzed.add(src);

    try {
        const imgRes = await fetch(src);
        if (!imgRes.ok) return;

        const blob = await imgRes.blob();
        if (!blob.type.startsWith('image/')) return;

        const form = new FormData();
        form.append('file', blob, 'image.jpg');

        const res = await fetch(`${apiUrl}/v1/inference/predict`, {
            method: 'POST',
            headers: { 'X-API-Key': apiKey },
            body: form,
        });

        if (!res.ok) return;

        const data = await res.json();
        attachBadge(img, data.prediction, data.confidence);

    } catch {
        // Silent fail — passive mode should never disrupt the page
    }
}

// ── Queue processor ──
async function processQueue(apiKey, apiUrl) {
    if (running) return;
    running = true;

    while (queue.length > 0) {
        const img = queue.shift();
        await analyzeImage(img, apiKey, apiUrl);
        await sleep(DELAY_MS);
    }

    running = false;
}

// ── Filter ──
function qualifies(img) {
    const src = img.currentSrc || img.src;
    if (!src || src.startsWith('data:')) return false;
    if (analyzed.has(src)) return false;
    const { width, height } = img.getBoundingClientRect();
    return width >= MIN_PX && height >= MIN_PX;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Init ──
async function init() {
    const { passiveBadge, apiKey, apiUrl } =
        await chrome.storage.sync.get(['passiveBadge', 'apiKey', 'apiUrl']);

    if (!passiveBadge || !apiKey || !apiUrl) return;

    try {
        const apiHost = new URL(apiUrl).hostname;
        if (window.location.hostname === apiHost) return;
    } catch {
        // keep on silenty 
    }

    const base = apiUrl.replace(/\/$/, '');

    // Badge images already on the page
    const existing = Array.from(document.querySelectorAll('img'))
        .filter(qualifies)
        .slice(0, MAX_PER_PAGE);

    existing.forEach(img => queue.push(img));
    processQueue(apiKey, base);

    // Watch for dynamically added images
    const observer = new MutationObserver(mutations => {
        for (const mutation of mutations) {
            for (const node of mutation.addedNodes) {
                if (node.nodeName === 'IMG' && qualifies(node)) {
                    if (queue.length < MAX_PER_PAGE) {
                        queue.push(node);
                        processQueue(apiKey, base);
                    }
                }
                if (node.querySelectorAll) {
                    node.querySelectorAll('img').forEach(img => {
                        if (qualifies(img) && queue.length < MAX_PER_PAGE) {
                            queue.push(img);
                            processQueue(apiKey, base);
                        }
                    });
                }
            }
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
}

// Wait for DOM if needed
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}