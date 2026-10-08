(() => {
    const { DEFAULT_API_URL } = AON;
    const MIN_PX = 150;          // ignore small images
    const MAX_PER_PAGE = 6;      // keep passive mode gentle
    const DELAY_MS = 2500;       // gap between requests

    const COLORS = { REAL: '#27ae60', AI_GENERATED: '#c0392b' };
    const UNCERTAIN_COLOR = '#d4a017';
    const queue = [];
    const seen = new Set();
    let running = false;

    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    // The API returns P(AI). For Human Made the verdict's certainty is 1 - P(AI).
    function badgeText(prediction, confidence) {
        if (prediction === 'REAL') return `Human Made · ${Math.round((1 - confidence) * 100)}%`;
        if (prediction === 'AI_GENERATED') return `AI Generated · ${Math.round(confidence * 100)}%`;
        return 'Uncertain';
    }

    function attachBadge(img, prediction, confidence) {
        const parent = img.parentElement;
        if (!parent || parent.querySelector('[data-aon-badge]')) return;
        if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';

        const badge = document.createElement('div');
        badge.dataset.aonBadge = 'true';
        badge.textContent = badgeText(prediction, confidence);
        Object.assign(badge.style, {
            position: 'absolute', top: '8px', right: '8px', zIndex: '2147483647',
            background: COLORS[prediction] || UNCERTAIN_COLOR, color: '#fff',
            padding: '3px 9px', borderRadius: '99px', pointerEvents: 'none', whiteSpace: 'nowrap',
            font: '700 10px/1.4 ui-monospace, "Cascadia Code", Menlo, monospace', letterSpacing: '0.04em',
            boxShadow: '0 2px 8px rgba(0,0,0,0.45)', opacity: '0', transition: 'opacity 0.35s ease',
        });
        parent.appendChild(badge);
        requestAnimationFrame(() => requestAnimationFrame(() => { badge.style.opacity = '0.93'; }));
    }

    async function processQueue() {
        if (running) return;
        running = true;
        while (queue.length) {
            const img = queue.shift();
            try {
                const reply = await chrome.runtime.sendMessage({ type: 'aon-predict', src: img.currentSrc || img.src });
                if (reply?.ok) attachBadge(img, reply.result.prediction, reply.result.confidence);
            } catch { /* extension reloaded or page closing: stay silent */ }
            await sleep(DELAY_MS);
        }
        running = false;
    }

    function consider(img) {
        const src = img.currentSrc || img.src;
        if (seen.size >= MAX_PER_PAGE || !/^https?:/.test(src) || seen.has(src)) return;
        const { width, height } = img.getBoundingClientRect();
        if (width < MIN_PX || height < MIN_PX) return;
        seen.add(src);
        queue.push(img);
        processQueue();
    }

    async function init() {
        const { passiveBadge, apiUrl } = await chrome.storage.sync.get({ passiveBadge: false, apiUrl: DEFAULT_API_URL });
        if (!passiveBadge) return;
        try { if (location.origin === new URL(apiUrl).origin) return; } catch { /* malformed URL: carry on */ }

        document.querySelectorAll('img').forEach(consider);
        new MutationObserver((mutations) => {
            for (const m of mutations) for (const node of m.addedNodes) {
                if (node.nodeName === 'IMG') consider(node);
                else if (node.querySelectorAll) node.querySelectorAll('img').forEach(consider);
            }
        }).observe(document.body, { childList: true, subtree: true });
    }

    init();
})();