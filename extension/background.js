/**
 * background.js — Service Worker
 *
 * Responsibilities:
 * 1. Register the right-click context menu on install
 * 2. On click: store loading state → open results window → fetch image → call API → store result
 *
 * Why a service worker?
 * Manifest V3 replaced persistent background pages with service workers.
 * They spin up on demand and terminate when idle — no persistent state in memory.
 * That's why we use chrome.storage.session for passing data to results.html.
 */

const MENU_ID = 'ai-detector-analyze';

// ── Register context menu ────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: MENU_ID,
        title: '🔍 Analyze with AI Detector',
        contexts: ['image'],  // only appears on right-clicking an image
    });
});

// ── Handle right-click ───────────────────────────────────────────────────────

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId !== MENU_ID) return;

    const imageUrl = info.srcUrl;
    if (!imageUrl) return;

    // 1. Store "loading" state immediately
    await chrome.storage.session.set({
        pendingAnalysis: {
            status: 'loading',
            imageUrl,
            filename: extractFilename(imageUrl),
            startedAt: Date.now(),
        },
    });

    // 2. Open results window right away — user sees a spinner while we work
    chrome.windows.create({
        url: chrome.runtime.getURL('results.html'),
        type: 'popup',
        width: 440,
        height: 620,
        focused: true,
    });

    // 3. Run analysis in the background (results window polls for completion)
    await runAnalysis(imageUrl);
});

// ── Analysis pipeline ────────────────────────────────────────────────────────

async function runAnalysis(imageUrl) {
    try {
        // Load user settings
        const {
            apiKey = '',
            apiUrl = 'http://localhost:8000',
            mode = 'explain',
        } = await chrome.storage.sync.get(['apiKey', 'apiUrl', 'mode']);

        if (!apiKey) throw new Error('No API key set. Click the extension icon to add one.');

        const base = apiUrl.replace(/\/$/, '');
        const endpoint = mode === 'explain'
            ? '/v1/inference/explain'
            : '/v1/inference/predict';

        // Fetch the image bytes from the source URL
        // Service workers can fetch cross-origin URLs that are publicly accessible
        const imgRes = await fetch(imageUrl);
        if (!imgRes.ok) {
            throw new Error(`Could not download image (HTTP ${imgRes.status}). It may be protected.`);
        }
        const blob = await imgRes.blob();
        const filename = extractFilename(imageUrl);

        // Build multipart form — same format as the playground
        const formData = new FormData();
        formData.append('file', blob, filename);

        // Call the API
        const apiRes = await fetch(`${base}${endpoint}`, {
            method: 'POST',
            headers: { 'X-API-Key': apiKey },
            body: formData,
        });

        if (apiRes.status === 401) throw new Error('Invalid API key. Update it in extension settings.');
        if (apiRes.status === 413) throw new Error('Image is too large (max 10 MB).');
        if (apiRes.status === 400) throw new Error('Unsupported image format. Only JPEG and PNG are accepted.');
        if (!apiRes.ok) throw new Error(`API returned an error (HTTP ${apiRes.status}).`);

        const result = await apiRes.json();

        // Store successful result — results.html will pick this up
        await chrome.storage.session.set({
            pendingAnalysis: {
                status: 'success',
                imageUrl,
                filename,
                result,
                apiUrl: base,
            },
        });

    } catch (err) {
        // Store error — results.html will show it
        await chrome.storage.session.set({
            pendingAnalysis: {
                status: 'error',
                imageUrl,
                error: err.message,
            },
        });
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function extractFilename(url) {
    try {
        const pathname = new URL(url).pathname;
        const name = pathname.split('/').filter(Boolean).pop();
        return name || 'image.jpg';
    } catch {
        return 'image.jpg';
    }
}