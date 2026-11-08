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
importScripts('config.js');
const { DEMO_KEY, DEFAULT_API_URL } = AON;

const MENU_ID = 'aon-analyze';
const MAX_PX = 1500;
const TIMEOUT_MS = 90000;           // free hosting can take up to a minute to wake
const UPLOAD_TYPES = ['image/jpeg', 'image/png'];

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.removeAll(() =>
        chrome.contextMenus.create({ id: MENU_ID, title: 'Analyze with AI Detector', contexts: ['image'] }));
});

// Right-click flow: one storage record per analysis, so several windows never collide.
chrome.contextMenus.onClicked.addListener(async (info) => {
    if (info.menuItemId !== MENU_ID || !/^(https?:|data:image\/)/.test(info.srcUrl || '')) return;

    const settings = await getSettings();
    const id = crypto.randomUUID();
    const key = `analysis:${id}`;
    const base = { imageUrl: info.srcUrl, filename: filenameFrom(info.srcUrl), mode: settings.mode };

    await chrome.storage.session.set({ [key]: { ...base, status: 'loading', startedAt: Date.now() } });
    chrome.windows.create({
        url: chrome.runtime.getURL(`results.html?id=${id}`), type: 'popup', width: 460, height: 640, focused: true,
    });

    try {
        const result = await analyze(info.srcUrl, settings);
        await chrome.storage.session.set({ [key]: { ...base, status: 'success', result, apiUrl: settings.apiUrl } });
    } catch (err) {
        await chrome.storage.session.set({ [key]: { ...base, status: 'error', error: friendly(err) } });
    }
});

// Passive badges: content scripts ask us to fetch, because this worker is not bound by page CORS.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (sender.id !== chrome.runtime.id || msg?.type !== 'aon-predict' || !/^https?:/.test(msg.src || '')) return;
    getSettings()
        .then((s) => analyze(msg.src, { ...s, mode: 'predict' }))
        .then((result) => sendResponse({ ok: true, result }))
        .catch((err) => sendResponse({ ok: false, error: friendly(err) }));
    return true;                      // keep the channel open for the async reply
});

async function getSettings() {
    const s = await chrome.storage.sync.get({ apiKey: DEMO_KEY, apiUrl: DEFAULT_API_URL, mode: 'explain' });
    return { ...s, apiUrl: s.apiUrl.replace(/\/$/, '') };
}

async function analyze(imageUrl, { apiKey, apiUrl, mode }) {
    const source = await fetch(imageUrl);
    if (!source.ok) throw Object.assign(new Error('image'), { imageStatus: source.status });
    const blob = await toUploadable(await source.blob());

    const form = new FormData();
    const name = filenameFrom(imageUrl).replace(/\.\w+$/, '') + (blob.type === 'image/png' ? '.png' : '.jpg');
    form.append('file', blob, name);

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
        const res = await fetch(`${apiUrl}/v1/inference/${mode === 'explain' ? 'explain' : 'predict'}`, {
            method: 'POST', headers: { 'X-API-Key': apiKey }, body: form, signal: ctrl.signal,
        });
        if (!res.ok) throw Object.assign(new Error('http'), { status: res.status });
        return await res.json();
    } finally {
        clearTimeout(timer);
    }
}

// WebP, AVIF and GIF are re-encoded to JPEG; large images are shrunk.
async function toUploadable(blob) {
    const bitmap = await createImageBitmap(blob);
    const scale = Math.min(1, MAX_PX / Math.max(bitmap.width, bitmap.height));
    if (UPLOAD_TYPES.includes(blob.type) && scale === 1) { bitmap.close(); return blob; }
    const w = Math.round(bitmap.width * scale);
    const h = Math.round(bitmap.height * scale);
    const ctx = new OffscreenCanvas(w, h).getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close();
    return ctx.canvas.convertToBlob({ type: 'image/jpeg', quality: 0.92 });
}

function friendly(err) {
    const byStatus = {
        400: 'Unsupported or unreadable image.',
        401: 'The key was refused. Check it in the extension popup.',
        413: 'The image is too large.',
        429: 'Rate limit reached. Wait a moment and try again.',
    };
    if (err.status) return byStatus[err.status] || `The server answered ${err.status}.`;
    if (err.imageStatus) return `Could not download the image (HTTP ${err.imageStatus}).`;
    if (err.name === 'AbortError') return 'The server took too long to answer. It may be waking up, so try again.';
    if (err instanceof TypeError) return 'Could not reach the server or the image.';
    return 'This image could not be read.';
}

function filenameFrom(url) {
    try { return new URL(url).pathname.split('/').filter(Boolean).pop() || 'image.jpg'; }
    catch { return 'image.jpg'; }
}