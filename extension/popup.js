/**
 * popup.js — Settings popup
 * Loads saved settings from chrome.storage.sync on open.
 * Saves and tests the connection on button click.
 */

const get = id => document.getElementById(id);

// ── Load saved settings ──────────────────────────────────────────────────────

chrome.storage.sync.get(['apiUrl', 'apiKey', 'mode'], ({ apiUrl, apiKey, mode }) => {
    get('apiUrl').value = apiUrl || 'http://localhost:8000';
    get('apiKey').value = apiKey || '';
    get('mode').value = mode || 'explain';
});

// ── Save + test ──────────────────────────────────────────────────────────────

get('save').addEventListener('click', async () => {
    const apiUrl = get('apiUrl').value.trim().replace(/\/$/, '');
    const apiKey = get('apiKey').value.trim();
    const mode = get('mode').value;

    if (!apiUrl) { showStatus('Enter an API URL.', 'err'); return; }
    if (!apiKey) { showStatus('Enter an API key.', 'err'); return; }

    // Save before testing so the values are persisted even if the test fails
    await chrome.storage.sync.set({ apiUrl, apiKey, mode });
    showStatus('Saved. Testing connection…', '');

    try {
        const res = await fetch(`${apiUrl}/v1/inference/health`, {
            headers: { 'X-API-Key': apiKey },
        });

        if (res.ok) showStatus('✓ Connected and ready', 'ok');
        else if (res.status === 401) showStatus('✗ Invalid API key', 'err');
        else if (res.status === 503) showStatus('⚠ API is degraded — check Docker logs', 'err');
        else showStatus(`✗ Server returned HTTP ${res.status}`, 'err');

    } catch {
        showStatus('✗ Could not reach the API — is Docker running?', 'err');
    }
});

// ── Helpers ──────────────────────────────────────────────────────────────────

function showStatus(msg, type) {
    const el = get('status');
    el.textContent = msg;
    el.className = `status ${type}`;
}