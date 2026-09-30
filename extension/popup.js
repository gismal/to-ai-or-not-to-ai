const $ = (id) => document.getElementById(id);
let currentMode = 'explain';

// Load saved settings
chrome.storage.sync.get(
    ['apiUrl', 'apiKey', 'mode', 'passiveBadge'],
    ({ apiUrl, apiKey, mode, passiveBadge }) => {
        $('apiUrl').value = apiUrl || '';
        $('apiKey').value = apiKey || '';
        $('passiveBadge').checked = passiveBadge || false;
        setMode(mode || 'explain');

        if (apiKey && apiUrl) pingApi(apiUrl, apiKey);
    }
);

// ── Mode toggle ──
function setMode(m) {
    currentMode = m;
    $('modeFast').classList.toggle('on', m === 'predict');
    $('modeExplain').classList.toggle('on', m === 'explain');
}

$('modeFast').addEventListener('click', () => setMode('predict'));
$('modeExplain').addEventListener('click', () => setMode('explain'));

// ── Save ──
$('save').addEventListener('click', async () => {
    const apiUrl = $('apiUrl').value.trim().replace(/\/$/, '');
    const apiKey = $('apiKey').value.trim();

    if (!apiUrl) { showStatus('Enter an API URL.', 'err'); return; }
    if (!apiKey) { showStatus('Enter an API key.', 'err'); return; }

    await chrome.storage.sync.set({
        apiUrl,
        apiKey,
        mode: currentMode,
        passiveBadge: $('passiveBadge').checked,
    });

    showStatus('Saved. Testing connection...', '');
    await pingApi(apiUrl, apiKey);
});

// ── Ping ──
async function pingApi(apiUrl, apiKey) {
    try {
        const res = await fetch(`${apiUrl}/v1/inference/health`, {
            headers: { 'X-API-Key': apiKey },
        });
        if (res.ok) {
            showStatus('Connected and ready.', 'ok');
            setDot('ok');
        } else if (res.status === 401) {
            showStatus('Invalid API key.', 'err');
            setDot('err');
        } else {
            showStatus(`Server returned ${res.status}.`, 'err');
            setDot('err');
        }
    } catch {
        showStatus('Could not reach the API.', 'err');
        setDot('err');
    }
}

function showStatus(msg, type) {
    const el = $('status');
    el.textContent = msg;
    el.className = `status ${type}`;
}

function setDot(state) {
    const dot = $('connDot');
    dot.className = `conn-dot ${state}`;
    dot.title = state === 'ok' ? 'API connected' : 'API unreachable';
}