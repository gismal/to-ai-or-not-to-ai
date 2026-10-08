// Defaults make the extension work the moment it is installed.
// Empty fields in "Use your own server or key" mean "use these".
const { DEMO_KEY, DEFAULT_API_URL } = AON;   // see config.js
const DEFAULTS = { apiUrl: DEFAULT_API_URL, apiKey: DEMO_KEY, mode: 'explain', passiveBadge: false };

const $ = (id) => document.getElementById(id);
let state = { ...DEFAULTS };
let checkId = 0;

init();

async function init() {
    state = await chrome.storage.sync.get(DEFAULTS);

    $('apiUrl').placeholder = DEFAULT_API_URL;
    $('apiUrl').value = state.apiUrl === DEFAULT_API_URL ? '' : state.apiUrl;
    $('apiKey').value = state.apiKey === DEMO_KEY ? '' : state.apiKey;
    $('passiveBadge').checked = state.passiveBadge;
    $('adv').open = state.apiUrl !== DEFAULT_API_URL || state.apiKey !== DEMO_KEY;
    renderMode(state.mode);

    $('modeFast').addEventListener('click', () => setMode('predict'));
    $('modeExplain').addEventListener('click', () => setMode('explain'));
    $('passiveBadge').addEventListener('change', (e) => chrome.storage.sync.set({ passiveBadge: e.target.checked }));
    $('save').addEventListener('click', saveCustom);
    $('reset').addEventListener('click', resetToDemo);

    check();
}

function renderMode(m) {
    $('modeFast').classList.toggle('on', m === 'predict');
    $('modeExplain').classList.toggle('on', m === 'explain');
    $('modeFast').setAttribute('aria-pressed', String(m === 'predict'));
    $('modeExplain').setAttribute('aria-pressed', String(m === 'explain'));
}

function setMode(m) {
    state.mode = m;
    renderMode(m);
    chrome.storage.sync.set({ mode: m });   // saved immediately, no button needed
}

async function saveCustom() {
    const url = ($('apiUrl').value.trim() || DEFAULT_API_URL).replace(/\/$/, '');
    if (!/^https?:\/\//.test(url)) { showStatus('Enter a full address starting with https://', true); return; }
    const key = $('apiKey').value.trim() || DEMO_KEY;
    state.apiUrl = url;
    state.apiKey = key;
    await chrome.storage.sync.set({ apiUrl: url, apiKey: key });
    showStatus('');
    check();
}

async function resetToDemo() {
    state.apiUrl = DEFAULT_API_URL;
    state.apiKey = DEMO_KEY;
    await chrome.storage.sync.set({ apiUrl: DEFAULT_API_URL, apiKey: DEMO_KEY });
    $('apiUrl').value = '';
    $('apiKey').value = '';
    showStatus('');
    check();
}

// Health check. Free hosting sleeps, so a slow first answer is explained, not hidden.
async function check() {
    const id = ++checkId;
    const demo = state.apiKey === DEMO_KEY;
    setLine('Setting the stage...');
    setDot('');

    const ctrl = new AbortController();
    const slow = setTimeout(() => id === checkId &&
        setLine('Waking the server. Free hosting sleeps, so this can take up to a minute.'), 4000);
    const hard = setTimeout(() => ctrl.abort(), 70000);

    try {
        const res = await fetch(`${state.apiUrl.replace(/\/$/, '')}/v1/inference/health`, {
            headers: { 'X-API-Key': state.apiKey }, signal: ctrl.signal,
        });
        if (id !== checkId) return;
        if (res.ok) { setLine(demo ? 'Public demo, ready.' : 'Your key, ready.'); setDot('ok'); }
        else if (res.status === 401) {
            setLine(demo ? 'The demo key was refused. Try again later.' : 'That key was not recognised.', true);
            setDot('err');
        } else { setLine(`The server answered ${res.status}.`, true); setDot('err'); }
    } catch {
        if (id === checkId) { setLine('Could not reach the server.', true); setDot('err'); }
    } finally {
        clearTimeout(slow);
        clearTimeout(hard);
    }
}

function setLine(msg, isErr = false) {
    const el = $('accessLine');
    el.textContent = msg;
    el.classList.toggle('err', isErr);
}
function showStatus(msg, isErr = false) {
    const el = $('status');
    el.textContent = msg;
    el.classList.toggle('err', isErr);
}
function setDot(state_) {
    const dot = $('connDot');
    dot.className = `conn-dot ${state_}`;
    dot.setAttribute('aria-label', state_ === 'ok' ? 'Connected' : state_ === 'err' ? 'Unreachable' : 'Checking');
}