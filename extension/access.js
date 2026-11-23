// Site access, asked for only when it is needed. Shared by the background worker,
// the popup and the results page.
//
// The extension installs with access to its own API only. It asks for more when:
// - a right-clicked image lives on another site (all sites, or just that image's site);
// - passive mode is switched on (all sites: it marks images on every page you visit);
// - you point it at your own server (that server's address).
//
// chrome.permissions.request() only works during a click, so callers must call
// these request functions straight from a click handler, before any other await.

const ALL_SITES = Object.freeze({ origins: ['<all_urls>'] });
const PASSIVE_SCRIPT_ID = 'aon-passive-badges';

function originPattern(url) {
  return `${new URL(url).origin}/*`;
}

function needsSiteAccess(url) {
  return /^https?:/.test(url || '');   // data: images are inside the page, no access needed
}

async function canReach(url) {
  if (!needsSiteAccess(url)) return true;
  return chrome.permissions.contains({ origins: [originPattern(url)] });
}

function requestAllSites() {
  return chrome.permissions.request(ALL_SITES);
}

function requestSite(url) {
  return chrome.permissions.request({ origins: [originPattern(url)] });
}