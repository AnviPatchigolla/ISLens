// Lightweight on-page subtitle overlay. All camera capture, MediaPipe
// inference, and WebSocket work happens in panel.js (the extension panel);
// this script only renders whatever text the panel tells it to, so it stays
// unaffected by page-specific CSP or camera permission quirks.

let banner = null;
let hideTimer = null;

function ensureBanner() {
  if (banner) return banner;
  banner = document.createElement("div");
  banner.id = "islens-subtitle-banner";
  document.documentElement.appendChild(banner);
  return banner;
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === "islens-caption") {
    const el = ensureBanner();
    el.textContent = msg.text;
    el.classList.add("islens-visible");
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => el.classList.remove("islens-visible"), 3500);
  }
});
