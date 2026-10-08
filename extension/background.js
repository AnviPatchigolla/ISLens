// Opens the ISLens panel as a persistent popup window (not the default
// browser_action popup, which closes on blur — bad for a live camera panel
// the presenter needs to keep open next to their call window).
chrome.action.onClicked.addListener(() => {
  chrome.windows.create({
    url: chrome.runtime.getURL("panel.html"),
    type: "popup",
    width: 420,
    height: 640,
  });
});
