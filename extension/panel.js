// ISLens live panel: webcam -> MediaPipe Hands -> rolling buffer -> WS classify -> TTS + overlay.
// Frame wire format matches backend/features.py and recorder/recorder.js exactly:
// { hands: [ { handedness: "Left"|"Right", landmarks: [{x,y,z} x21] }, ... up to 2 ] }

const BUFFER_LEN = 30;
const MOTION_HISTORY = 10;
const MOTION_THRESHOLD = 0.012; // mean abs landmark displacement (normalized coords) between consecutive frames
const CLASSIFY_THROTTLE_MS = 600;
const SPEAK_DEBOUNCE_MS = 2500;

const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const statusEl = document.getElementById("status");
const backendEl = document.getElementById("backend");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const classifyNowBtn = document.getElementById("classifyNowBtn");
const overlayOnPageEl = document.getElementById("overlayOnPage");
const langEl = document.getElementById("lang");
const sttLangEl = document.getElementById("sttLang");
const captionEl = document.getElementById("caption");
const confidenceEl = document.getElementById("confidence");
const roleDeafEl = document.getElementById("roleDeaf");
const roleHearingEl = document.getElementById("roleHearing");
const deafControlsEl = document.getElementById("deafControls");
const hearingControlsEl = document.getElementById("hearingControls");

let hands = null;
let stream = null;
let running = false;
let ws = null;
let wsReady = false;
let awaitingResponse = false;
let recognition = null;
let recognitionShouldRun = false;
let sessionWs = null;
let sessionWsReady = false;

let buffer = []; // raw frames, newest last
const movementScores = [];
let lastClassifyAt = 0;
let lastSpokenLabel = null;
let lastSpokenAt = 0;

function currentRole() {
  return roleHearingEl.checked ? "hearing" : "deaf";
}

function backendHost() {
  return backendEl.value.trim().replace(/^wss?:\/\//, "").replace(/\/.*$/, "");
}

function wsUrl(path) {
  return `ws://${backendHost()}${path}`;
}

function applyRoleVisibility() {
  const hearing = currentRole() === "hearing";
  deafControlsEl.classList.toggle("hidden", hearing);
  hearingControlsEl.classList.toggle("hidden", !hearing);
  classifyNowBtn.classList.toggle("hidden", hearing);
}

function setStatus(text) {
  statusEl.textContent = text;
}

function resultsToFrame(results) {
  const outHands = [];
  const landmarksList = results.multiHandLandmarks || [];
  const handednessList = results.multiHandedness || [];
  for (let i = 0; i < landmarksList.length; i++) {
    const handedness = handednessList[i] ? handednessList[i].label : null;
    if (!handedness) continue;
    outHands.push({
      handedness,
      landmarks: landmarksList[i].map((p) => ({ x: p.x, y: p.y, z: p.z })),
    });
  }
  return { hands: outHands };
}

function drawOverlay(results) {
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  const list = results.multiHandLandmarks || [];
  for (const landmarks of list) {
    ctx.fillStyle = "#ff5252";
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * canvasEl.width, p.y * canvasEl.height, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  ctx.restore();
}

function frameMovement(a, b) {
  // Mean abs (x,y) displacement across hands present in both frames.
  const byHand = (f) => Object.fromEntries(f.hands.map((h) => [h.handedness, h.landmarks]));
  const am = byHand(a);
  const bm = byHand(b);
  let total = 0;
  let count = 0;
  for (const key of Object.keys(am)) {
    if (!bm[key]) continue;
    for (let i = 0; i < am[key].length; i++) {
      total += Math.abs(am[key][i].x - bm[key][i].x) + Math.abs(am[key][i].y - bm[key][i].y);
      count += 2;
    }
  }
  return count > 0 ? total / count : 0;
}

function pushFrame(frame) {
  buffer.push(frame);
  if (buffer.length > BUFFER_LEN) buffer.shift();

  if (buffer.length >= 2) {
    const score = frameMovement(buffer[buffer.length - 1], buffer[buffer.length - 2]);
    movementScores.push(score);
    if (movementScores.length > MOTION_HISTORY) movementScores.shift();
  }
}

function recentMovement() {
  if (movementScores.length === 0) return 0;
  return movementScores.reduce((a, b) => a + b, 0) / movementScores.length;
}

function sendForClassification(force) {
  if (!wsReady || awaitingResponse) return;
  if (buffer.length < BUFFER_LEN * 0.5) return;
  const now = Date.now();
  if (!force) {
    if (now - lastClassifyAt < CLASSIFY_THROTTLE_MS) return;
    if (recentMovement() < MOTION_THRESHOLD) return;
  }
  lastClassifyAt = now;
  awaitingResponse = true;
  ws.send(JSON.stringify({ frames: buffer }));
}

function speak(text, lang) {
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = lang === "hi" ? "hi-IN" : "en-US";
  const voice = window.speechSynthesis.getVoices().find((v) => v.lang.startsWith(lang === "hi" ? "hi" : "en"));
  if (voice) utter.voice = voice;
  else if (lang === "hi") console.warn("No Hindi voice found on this system; falling back to the default voice/lang tag.");
  window.speechSynthesis.speak(utter);
}

async function relayCaptionToActiveTab(text) {
  if (!overlayOnPageEl.checked) return;
  try {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (tab && tab.id) {
      chrome.tabs.sendMessage(tab.id, { type: "islens-caption", text }, () => {
        // Swallow errors from tabs with no content script (e.g. chrome:// pages).
        void chrome.runtime.lastError;
      });
    }
  } catch (e) {
    // Active tab may not be queryable (no tabs, devtools, etc). Non-fatal for the demo.
  }
}

function handleClassifyResult(msg) {
  awaitingResponse = false;
  if (msg.status !== "ok") {
    setStatus(msg.message || "Classifier error");
    return;
  }
  const lang = langEl.value;
  const text = lang === "hi" ? msg.display_hi : msg.display;
  confidenceEl.textContent = `${text}: ${(msg.confidence * 100).toFixed(0)}%`;
  if (!msg.above_threshold) return;

  const now = Date.now();
  if (msg.label === lastSpokenLabel && now - lastSpokenAt < SPEAK_DEBOUNCE_MS) return;

  lastSpokenLabel = msg.label;
  lastSpokenAt = now;
  captionEl.textContent = text;
  speak(text, lang);
  relayCaptionToActiveTab(text);
  publishToSession({ type: "caption", source: "sign", text_en: msg.display, text_hi: msg.display_hi });
}

function publishToSession(payload) {
  if (!sessionWsReady) return;
  sessionWs.send(JSON.stringify(payload));
}

function handleSessionMessage(msg) {
  if (msg.type !== "caption") return;
  const text = msg.source === "sign" ? (langEl.value === "hi" ? msg.text_hi : msg.text_en) : msg.text;
  if (!text) return;
  captionEl.textContent = text;
  relayCaptionToActiveTab(text);
  if (msg.source === "sign") {
    speak(text, langEl.value);
  }
}

function connectSessionWebSocket() {
  let hadError = false;
  sessionWs = new WebSocket(wsUrl("/ws/session"));
  sessionWs.onopen = () => {
    sessionWsReady = true;
    hadError = false;
    if (!wsReady) setStatus("Connected to session relay.");
  };
  sessionWs.onmessage = (event) => {
    try {
      handleSessionMessage(JSON.parse(event.data));
    } catch (e) {
      console.warn("Bad session message:", e);
    }
  };
  sessionWs.onerror = () => {
    hadError = true;
    console.error("session WebSocket error connecting to", wsUrl("/ws/session"));
    setStatus(`Can't reach backend at ${backendHost()} (session). Check IP, WiFi, and firewall.`);
  };
  sessionWs.onclose = (event) => {
    sessionWsReady = false;
    if (running && !hadError) {
      setStatus(`Session relay closed (code ${event.code}${event.reason ? ": " + event.reason : ""}).`);
    }
  };
}

function disconnectSessionWebSocket() {
  if (sessionWs) {
    sessionWs.close();
    sessionWs = null;
  }
  sessionWsReady = false;
}

function connectWebSocket() {
  let hadError = false;
  ws = new WebSocket(wsUrl("/ws/classify"));
  ws.onopen = () => {
    wsReady = true;
    hadError = false;
    setStatus("Connected. Sign a phrase to see it recognized.");
  };
  ws.onmessage = (event) => {
    try {
      handleClassifyResult(JSON.parse(event.data));
    } catch (e) {
      setStatus("Bad response from backend");
      awaitingResponse = false;
    }
  };
  ws.onerror = () => {
    hadError = true;
    console.error("classify WebSocket error connecting to", wsUrl("/ws/classify"));
    setStatus(`Can't reach backend at ${backendHost()} (classify). Check IP, WiFi, and firewall.`);
  };
  ws.onclose = (event) => {
    wsReady = false;
    if (running && !hadError) {
      setStatus(`Backend connection closed (code ${event.code}${event.reason ? ": " + event.reason : ""}).`);
    }
  };
}

function onResults(results) {
  drawOverlay(results);
  pushFrame(resultsToFrame(results));
  sendForClassification(false);
}

function setupHands() {
  const h = new Hands({
    locateFile: (file) => chrome.runtime.getURL(`vendor/mediapipe/${file}`),
  });
  h.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
    selfieMode: true,
  });
  h.onResults(onResults);
  return h;
}

async function startDeafMode() {
  setStatus("Requesting camera...");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 360, height: 270 }, audio: false });
  } catch (e) {
    console.error("getUserMedia failed:", e.name, e.message);
    setStatus(`Camera error: ${e.name} — ${e.message}`);
    startBtn.disabled = false;
    return;
  }
  videoEl.srcObject = stream;
  await new Promise((resolve) => (videoEl.onloadedmetadata = resolve));
  await videoEl.play();

  setStatus("Loading MediaPipe Hands...");
  hands = setupHands();

  setStatus("Connecting to backend...");
  connectWebSocket();

  running = true;
  buffer = [];
  movementScores.length = 0;
  lastSpokenLabel = null;

  const loop = async () => {
    if (!running) return;
    await hands.send({ image: videoEl });
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  stopBtn.disabled = false;
  classifyNowBtn.disabled = false;
  setStatus("Starting camera loop...");
}

function stopDeafMode() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  wsReady = false;
  awaitingResponse = false;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
}

function setupRecognition() {
  const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognitionCtor) {
    setStatus("Speech recognition isn't supported in this browser.");
    return null;
  }
  const rec = new SpeechRecognitionCtor();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = sttLangEl.value === "hi" ? "hi-IN" : "en-US";

  rec.onresult = (event) => {
    let finalText = "";
    let interimText = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      if (result.isFinal) finalText += result[0].transcript;
      else interimText += result[0].transcript;
    }
    if (finalText.trim()) {
      const text = finalText.trim();
      captionEl.textContent = text;
      confidenceEl.textContent = "Heard (final)";
      relayCaptionToActiveTab(text);
      publishToSession({ type: "caption", source: "speech", text, lang: sttLangEl.value });
    } else if (interimText.trim()) {
      captionEl.textContent = interimText.trim();
      confidenceEl.textContent = "Listening...";
    }
  };
  rec.onerror = (event) => {
    console.warn("SpeechRecognition error:", event.error);
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      setStatus(`Microphone error: ${event.error}`);
      recognitionShouldRun = false;
    }
    // Other errors (e.g. "no-speech") are recovered by onend's auto-restart.
  };
  rec.onend = () => {
    // Chrome's continuous recognition still stops itself after a period of
    // silence; restart automatically while the presenter hasn't clicked Stop.
    if (recognitionShouldRun) {
      try {
        rec.start();
      } catch (e) {
        // start() throws if called while already starting; ignore.
      }
    }
  };
  return rec;
}

async function startHearingMode() {
  setStatus("Requesting microphone...");
  try {
    // Ask for mic access up front so the permission prompt (and any denial)
    // is explicit, even though SpeechRecognition manages the stream itself.
    const probe = await navigator.mediaDevices.getUserMedia({ audio: true });
    probe.getTracks().forEach((t) => t.stop());
  } catch (e) {
    console.error("getUserMedia (audio) failed:", e.name, e.message);
    setStatus(`Microphone error: ${e.name} — ${e.message}`);
    startBtn.disabled = false;
    return;
  }

  recognition = setupRecognition();
  if (!recognition) {
    startBtn.disabled = false;
    return;
  }

  recognitionShouldRun = true;
  running = true;
  recognition.start();
  captionEl.textContent = " ";
  stopBtn.disabled = false;
  setStatus("Listening for speech...");
}

function stopHearingMode() {
  recognitionShouldRun = false;
  if (recognition) {
    recognition.stop();
    recognition = null;
  }
}

async function start() {
  startBtn.disabled = true;
  roleDeafEl.disabled = true;
  roleHearingEl.disabled = true;
  if (currentRole() === "hearing") {
    await startHearingMode();
  } else {
    await startDeafMode();
  }
  if (stopBtn.disabled) {
    // Start failed (permission denied, unsupported API, etc): unlock the role picker again.
    roleDeafEl.disabled = false;
    roleHearingEl.disabled = false;
    return;
  }
  connectSessionWebSocket();
}

function stop() {
  running = false;
  stopDeafMode();
  stopHearingMode();
  disconnectSessionWebSocket();
  roleDeafEl.disabled = false;
  roleHearingEl.disabled = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  classifyNowBtn.disabled = true;
  setStatus("Stopped.");
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
classifyNowBtn.addEventListener("click", () => sendForClassification(true));
roleDeafEl.addEventListener("change", applyRoleVisibility);
roleHearingEl.addEventListener("change", applyRoleVisibility);
applyRoleVisibility();
