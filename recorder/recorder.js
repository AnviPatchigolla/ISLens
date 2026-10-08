// ISLens data recorder: webcam -> MediaPipe Hands -> raw landmark frames -> POST /record
// Wire format for one frame (must match backend/features.py exactly):
// { hands: [ { handedness: "Left"|"Right", landmarks: [{x,y,z} x21] }, ... up to 2 ] }

const TARGET_FRAMES = 30;
const RECORD_MS = 1300; // slightly over 1s of capture at ~24-30fps
const COUNTDOWN_STEPS = ["3", "2", "1", "Go!"];

const statusEl = document.getElementById("status");
const backendEl = document.getElementById("backend");
const phraseEl = document.getElementById("phrase");
const recordBtn = document.getElementById("recordBtn");
const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const countdownEl = document.getElementById("countdown");
const countsTable = document.getElementById("counts");
const logEl = document.getElementById("log");

let phrases = [];
let display = {};
let latestResults = null;
let recording = false;
let capturedFrames = [];

function log(msg) {
  const line = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.textContent = line + "\n" + logEl.textContent;
}

function backendUrl() {
  return backendEl.value.replace(/\/$/, "");
}

async function loadPhrases() {
  const res = await fetch(`${backendUrl()}/phrases`);
  const data = await res.json();
  phrases = data.phrases;
  display = data.display;
  phraseEl.innerHTML = phrases.map((p) => `<option value="${p}">${display[p]}</option>`).join("");
  await refreshCounts();
}

async function refreshCounts() {
  const res = await fetch(`${backendUrl()}/counts`);
  const data = await res.json();
  countsTable.innerHTML =
    "<tr><th>Phrase</th><th>Samples recorded (on disk)</th></tr>" +
    phrases.map((p) => `<tr id="count-${p}"><td>${display[p]}</td><td>${data.counts[p] || 0}</td></tr>`).join("");
}

function bumpCount(phrase, count) {
  const row = document.getElementById(`count-${phrase}`);
  if (row) row.children[1].textContent = count;
}

function resultsToFrame(results) {
  const hands = [];
  const landmarksList = results.multiHandLandmarks || [];
  const handednessList = results.multiHandedness || [];
  for (let i = 0; i < landmarksList.length; i++) {
    const handedness = handednessList[i] ? handednessList[i].label : null;
    if (!handedness) continue;
    hands.push({
      handedness,
      landmarks: landmarksList[i].map((p) => ({ x: p.x, y: p.y, z: p.z })),
    });
  }
  return { hands };
}

function drawOverlay(results) {
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      if (window.drawConnectors && window.HAND_CONNECTIONS) {
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "#00e676", lineWidth: 2 });
      }
      if (window.drawLandmarks) {
        drawLandmarks(ctx, landmarks, { color: "#ff5252", radius: 3 });
      }
    }
  }
  ctx.restore();
}

function onResults(results) {
  latestResults = results;
  drawOverlay(results);
  if (recording) {
    capturedFrames.push(resultsToFrame(results));
  }
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false });
  videoEl.srcObject = stream;
  await new Promise((resolve) => (videoEl.onloadedmetadata = resolve));
  await videoEl.play();
}

function setupHands() {
  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });
  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
    selfieMode: true,
  });
  hands.onResults(onResults);
  return hands;
}

async function mainLoop(hands) {
  const step = async () => {
    await hands.send({ image: videoEl });
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runCountdownAndRecord() {
  recordBtn.disabled = true;
  countdownEl.style.display = "block";
  for (const step of COUNTDOWN_STEPS) {
    countdownEl.textContent = step;
    await sleep(500);
  }
  countdownEl.style.display = "none";

  capturedFrames = [];
  recording = true;
  statusEl.textContent = "Recording... perform the sign now.";
  await sleep(RECORD_MS);
  recording = false;

  const nonEmpty = capturedFrames.filter((f) => f.hands.length > 0);
  statusEl.textContent = `Captured ${capturedFrames.length} frames (${nonEmpty.length} with a hand detected).`;

  if (nonEmpty.length < TARGET_FRAMES * 0.3) {
    log(`Rejected sample: too few frames had a detected hand (${nonEmpty.length}/${capturedFrames.length}). Try again with better lighting/hand visibility.`);
    recordBtn.disabled = false;
    return;
  }

  const phrase = phraseEl.value;
  try {
    const res = await fetch(`${backendUrl()}/record`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase, frames: capturedFrames }),
    });
    const data = await res.json();
    if (data.status === "ok") {
      bumpCount(phrase, data.count);
      log(`Saved sample for "${display[phrase]}" (${data.count} total recorded on disk for this phrase).`);
    } else {
      log(`Backend rejected sample: ${data.message}`);
    }
  } catch (err) {
    log(`Failed to reach backend at ${backendUrl()}: ${err}`);
  }
  recordBtn.disabled = false;
}

async function init() {
  try {
    await loadPhrases();
  } catch (err) {
    statusEl.textContent = `Could not reach backend at ${backendUrl()}. Start it (uvicorn) and reload this page.`;
    return;
  }

  statusEl.textContent = "Requesting camera access...";
  await setupCamera();

  statusEl.textContent = "Loading MediaPipe Hands model...";
  const hands = setupHands();
  await mainLoop(hands);

  statusEl.textContent = "Ready. Pick a phrase, then click Record.";
  recordBtn.disabled = false;
  recordBtn.addEventListener("click", runCountdownAndRecord);
}

init();
