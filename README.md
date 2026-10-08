# ISLens — working prototype (5 phrases)

A minimal end-to-end demo of the ISLens pitch: sign one of **hello, good
morning, help, thank you, stop** in front of your webcam and hear it spoken
aloud, via a Chrome extension talking to a local Python/PyTorch backend.

This is a demo scoped to a tiny 5-phrase vocabulary, not a production ISL
recognizer. It exists to prove the architecture (client-side landmark
extraction → WebSocket → LSTM classifier → speech synthesis) end-to-end.

## 1. Start the backend

```powershell
cd backend
python -m venv .venv          # first time only
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # first time only
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Leave this running for everything below. On first run there's no trained
model yet — `/ws/classify` will just report that until you train one (step 3).
`--host 0.0.0.0` (instead of `127.0.0.1`) is what makes it reachable from a
second laptop for the two-laptop demo in step 5 — it still works fine for a
single-laptop demo too. Windows Firewall will likely prompt the first time;
allow Python on the **Private network** profile.

## 2. Record training samples

Open `recorder/recorder.html` directly in a normal Chrome tab (not the
extension — this page loads MediaPipe from a CDN, so it needs internet
access and a plain browser tab, e.g. `file:///C:/Users/Admin/Desktop/nitin/SIH/recorder/recorder.html`).

- Allow camera access when prompted.
- Pick a phrase from the dropdown.
- Click **Record sample**, wait for the 3-2-1 countdown, then perform the
  sign clearly within the ~1.3s recording window.
- Repeat **15-20 times per phrase**, with small natural variation each time
  (slightly different speed/position) — this matters more than raw count
  for a 5-class model to generalize.
- The table shows how many samples are saved on disk per phrase so far.

Tips for clean data: good lighting, hands clearly in frame, avoid
recording while your hand is mid-transition into/out of the sign.

## 3. Train the model

```powershell
cd backend
.\.venv\Scripts\python.exe train.py
```

This reads everything in `data/raw/`, trains a small LSTM, and saves
`models/islens_lstm.pt`. It prints validation accuracy — if it's low
(well below ~60-70%), record a few more/cleaner samples for the weaker
phrases and re-run. **Restart the backend** (or just re-run `train.py`,
then restart uvicorn) after training so it picks up the new model.

## 4. Load the Chrome extension

- Go to `chrome://extensions`, enable **Developer mode**, click
  **Load unpacked**, select the `extension/` folder.
- Click the ISLens toolbar icon — it opens a small panel window.
- At the top of the panel, pick **"I am"**: Deaf or Hearing. This choice is
  locked while running — click Stop to switch it.
- The **Backend** field takes just `host:port` (default `127.0.0.1:8000`,
  i.e. talk to the backend running on this same laptop) — the panel builds
  the actual WebSocket URLs from it. For the two-laptop setup in step 5,
  this is the one field that changes on the second laptop.
- **Speak/caption in** (English/Hindi) controls both your own recognized
  signs and — see step 5 — how incoming signs from another laptop get
  spoken on this one.

### Deaf mode (sign → speech)

- Click **Start**, allow camera access.
- Sign one of the 5 phrases. When the panel detects enough hand motion it
  sends the last ~1s of landmarks to the backend automatically; once it's
  confident it speaks the recognized phrase (in the language chosen under
  "Speak/caption in") and, if "Show captions on active tab" is checked,
  overlays it as a subtitle banner on whatever tab is currently focused
  (e.g. a Google Meet call).
- **Classify now** forces a classification of the current buffer
  immediately — useful as a manual fallback if the automatic motion
  trigger doesn't fire during a live demo.

### Hearing mode (speech → captions)

- Click **Start**, allow microphone access.
- Speak normally in the language chosen under "Listen for" (English or
  Hindi). The panel continuously transcribes what it hears (via the
  browser's built-in Web Speech API — this sends audio to the browser's
  speech-recognition service, not to the ISLens backend) and shows it live,
  finalizing each utterance as a caption. If "Show captions on active tab"
  is checked, each finalized line is also overlaid on the active tab, so a
  deaf participant looking at the shared call window can read it.
- There's no manual fallback button here since recognition runs
  continuously; if it stops picking up speech, check the microphone
  permission and that you're not muted at the OS level.

## 5. Two-laptop demo (bidirectional)

One laptop signs (Deaf role), the other is a normal Google Meet participant
(Hearing role); captions/speech flow both ways. Each laptop transcribes its
*own* person's input at the source — the hearing laptop's mic, the deaf
laptop's webcam — and the backend relays whatever one produces to the
other over `/ws/session`. (Why not capture Meet's call audio directly on
one laptop instead of needing the extension on both? Chrome's built-in
speech recognition can only listen to a live microphone, not a captured
tab/system-audio stream — this relay approach sidesteps that limitation
using only what's already built.)

**Setup:**

1. Pick one laptop to host the backend (steps 1-3 above, including
   training). Find its LAN IPv4: `ipconfig`, look for the `IPv4 Address`
   under your active adapter (e.g. `Wi-Fi`) — something like
   `192.168.1.23`. Both laptops must be on the **same WiFi network**.
2. Copy the `extension/` folder to the second laptop (USB drive, shared
   folder, etc.) — it doesn't need `backend/`, `data/`, `models/`, or
   Python, just Chrome + the extension folder. Load it unpacked there too.
3. On the **host** laptop's panel, leave Backend as `127.0.0.1:8000`.
   On the **second** laptop's panel, set Backend to `<host-LAN-IP>:8000`
   (e.g. `192.168.1.23:8000`).
4. Set roles: Deaf on the signer's laptop, Hearing on the other. Click
   Start on both, join the same Google Meet call from both (or just have
   the Meet tab focused on each — the extension overlays onto whichever
   tab is currently active).
5. Sign a phrase on the Deaf laptop → the Hearing laptop speaks it (through
   its own speakers — the hearing person is sitting right there, so this
   doesn't need to be injected into Meet's outgoing audio) and shows it as
   a caption overlay. Speak on the Hearing laptop → the Deaf laptop
   overlays the transcribed caption on its Meet tab.

If nothing arrives on the other side: confirm both panels' status lines
show a connection (not "WebSocket error"), confirm the LAN IP is correct
and hasn't changed, and check Windows Firewall didn't block the host's
`0.0.0.0` bind.

## Known limitations (by design, for a hackathon-scale prototype)

- Vocabulary for sign recognition is fixed to the 5 trained phrases;
  anything else will be misclassified as one of them (there's no
  reject/"unknown" class).
- Speech-to-caption uses the browser's Web Speech API directly with no
  translation step — it transcribes whatever language you tell it to
  listen for, it doesn't translate between languages.
- The two-laptop relay (`/ws/session`) is one unauthenticated global room —
  fine for a private demo on one WiFi network, not something to expose
  beyond that. It also means the real-product end-state (hearing side
  needs no extension at all) isn't what this demo build does — see the
  two-laptop section above for why.
- The extension captures the local webcam/microphone via its own panel
  window rather than injecting capture code into arbitrary pages — this
  sidesteps MV3 content-script CSP/permission issues and is more reliable
  for a live demo. The recognized captions can still be relayed onto the
  active tab for the "overlay on the call" effect.
