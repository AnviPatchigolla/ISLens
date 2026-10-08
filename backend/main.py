"""ISLens backend.

Endpoints:
  GET  /phrases          -> the supported phrase labels (recorder + extension both read this)
  POST /record            -> save one recorded landmark sequence for a phrase (used by recorder.html)
  WS   /ws/classify        -> classify a landmark sequence, used by the extension panel
  WS   /ws/session          -> broadcast relay for the two-laptop demo: forwards any caption
                                 message from one connected panel to every other connected panel

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import sequence_to_tensor, SEQ_LEN
from labels import PHRASES, PHRASE_DISPLAY, PHRASE_DISPLAY_HI
from model import load_checkpoint

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "islens_lstm.pt"
CONFIDENCE_THRESHOLD = 0.6

app = FastAPI(title="ISLens backend")

# The recorder page and extension both talk to this over localhost during
# development; CORS is wide open here for demo convenience only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_ckpt = None


def _try_load_model():
    global _model, _ckpt
    if MODEL_PATH.exists():
        _model, _ckpt = load_checkpoint(str(MODEL_PATH))
        print(f"Loaded model from {MODEL_PATH} (classes={_ckpt['classes']})")
    else:
        _model, _ckpt = None, None
        print(f"No trained model at {MODEL_PATH} yet. /ws/classify will report an error until train.py is run.")


@app.on_event("startup")
def startup():
    _try_load_model()


class RecordRequest(BaseModel):
    phrase: str
    frames: list


@app.get("/phrases")
def get_phrases():
    return {"phrases": PHRASES, "display": PHRASE_DISPLAY, "display_hi": PHRASE_DISPLAY_HI}


@app.get("/counts")
def get_counts():
    counts = {}
    for phrase in PHRASES:
        phrase_dir = DATA_DIR / phrase
        counts[phrase] = len(list(phrase_dir.glob("*.json"))) if phrase_dir.exists() else 0
    return {"counts": counts}


@app.post("/record")
def record(req: RecordRequest):
    if req.phrase not in PHRASES:
        return {"status": "error", "message": f"Unknown phrase '{req.phrase}'"}
    if not req.frames:
        return {"status": "error", "message": "No frames provided"}

    phrase_dir = DATA_DIR / req.phrase
    phrase_dir.mkdir(parents=True, exist_ok=True)
    out_path = phrase_dir / f"{uuid.uuid4().hex}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"phrase": req.phrase, "frames": req.frames}, fh)

    count = len(list(phrase_dir.glob("*.json")))
    return {"status": "ok", "count": count}


@app.websocket("/ws/classify")
async def ws_classify(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                frames = payload.get("frames", [])
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "message": "invalid JSON"})
                continue

            if _model is None:
                await websocket.send_json({
                    "status": "error",
                    "message": "Model not trained yet. Run backend/train.py after recording samples.",
                })
                continue

            tensor = sequence_to_tensor(frames, seq_len=_ckpt["seq_len"])
            x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, dim)
            with torch.no_grad():
                logits = _model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

            top_idx = int(np.argmax(probs))
            label = _ckpt["classes"][top_idx]
            confidence = float(probs[top_idx])

            await websocket.send_json({
                "status": "ok",
                "label": label,
                "display": PHRASE_DISPLAY.get(label, label),
                "display_hi": PHRASE_DISPLAY_HI.get(label, label),
                "confidence": confidence,
                "above_threshold": confidence >= CONFIDENCE_THRESHOLD,
            })
    except WebSocketDisconnect:
        pass


# Connected panels for the two-laptop demo relay. A plain list is enough
# here: this is a single ad hoc private session on one WiFi network, not a
# multi-tenant service, so there's no need for rooms/auth/ids.
_session_clients: list[WebSocket] = []


@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket):
    await websocket.accept()
    _session_clients.append(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            for client in list(_session_clients):
                if client is websocket:
                    continue
                try:
                    await client.send_text(raw)
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _session_clients:
            _session_clients.remove(websocket)
