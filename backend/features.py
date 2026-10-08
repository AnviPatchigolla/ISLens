"""Turns raw MediaPipe Hands landmarks into a fixed-size, translation/scale
invariant feature sequence for the LSTM.

This is the ONLY place normalization happens. Both the recorder tool and the
live extension send raw landmarks over the wire (JSON); training and
inference both call into this module, so there is no risk of the two paths
drifting apart.

Wire format for one frame (produced by recorder.js / content.js):
{
  "hands": [
    {"handedness": "Left" | "Right", "landmarks": [{"x":.., "y":.., "z":..}, ... 21]},
    ...  # 0, 1, or 2 entries
  ]
}
"""

import numpy as np

NUM_LANDMARKS = 21
LANDMARKS_PER_HAND = NUM_LANDMARKS * 3  # x, y, z
FRAME_DIM = 2 + 2 * LANDMARKS_PER_HAND  # presence flags + left + right = 128
SEQ_LEN = 30

# Landmark index used as the wrist/origin, and the one used to derive a
# scale factor (distance wrist -> middle-finger MCP), per MediaPipe Hands'
# 21-point topology.
WRIST_IDX = 0
SCALE_REF_IDX = 9
MIN_SCALE = 1e-6


def _normalize_hand(landmarks):
    """21 {x,y,z} dicts -> 63 translation/scale-invariant floats."""
    pts = np.array([[p["x"], p["y"], p["z"]] for p in landmarks], dtype=np.float32)
    origin = pts[WRIST_IDX]
    centered = pts - origin
    scale = float(np.linalg.norm(centered[SCALE_REF_IDX]))
    if scale < MIN_SCALE:
        scale = MIN_SCALE
    return (centered / scale).flatten()


def frame_to_vector(frame):
    """One frame dict -> 128-dim feature vector (see FRAME_DIM)."""
    left = np.zeros(LANDMARKS_PER_HAND, dtype=np.float32)
    right = np.zeros(LANDMARKS_PER_HAND, dtype=np.float32)
    left_present = 0.0
    right_present = 0.0

    for hand in frame.get("hands", []):
        vec = _normalize_hand(hand["landmarks"])
        if hand.get("handedness") == "Left":
            left = vec
            left_present = 1.0
        elif hand.get("handedness") == "Right":
            right = vec
            right_present = 1.0

    return np.concatenate([[left_present, right_present], left, right])


def sequence_to_tensor(frames, seq_len=SEQ_LEN):
    """List of frame dicts -> (seq_len, FRAME_DIM) float32 array.

    Pads by repeating the last frame if too short, truncates evenly-spaced
    if too long, so recordings that are a few frames off still line up.
    """
    vectors = [frame_to_vector(f) for f in frames]
    if not vectors:
        return np.zeros((seq_len, FRAME_DIM), dtype=np.float32)

    if len(vectors) < seq_len:
        pad = [vectors[-1]] * (seq_len - len(vectors))
        vectors = vectors + pad
    elif len(vectors) > seq_len:
        idx = np.linspace(0, len(vectors) - 1, seq_len).round().astype(int)
        vectors = [vectors[i] for i in idx]

    return np.stack(vectors).astype(np.float32)
