# app_video.py — face-focused single-video AI/Real (no fusion models)

from __future__ import annotations
import os, tempfile
import cv2, numpy as np
import streamlit as st
from utils.image_forensics import ai_probability  # your existing image scorer

# ---------------- I/O helpers ----------------

def _save_to_temp(up) -> str | None:
    try:
        suf = os.path.splitext(up.name)[1].lower() or ".mp4"
        fd, path = tempfile.mkstemp(prefix="vid_", suffix=suf)
        os.close(fd)
        with open(path, "wb") as f: f.write(up.read())
        return path
    except Exception:
        return None

def _sample_frames(path: str, max_frames: int = 32) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    if total > 0:
        idxs = np.linspace(0, max(0, total-1), num=min(max_frames, total), dtype=int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok and fr is not None: frames.append(fr)
    else:
        # fallback: take every 3rd frame
        k = 0
        while len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok: break
            if k % 3 == 0: frames.append(fr)
            k += 1
    cap.release()
    return frames

# ------------- face detection & crops -------------

_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _largest_face(bgr: np.ndarray) -> tuple[int,int,int,int] | None:
    try:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = _haar.detectMultiScale(g, 1.1, 4, minSize=(80,80))
        if len(faces) == 0: return None
        x,y,w,h = max(faces, key=lambda r: r[2]*r[3])
        # keep a slightly tighter crop to avoid background
        pad = int(max(w,h)*0.05)
        x = max(0, x+pad); y = max(0, y+pad)
        w = max(1, w-2*pad); h = max(1, h-2*pad)
        return (x,y,w,h)
    except Exception:
        return None

def _crop_face_or_center(bgr: np.ndarray, size=224) -> np.ndarray:
    box = _largest_face(bgr)
    if box is None:
        # center crop (square) if no face found
        h, w = bgr.shape[:2]
        s = min(h, w)
        x1 = (w - s)//2; y1 = (h - s)//2
        crop = bgr[y1:y1+s, x1:x1+s]
    else:
        x,y,w,h = box
        crop = bgr[y:y+h, x:x+w]
    if crop is None or crop.size == 0: crop = bgr
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

# ------------- temporal artifact score -------------

def _temporal_score(face_frames: list[np.ndarray]) -> float:
    """0..1; high -> suspicious (too flickery OR too smooth)."""
    if len(face_frames) < 3: return 0.0
    diffs = []
    prev = None
    for fr in face_frames:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (0,0), 0.8)
        if prev is not None:
            d = cv2.absdiff(g, prev)
            diffs.append(float(np.mean(d)))
        prev = g
    if not diffs: return 0.0
    arr = np.array(diffs, np.float32)
    m, s = float(arr.mean()), float(arr.std())
    n_m = np.clip((m - 3.0)/22.0, 0.0, 1.0)
    n_s = np.clip((s - 1.5)/8.0, 0.0, 1.0)
    too_smooth  = 1.0 - max(n_m, n_s)
    too_flicker = n_s
    return float(np.clip(0.55*too_flicker + 0.45*too_smooth, 0.0, 1.0))

# ------------- decision logic -------------

def _video_ai_probability(face_frames: list[np.ndarray]) -> tuple[float, dict]:
    """Return final_prob (0..1) and diagnostics."""
    if not face_frames: return 0.0, {"frames": 0}

    # per-frame AI probabilities on face crops
    per = []
    for fr in face_frames:
        p, _ = ai_probability(fr)     # 0..1
        per.append(float(np.clip(p, 0.0, 1.0)))

    arr = np.array(per, dtype=np.float32)
    pmax = float(arr.max())
    pmean = float(arr.mean())
    p95 = float(np.percentile(arr, 95))
    p75 = float(np.percentile(arr, 75))

    # how many frames are confidently AI?
    hi = float((arr >= 0.65).mean())  # fraction of frames

    temporal = _temporal_score(face_frames)

    # conservative fusion (face-focused)
    fused = 0.50*pmax + 0.30*pmean + 0.20*temporal

    # hard rules to avoid “obvious AI => Real”
    if hi >= 0.25:      # ≥25% frames confidently AI
        fused = max(fused, 0.62)
    if temporal >= 0.70:
        fused = max(fused, 0.60)

    return float(np.clip(fused, 0.0, 1.0)), {
        "frames": len(face_frames),
        "pmax": pmax, "pmean": pmean, "p95": p95, "p75": p75,
        "hi_frac": hi, "temporal": temporal
    }

# ------------- Streamlit UI -------------

def run_video():
    st.subheader("Single Video — AI or Real (face-focused)")
    up = st.file_uploader(
        "Upload a short video (≤60s ideal)", 
        type=["mp4","mov","m4v","avi","mpeg","mpg","mkv"],
        key="video_single_face"
    )
    if not up:
        st.info("Upload a video to analyze.")
        return

    path = _save_to_temp(up)
    if not path:
        st.error("Could not read this video.")
        return

    st.video(path)

    with st.status("Sampling frames…", expanded=False):
        frames = _sample_frames(path, max_frames=32)

    if not frames:
        st.error("No frames extracted.")
        return

    with st.status("Cropping faces…", expanded=False):
        face_frames = [_crop_face_or_center(fr, 224) for fr in frames]

    with st.status("Scoring…", expanded=False):
        prob, diag = _video_ai_probability(face_frames)

    # Stricter threshold than images
    THRESH = 0.55
    if prob >= THRESH:
        st.error(f"AI Generated — probability {prob*100:.1f}%")
    else:
        st.success(f"Real Video — AI probability {prob*100:.1f}%")

    with st.expander("Diagnostics"):
     st.write(
        "Frames: {frames}\n"
        "pmax: {pmax:.2f}\n"
        "p95: {p95:.2f}\n"
        "p75: {p75:.2f}\n"
        "Mean: {mean:.2f}\n"
        "High-confidence frame fraction (>=0.65): {hi:.2f}\n"
        "Temporal artifact score: {temp:.2f}".format(
            frames=diag.get("frames", 0),
            pmax=diag.get("pmax", 0.0),
            p95=diag.get("p95", 0.0),
            p75=diag.get("p75", 0.0),
            mean=diag.get("pmean", 0.0),
            hi=diag.get("hi_frac", 0.0),
            temp=diag.get("temporal", 0.0),
        )
    )
    st.caption("Tip: If obvious AI is still 'Real', increase THRESH to 0.60–0.70.")
    try:
        os.remove(path)
    except Exception:
        pass
