# utils/video_forensics.py
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict

# reuse ai_probability from image_forensics
try:
    from .image_forensics import ai_probability
except Exception:
    try:
        from image_forensics import ai_probability  # fallback
    except Exception:
        ai_probability = None


def _sample_video_frames(path: str, max_samples: int = 64) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: List[np.ndarray] = []
    if total <= 0:
        read = 0
        while read < max_samples:
            ok, f = cap.read()
            if not ok or f is None:
                break
            frames.append(f)
            read += 1
        cap.release()
        return frames

    idxs = np.linspace(0, max(0, total - 1), num=min(max_samples, max(1, total))).astype(int)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
    cap.release()
    return frames


def _rescale_if_needed(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    if arr.max() > 1.5:  # likely 0..100
        return arr / 100.0
    return arr


def video_ai_probability(path: str, samples: int = 64, debug: bool = False) -> Tuple[float, Dict]:
    """
    Sample frames and run image_forensics.ai_probability on each frame.
    Returns (aggregated_score_in_0_1, details) where details includes:
      - per_frame: list of probs
      - per_feature_means: averaged features from ai_probability debug details
      - mean/median/max/topk_mean/proportion...
    """
    if ai_probability is None:
        raise RuntimeError("ai_probability not available — ensure utils/image_forensics.py exports ai_probability")

    frames = _sample_video_frames(path, max_samples=samples)
    if not frames:
        return 0.0, {"err": "no_frames", "frames": 0}

    per_frame_probs: List[float] = []
    per_frame_details: List[Dict] = []

    for f in frames:
        try:
            # request debug details so we can aggregate features
            p_raw, det = ai_probability(f, debug=True)
            p = float(p_raw)
            per_frame_probs.append(p)
            per_frame_details.append(det if isinstance(det, dict) else {})
        except Exception:
            per_frame_probs.append(0.0)
            per_frame_details.append({})

    arr = np.asarray(per_frame_probs, dtype=np.float32) if per_frame_probs else np.asarray([], dtype=np.float32)
    arr = _rescale_if_needed(arr)

    mean = float(arr.mean()) if arr.size else 0.0
    median = float(np.median(arr)) if arr.size else 0.0
    mx = float(arr.max()) if arr.size else 0.0

    # top-k mean
    topk = 3
    if arr.size:
        k = min(topk, arr.size)
        topk_mean = float(np.sort(arr)[-k:].mean())
    else:
        topk_mean = 0.0

    prop_ge_045 = float((arr >= 0.45).mean()) if arr.size else 0.0

    # Aggregate per-feature means from per_frame_details if available.
    feature_keys = ["skin_smooth_face", "tex_rep_face", "ela_face", "fft_face", "hf_face", "tex_rep_global", "ela_global"]
    feat_acc = {k: [] for k in feature_keys}
    for d in per_frame_details:
        if not isinstance(d, dict):
            continue
        for k in feature_keys:
            v = d.get(k)
            if v is None:
                continue
            try:
                feat_acc[k].append(float(v))
            except Exception:
                pass

    per_feature_means = {}
    for k, vals in feat_acc.items():
        if vals:
            per_feature_means[k + "_mean"] = float(np.mean(vals))
            per_feature_means[k + "_std"] = float(np.std(vals))
            per_feature_means[k + "_max"] = float(np.max(vals))
        else:
            per_feature_means[k + "_mean"] = 0.0
            per_feature_means[k + "_std"] = 0.0
            per_feature_means[k + "_max"] = 0.0

    details = {
        "per_frame": per_frame_probs,
        "mean": mean,
        "median": median,
        "max": mx,
        "topk_mean": topk_mean,
        "prop_ge_045": prop_ge_045,
        "samples": int(arr.size),
        "per_feature_means": per_feature_means,
    }

    if debug:
        details["raw_per_frame_details"] = per_frame_details

    aggregated = max(mx, topk_mean, median, mean * 1.2, prop_ge_045)
    aggregated = float(max(0.0, min(1.0, aggregated)))
    return aggregated, details