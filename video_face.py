from _future_ import annotations
import cv2
import numpy as np
from typing import Tuple, List, Dict

# reuse ai_probability from image_forensics
try:
    from .image_forensics import ai_probability
except Exception:
    # fallback if package layout differs
    from image_forensics import ai_probability  # type: ignore


def _sample_video_frames(path: str, max_samples: int = 64) -> List[np.ndarray]:
    """Sample up to max_samples frames uniformly across the video."""
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
    """If values look like percentages (max > 1.5), scale to 0..1."""
    if arr.size == 0:
        return arr
    if arr.max() > 1.5:  # likely 0..100 scale
        return arr / 100.0
    return arr


def video_ai_probability(path: str, samples: int = 64, debug: bool = False) -> Tuple[float, Dict]:
    """
    Sample frames and run image_forensics.ai_probability on each frame.
    Returns (aggregated_score_in_0_1, details).

    Aggregation:
      - rescale frame probs if they appear to be 0..100
      - compute mean, median, max, topk_mean (top 3 frames)
      - compute prop_ge_045 (proportion frames >= 0.45)
      - aggregated score = max(max, topk_mean, median, mean*1.2, prop_ge_045)
        (this makes the aggregator sensitive to single strong frames and to a cluster)
    """
    frames = _sample_video_frames(path, max_samples=samples)
    if not frames:
        return 0.0, {"err": "no_frames", "frames": 0}

    per_frame: List[float] = []
    for f in frames:
        try:
            p_raw, _ = ai_probability(f, debug=False)  # expected (prob, details)
            p = float(p_raw)
        except Exception:
            p = 0.0
        per_frame.append(p)

    arr = np.asarray(per_frame, dtype=np.float32) if per_frame else np.asarray([], dtype=np.float32)

    # Fix scale if detector returned 0..100
    arr = _rescale_if_needed(arr)

    mean = float(arr.mean()) if arr.size else 0.0
    median = float(np.median(arr)) if arr.size else 0.0
    mx = float(arr.max()) if arr.size else 0.0

    # top-k mean: mean of the top 3 frame probs (robust to sparse strong signals)
    topk = 3
    if arr.size:
        k = min(topk, arr.size)
        topk_mean = float(np.sort(arr)[-k:].mean())
    else:
        topk_mean = 0.0

    prop_ge_045 = float((arr >= 0.45).mean()) if arr.size else 0.0

    details = {
        "per_frame": per_frame,   # raw values (as returned by ai_probability)
        "mean": mean,
        "median": median,
        "max": mx,
        "topk_mean": topk_mean,
        "prop_ge_045": prop_ge_045,
        "samples": int(arr.size)
    }

    if debug:
        details["raw_arr_scaled"] = arr.tolist()

    # Aggressive aggregator tuned to catch single-strong or top-k signals
    aggregated = max(mx, topk_mean, median, mean * 1.2, prop_ge_045)

    # clamp
    aggregated = float(max(0.0, min(1.0, aggregated)))

    return aggregated, details