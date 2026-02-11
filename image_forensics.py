# utils/image_forensics.py
# Image forensic helpers: ai_probability, localization heatmaps, face crop, embedding, identity_similarity.
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Optional embedding backend (face_recognition)
try:
    import face_recognition
    _HAS_FR = True
except Exception:
    face_recognition = None
    _HAS_FR = False


# ---------------- small helpers ----------------

def _l2norm(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return None
    return v / n


# ---------------- face detection / crop ----------------

def _detect_face_rects(bgr: np.ndarray) -> list:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return []
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80)
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _crop_face_strict(bgr: np.ndarray, margin: float = 0.25) -> Optional[np.ndarray]:
    boxes = _detect_face_rects(bgr)
    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    cx, cy = x + w // 2, y + h // 2
    s = int(max(w, h) * (1.0 + margin))
    x1 = max(0, cx - s // 2)
    y1 = max(0, cy - s // 2)
    x2 = min(bgr.shape[1], x1 + s)
    y2 = min(bgr.shape[0], y1 + s)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def crop_face(bgr: np.ndarray) -> Optional[np.ndarray]:
    return _crop_face_strict(bgr)


# ---------------- embedding (face_recognition or fallback) ----------------

def _fallback_embedding(bgr: np.ndarray, out_dim: int = 128) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return np.zeros(out_dim, dtype=np.float32)
    hist = cv2.calcHist([bgr], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256]).flatten()
    norm = np.linalg.norm(hist) + 1e-9
    hist = (hist / norm).astype(np.float32)
    if hist.size < out_dim:
        hist = np.pad(hist, (0, out_dim - hist.size), mode='constant')
    return hist[:out_dim].astype(np.float32)


def face_embedding(bgr: np.ndarray) -> Optional[np.ndarray]:
    if bgr is None or bgr.size == 0:
        return None
    if _HAS_FR:
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            if not boxes:
                return None
            enc = face_recognition.face_encodings(rgb, known_face_locations=[boxes[0]])
            if not enc:
                return None
            return np.asarray(enc[0], dtype=np.float32)
        except Exception:
            return _fallback_embedding(bgr, out_dim=128)
    return _fallback_embedding(bgr, out_dim=128)


# ---------------- ORB + RANSAC geometry ----------------

def _orb_match_score(faceA: np.ndarray, faceB: np.ndarray) -> Tuple[float, int, int, int]:
    try:
        grayA = cv2.cvtColor(faceA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(faceB, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0, 0, 0, 0

    orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, edgeThreshold=15, patchSize=31)
    k1, d1 = orb.detectAndCompute(grayA, None)
    k2, d2 = orb.detectAndCompute(grayB, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return 0.0, 0, len(k1) if k1 is not None else 0, len(k2) if k2 is not None else 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < 12:
        return 0.0, 0, len(good), len(knn)

    ptsA = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return 0.0, 0, len(good), len(knn)

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(1, len(good))
    score01 = float(np.clip((inlier_ratio - 0.35) / 0.65, 0.0, 1.0))
    return score01, inliers, len(good), len(knn)


# ---------------- helper features for AI detection ----------------

def _highfreq_ratio_fft(bgr: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        target = 256
        if max(h, w) > target:
            scale = target / max(h, w)
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        total = magnitude.sum() + 1e-9
        H, W = magnitude.shape
        cy, cx = H//2, W//2
        Y, X = np.ogrid[:H, :W]
        r = np.sqrt((X-cx)**2 + (Y-cy)**2)
        r_norm = r / r.max()
        mask = (r_norm >= 0.6).astype(np.float32)
        high = (magnitude * mask).sum()
        ratio = float(high / total)
        return float(np.tanh(ratio * 3.0))
    except Exception:
        return 0.0


def _highfreq_ratio(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    hf = gray - low
    hf_energy = float(np.mean(np.abs(hf)))
    base = float(np.mean(np.abs(gray))) + 1e-6
    ratio = np.clip(hf_energy / base, 0.0, 5.0)
    return float(np.tanh(ratio * 0.6))


def _ela_inconsistency(bgr: np.ndarray, q: int = 90) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return 0.0
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr, rec).astype(np.float32)
    v = float(np.mean(diff))
    return float(np.clip(v / 30.0, 0.0, 1.0))


def _noise_irregularity(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    noise = cv2.subtract(gray, blur).astype(np.float32)
    overall = float(np.std(noise)) + 1e-6
    h, w = noise.shape
    bs = 32
    blocks = []
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            blk = noise[y:y+bs, x:x+bs]
            if blk.size:
                blocks.append(float(np.std(blk)))
    if not blocks:
        return 0.0
    block_std = float(np.std(blocks))
    val = np.clip((block_std / overall), 0.0, 5.0)
    return float(np.tanh(val * 0.9))


def _laplacian_var(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))


def _skin_smoothness_score(face_bgr: np.ndarray) -> float:
    try:
        if face_bgr is None or face_bgr.size == 0:
            return 0.0
        small = cv2.resize(face_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        smooth = cv2.bilateralFilter(small, d=9, sigmaColor=75, sigmaSpace=75)
        diff = cv2.absdiff(small, smooth).astype(np.float32)
        m = float(np.mean(diff))
        score = np.clip((20.0 - m) / 20.0, 0.0, 1.0)
        return float(score)
    except Exception:
        return 0.0


def _texture_repetition_score(bgr: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        target = 256
        if max(h, w) > target:
            scale = target / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift) + 1e-9
        logm = np.log(mag)

        baseline = cv2.GaussianBlur(logm, (9, 9), 0)
        diff = logm - baseline

        cy, cx = diff.shape[0] // 2, diff.shape[1] // 2
        diff[cy-2:cy+3, cx-2:cx+3] = 0.0

        m = diff.mean()
        s = diff.std() + 1e-9
        peaks = (diff > (m + 1.2 * s)).astype(np.float32)
        peak_count = peaks.sum()
        area = diff.size
        ratio = float(peak_count / (area + 1e-9))
        score = float(np.tanh(ratio * 400.0))
        return score
    except Exception:
        return 0.0


# ---------------- AI probability (images) ----------------

def ai_probability(bgr: np.ndarray, debug: bool = False) -> Tuple[float, Dict]:
    """
    Face-aware image AI probability detector with texture repetition.
    Returns (prob, details).
    """
    if bgr is None or bgr.size == 0:
        return 0.0, {"err": "empty"}

    # global signals
    hf = _highfreq_ratio(bgr)
    ela90 = _ela_inconsistency(bgr, q=90)
    ela75 = _ela_inconsistency(bgr, q=75)
    ela_global = max(ela90, ela75)
    noise_irreg = _noise_irregularity(bgr)
    lap = _laplacian_var(bgr)
    lap_norm = float(np.clip(lap / 400.0, 0.0, 1.0))
    fft_hf = _highfreq_ratio_fft(bgr)
    tex_rep_global = _texture_repetition_score(bgr)

    face = _crop_face_strict(bgr)
    if face is not None:
        try:
            face_resized = cv2.resize(face, (256, 256), interpolation=cv2.INTER_AREA)
        except Exception:
            face_resized = face
        hf_face = _highfreq_ratio(face_resized)
        ela_face = max(_ela_inconsistency(face_resized, q=90),
                       _ela_inconsistency(face_resized, q=75))
        fft_face = _highfreq_ratio_fft(face_resized)
        noise_face = _noise_irregularity(face_resized)
        smooth_face = _skin_smoothness_score(face_resized)
        tex_rep_face = _texture_repetition_score(face_resized)
    else:
        hf_face = 0.0
        ela_face = 0.0
        fft_face = 0.0
        noise_face = 0.0
        smooth_face = 0.0
        tex_rep_face = 0.0

    try:
        hist = cv2.calcHist([bgr], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        hist_entropy = -np.sum([p * np.log2(p + 1e-12) for p in hist])
        s_hist = float(np.clip((5.5 - hist_entropy) / 2.5, 0.0, 1.0))
    except Exception:
        s_hist = 0.0

    s_hf_low = float(np.clip((0.25 - hf) / 0.25, 0.0, 1.0))
    s_hf_high_rep = 0.0
    if hf > 0.80:
        rep = max(tex_rep_global, tex_rep_face)
        s_hf_high_rep = float(np.clip((hf - 0.80) / 0.20, 0.0, 1.0) * rep)
    s_hf = float(np.clip(max(s_hf_low, s_hf_high_rep), 0.0, 1.0))

    s_ela = float(np.clip(max(ela_global, ela_face) * 1.35, 0.0, 1.0))
    s_noise = float(np.clip(max(noise_irreg, noise_face) * 1.02, 0.0, 1.0))
    s_lap = float(np.clip(1.0 - lap_norm, 0.0, 1.0))
    s_fft = float(np.clip(max(fft_hf, fft_face), 0.0, 1.0))
    s_smooth = float(np.clip(smooth_face, 0.0, 1.0))
    s_texture = float(np.clip(max(tex_rep_face, tex_rep_global * 0.8), 0.0, 1.0))

    feats = {
        "s_hf": s_hf,
        "s_ela": s_ela,
        "s_noise": s_noise,
        "s_lap": s_lap,
        "s_hist": s_hist,
        "s_fft": s_fft,
        "s_smooth": s_smooth,
        "s_texture": s_texture,
    }

    w = {
        "s_hf": 0.06,
        "s_ela": 0.38,
        "s_noise": 0.14,
        "s_lap": 0.05,
        "s_hist": 0.05,
        "s_fft": 0.12,
        "s_smooth": 0.18,
        "s_texture": 0.22,
    }

    raw = sum(feats[k] * w.get(k, 0.0) for k in feats)

    signals = sum(1 for v in feats.values() if v >= 0.48)

    amp = 1.0
    if signals >= 4:
        amp = 2.0
    elif signals == 3:
        amp = 1.6
    elif signals == 2:
        amp = 1.3
    elif signals == 1:
        amp = 1.08

    boosted = float(np.clip(raw * amp, 0.0, 1.0))
    final = float(np.clip(boosted ** 0.88, 0.0, 1.0))
    prob = float(np.clip(final, 0.0, 1.0))

    details = {
        "hf": hf,
        "ela90": ela90,
        "ela75": ela75,
        "ela_global": ela_global,
        "hf_face": hf_face,
        "ela_face": ela_face,
        "fft_face": fft_face,
        "noise_irreg": noise_irreg,
        "noise_face": noise_face,
        "lap_var": lap,
        "fft_hf": fft_hf,
        "tex_rep_global": tex_rep_global,
        "tex_rep_face": tex_rep_face,
        "skin_smooth_face": smooth_face,
        "hist_entropy_proxy": s_hist,
        "feats": feats,
        "weights": w,
        "raw": raw,
        "signals": int(signals),
        "amp": float(amp),
        "boosted": boosted,
        "final_prob": prob,
    }

    if debug:
        return prob, details
    return prob, details


# ---------------- identity similarity (geometry-first) ----------------

def _face_quality(bgr: np.ndarray) -> float:
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    var_lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = np.clip(var_lap / 200.0, 0.0, 1.0)
    size_ok = np.clip(min(bgr.shape[:2]) / 120.0, 0.0, 1.0)
    return 0.6 * blur + 0.4 * size_ok


def identity_similarity_pct(bgr_a: np.ndarray, bgr_b: np.ndarray) -> Tuple[float, Dict]:
    meta = {"qa": 0.0, "qb": 0.0, "pairs": 0, "mode": "", "inliers": 0, "good": 0}

    faceA = _crop_face_strict(bgr_a)
    faceB = _crop_face_strict(bgr_b)
    if faceA is None or faceB is None:
        meta["reason"] = "no_face"
        return 0.0, meta

    qa = _face_quality(faceA)
    qb = _face_quality(faceB)
    meta["qa"], meta["qb"] = qa, qb

    faceA = cv2.resize(faceA, (224, 224), interpolation=cv2.INTER_LINEAR)
    faceB = cv2.resize(faceB, (224, 224), interpolation=cv2.INTER_LINEAR)

    orb_s, inliers, good, total = _orb_match_score(faceA, faceB)
    meta["inliers"], meta["good"], meta["pairs"] = int(inliers), int(good), int(inliers)
    meta["mode"] = "orb_only"

    emb_s = None

    if inliers < 18 or orb_s < 0.25:
        orb_s = min(orb_s, 0.25)
        final01 = orb_s
    else:
        final01 = orb_s

    if min(qa, qb) < 0.35:
        final01 = max(0.0, final01 - 0.15)

    pct = float(np.clip(final01, 0.0, 1.0) * 100.0)
    return pct, meta


# ===========================================================
# LOCALIZATION (patch-wise suspiciousness heatmap)
# ===========================================================

def _patch_scores_grid(bgr: np.ndarray, patch: int = 64, stride: int = 32) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    Compute suspiciousness per patch (0..1). Returns:
      - score_grid: (Hk, Wk) float32 grid with per-patch scores.
      - meta: (patch, stride, Hk, Wk) as ints embedded as tuple: (patch,stride,Hk,Wk)
    Uses combination of ela, highfreq, texture repetition, smoothness.
    """
    h, w = bgr.shape[:2]
    if h < patch or w < patch:
        # resize up for patching to avoid single-pixel patches
        scale = max(1, patch // min(h, w))
        bgr = cv2.resize(bgr, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
        h, w = bgr.shape[:2]

    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    Hk = len(ys)
    Wk = len(xs)
    grid = np.zeros((Hk, Wk), dtype=np.float32)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch_img = bgr[y:y+patch, x:x+patch]
            if patch_img.size == 0:
                grid[i, j] = 0.0
                continue
            # compute features: ela, hf, tex rep, smooth
            ela90 = _ela_inconsistency(patch_img, q=90)
            hf = _highfreq_ratio(patch_img)
            tex = _texture_repetition_score(patch_img)
            smooth = _skin_smoothness_score(patch_img)
            lap = _laplacian_var(patch_img)
            # simple suspiciousness transform:
            s_ela = ela90
            s_hf = 0.0
            if hf > 0.75:
                s_hf = hf * tex
            s_smooth = smooth
            s_lap = np.clip((80.0 - lap) / 160.0, 0.0, 1.0)  # lower lap => more suspicious
            # weighted sum (conservative)
            score = 0.38 * s_ela + 0.22 * s_hf + 0.22 * s_smooth + 0.18 * s_lap
            grid[i, j] = float(np.clip(score, 0.0, 1.0))
    return grid, (patch, stride, Hk, Wk)


def localization_heatmap(bgr: np.ndarray, patch: int = 64, stride: int = 32, blur: int = 9) -> Tuple[np.ndarray, Tuple[int,int,int,int], Dict[str,Any]]:
    """
    Compute localization heatmap (resized to image size).
    Returns:
      - heatmap (H,W) float32 0..1 aligned with original image
      - grid_meta tuple (patch, stride, Hk, Wk)
      - info dict: contains 'grid' (Hk x Wk), 'bbox' (x1,y1,x2,y2) of top suspicious region and diagnostic scores
    """
    h0, w0 = bgr.shape[:2]
    grid, meta = _patch_scores_grid(bgr, patch=patch, stride=stride)
    patch_sz, stride_sz, Hk, Wk = meta
    # upsample grid to image size by nearest
    grid_resized = cv2.resize(grid, (w0, h0), interpolation=cv2.INTER_LINEAR)
    # smooth heatmap for nicer visualization
    heat = cv2.GaussianBlur(grid_resized, (blur, blur), 0)
    heat = np.clip(heat, 0.0, 1.0)

    # find top region: threshold by percentile
    flat = heat.ravel()
    if flat.size == 0:
        bbox = (0, 0, w0, h0)
    else:
        thr = np.percentile(flat, 90)
        mask = (heat >= thr).astype(np.uint8)
        # find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, ww, hh = cv2.boundingRect(c)
            # pad bbox a bit
            pad_x = max(4, int(0.03 * w0))
            pad_y = max(4, int(0.03 * h0))
            x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
            x2 = min(w0, x + ww + pad_x); y2 = min(h0, y + hh + pad_y)
            bbox = (x1, y1, x2, y2)
        else:
            # fallback to max location
            idx = int(np.argmax(flat))
            yy, xx = divmod(idx, w0)
            bw = min(w0, patch_sz); bh = min(h0, patch_sz)
            x1 = max(0, xx - bw // 2); y1 = max(0, yy - bh // 2)
            x2 = min(w0, x1 + bw); y2 = min(h0, y1 + bh)
            bbox = (x1, y1, x2, y2)

    info = {
        "grid": grid,
        "meta": {"patch": patch_sz, "stride": stride_sz, "Hk": Hk, "Wk": Wk},
        "bbox": bbox
    }
    return heat.astype(np.float32), meta, info


def overlay_heatmap_on_bgr(bgr: np.ndarray, heat: np.ndarray, alpha: float = 0.45, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay heatmap (0..1 float) on top of bgr image and return BGR overlay image.
    """
    h, w = bgr.shape[:2]
    if heat.shape[:2] != (h, w):
        heat_resized = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        heat_resized = heat
    heat8 = np.uint8(np.clip(heat_resized * 255.0, 0, 255))
    heat_color = cv2.applyColorMap(heat8, colormap)
    overlay = cv2.addWeighted(heat_color, alpha, bgr, 1.0 - alpha, 0)
    return overlay


# ---------------- SIMPLE FEATURE EXTRACTOR FOR TRAINING + MODEL PREDICTION ----------------

def ai_probability_rule_voting(bgr: np.ndarray, debug: bool = False):
    """
    Simplified feature extractor for RandomForest training.
    Returns:
        soft_prob (float)
        votes (int)
        details (dict) - includes 'feats'
    """
    prob, details = ai_probability(bgr, debug=True)
    feats = details.get("feats", {})

    votes = 0
    if feats.get("s_smooth", 0) > 0.55:
        votes += 1
    if feats.get("s_ela", 0) > 0.55:
        votes += 1
    if feats.get("s_texture", 0) > 0.45:
        votes += 1
    if feats.get("s_hf", 0) > 0.65:
        votes += 1

    soft_prob = float(np.clip(prob, 0.0, 1.0))
    details["votes"] = votes

    if debug:
        return soft_prob, votes, details
    return soft_prob, votes, details


__all__ = [
    "ai_probability",
    "identity_similarity_pct",
    "face_embedding",
    "crop_face",
    "ai_probability_rule_voting",
    "localization_heatmap",
    "overlay_heatmap_on_bgr",
]
