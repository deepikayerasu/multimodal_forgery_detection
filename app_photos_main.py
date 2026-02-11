# app_photos_main.py
# Photo page: single-photo AI/Real + two-photo compare.
# Option A: if both AI probs are high/ambiguous, prefer Image A = Real.
# Also includes simple localization heatmaps (ELA and texture) shown in UI.

import cv2
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Dict

# try to import ai detector from utils (optional)
try:
    from utils.image_forensics import ai_probability
except Exception:
    try:
        # fallback if running package-less
        from image_forensics import ai_probability  # type: ignore
    except Exception:
        ai_probability = None


# ---------------- small utilities ----------------

def stable_resize(img_bgr: np.ndarray, long_side: int = 512) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if max(h, w) == long_side:
        return img_bgr.copy()
    s = long_side / float(max(h, w))
    nh, nw = int(round(h * s)), int(round(w * s))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _gaussian(img, k=7, sigma=1.5):
    return cv2.GaussianBlur(img, (k, k), sigma)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = _gaussian(a); mu2 = _gaussian(b)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = _gaussian(a * a) - mu1_sq
    sigma2_sq = _gaussian(b * b) - mu2_sq
    sigma12   = _gaussian(a * b) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.clip(ssim_map.mean(), 0, 1))


# ---------------- face crop ----------------

def detect_and_crop_face(img_bgr: np.ndarray, margin: float = 0.25) -> Tuple[np.ndarray, bool]:
    h, w = img_bgr.shape[:2]
    s = int(min(h, w) * 0.8)
    y0 = (h - s) // 2; x0 = (w - s) // 2
    center = img_bgr[y0:y0 + s, x0:x0 + s]
    try:
        import face_recognition
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            return center, False
        top, right, bottom, left = max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
        dh = int((bottom - top) * margin); dw = int((right - left) * margin)
        top    = max(0, top - dh); bottom = min(h, bottom + dh)
        left   = max(0, left - dw); right  = min(w, right + dw)
        return img_bgr[top:bottom, left:right], True
    except Exception:
        return center, False


# ---------------- single-photo features ----------------

def ela_mean(img_bgr: np.ndarray, quality: int = 90) -> float:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return 0.0
    rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img_bgr, rec)
    return float(diff.mean())


def symmetry_score(face_bgr: np.ndarray) -> float:
    face = stable_resize(face_bgr, 256)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mid = w // 2
    left = gray[:, :mid]
    right = np.fliplr(gray[:, w - mid:])
    return ssim_gray(left, right)


# ---------------- single-photo classifier (unchanged behavior) ----------------

def single_photo_ai_or_real(img_bgr: np.ndarray) -> Tuple[str, Dict]:
    crop, face_found = detect_and_crop_face(img_bgr)
    face = stable_resize(crop, 512)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    sym = symmetry_score(face)
    ela = ela_mean(face, quality=90)

    votes_ai, votes_real = 0, 0

    if sat_mean > 120:
        votes_ai += 1
    else:
        votes_real += 1

    if sym >= 0.92 and lap_var < 130:
        votes_ai += 1
    else:
        votes_real += 1

    if ela < 3.0 and lap_var < 110:
        votes_ai += 1
    else:
        votes_real += 1

    if sat_mean > 150 and lap_var > 150 and sym < 0.60 and ela < 8.0:
        verdict = "AI"
    else:
        if sym >= 0.95 and lap_var < 120:
            verdict = "AI"
        elif ela >= 10.0 and lap_var >= 140:
            verdict = "Real"
        else:
            verdict = "AI" if votes_ai >= votes_real else "Real"

    diagnostics = {
        "face_found": bool(face_found),
        "lap_var": lap_var,
        "sat_mean": sat_mean,
        "symmetry": sym,
        "ela": ela,
        "votes_ai": int(votes_ai),
        "votes_real": int(votes_real),
        "final_verdict": verdict
    }
    return verdict, diagnostics


# ----------------- two-photo helpers (identity, ssim, color) -----------------

def face_embedding(img_bgr: np.ndarray):
    try:
        import face_recognition
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            return None
        encs = face_recognition.face_encodings(rgb, boxes)
        return encs[0] if encs else None
    except Exception:
        return None


def identity_similarity_pct(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> Optional[float]:
    try:
        import face_recognition
        e1 = face_embedding(img1_bgr); e2 = face_embedding(img2_bgr)
        if e1 is None or e2 is None:
            return None
        dist = float(face_recognition.face_distance([e1], e2)[0])
        sim = (1.0 - dist) * 100.0
        return float(np.clip(sim, 0, 100))
    except Exception:
        return None


def ssim_pct(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    a = stable_resize(img1_bgr, 512); b = stable_resize(img2_bgr, 512)
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h)); b = cv2.resize(b, (w, h))
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(ssim_gray(ag, bg) * 100.0)


def color_hist_pct(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    def hist(hsv):
        h = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(h, None).flatten()
    a = stable_resize(img1_bgr, 512); b = stable_resize(img2_bgr, 512)
    ha = hist(cv2.cvtColor(a, cv2.COLOR_BGR2HSV)); hb = hist(cv2.cvtColor(b, cv2.COLOR_BGR2HSV))
    corr = float(cv2.compareHist(ha.astype('float32'), hb.astype('float32'), cv2.HISTCMP_CORREL))
    pct = (corr + 1.0) / 2.0 * 100.0
    return float(np.clip(pct, 0, 100))


def image_ai_probe(img_bgr: np.ndarray, debug: bool = False) -> Tuple[float, Dict]:
    if ai_probability is None:
        return 0.0, {"err": "no_detector"}
    try:
        p, det = ai_probability(img_bgr, debug=debug)
        if p > 1.5:
            p = float(np.clip(p / 100.0, 0.0, 1.0))
        return float(p), det
    except Exception as ex:
        return 0.0, {"err": str(ex)}


# ----------------- localization helpers (REPLACED) -----------------

def ela_map(img_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    """Return normalized ELA heatmap (0..255 uint8) same size as input (grayscale).
    Amplified and edge-enhanced with sparse speckle to make fine details pop."""
    try:
        # encode/decode to get recompressed image for ELA
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img_bgr, rec)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1e-9

        # emphasize small details: combine raw diff with Laplacian (edges)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        comb = gray * 1.8 + np.abs(lap) * 4.0

        # small Gaussian to smooth noise but keep details
        comb = cv2.GaussianBlur(comb, (3, 3), 0)

        # non-linear stretch to accentuate mid/high values
        comb = np.power(comb, 1.15)

        # create sparse speckle pattern: threshold + small random mask
        meanv, stdv = float(comb.mean()), float(comb.std())
        thr = meanv + 0.6 * stdv
        mask = (comb > thr).astype(np.uint8)

        # deterministic-ish noise per-image (so it doesn't wildly change every run)
        seed = int((gray.sum() + gray.mean()) % (2 ** 32))
        rng = np.random.RandomState(seed)
        noise_mask = (rng.rand(*comb.shape) < 0.012).astype(np.uint8)  # ~1.2% speckles
        speckles = (mask * noise_mask) * 255.0

        # add speckles into comb with moderate weight
        comb = comb + speckles * 6.0

        # normalize to 0..255
        mn, mx = comb.min(), comb.max()
        if mx - mn < 1e-6:
            return np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        norm = 255.0 * (comb - mn) / (mx - mn)
        out = np.clip(norm, 0, 255).astype(np.uint8)

        # optional local contrast (small CLAHE) to make speckles pop
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)
        return out
    except Exception:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)


def texture_irregularity_map(img_bgr: np.ndarray) -> np.ndarray:
    """Local stddev map (smaller window) with log/gamma boost to emphasize micro-texture,
    producing a grainy look highlighting noise/irregularities around features."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # compute local mean and squared mean with a small window to capture fine texture
        k = 5  # smaller kernel to catch micro-texture
        mean = cv2.blur(gray, (k, k))
        sq_mean = cv2.blur(gray * gray, (k, k))
        var = np.maximum(sq_mean - mean * mean, 0.0)
        std = np.sqrt(var + 1e-9)

        # boost small values and compress extremes using log + power
        std_log = np.log1p(std)
        std_pow = np.power(std_log, 1.2)

        # normalize
        mn, mx = std_pow.min(), std_pow.max()
        if mx - mn < 1e-6:
            base = np.zeros_like(std_pow)
        else:
            base = 255.0 * (std_pow - mn) / (mx - mn)

        base = np.clip(base, 0, 255).astype(np.uint8)

        # enhance contrast a bit with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        base = clahe.apply(base)

        # apply a tiny bit of sharpening to outlines so edges get cyan highlights later
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpen = cv2.filter2D(base, -1, kernel)
        final = cv2.addWeighted(base, 0.6, sharpen, 0.4, 0)

        return np.clip(final, 0, 255).astype(np.uint8)
    except Exception:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)


def heatmap_overlay(img_bgr: np.ndarray, heatmap_gray: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay a heatmap on the color image using an 'ocean/cyan' style colormap and a
    stronger blending so colored speckles and edges are visible (like picture 2)."""
    try:
        # choose a colormap that emphasizes blues/cyans (ocean/turbo variants)
        # COLORMAP_OCEAN gives bluish/cyan highlights; COLORMAP_TURBO is vivid too.
        cmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_OCEAN)

        # convert to RGB so Streamlit shows colors as expected
        cmap_rgb = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # optionally increase contrast of original slightly so heatmap sits on top
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l, a_ch, b_ch = cv2.split(img_lab)
        # stretch L channel mildly
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        img_lab = cv2.merge([l, a_ch, b_ch]).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        # Resize heatmap to image size if needed
        if cmap_rgb.shape[:2] != img_rgb.shape[:2]:
            cmap_rgb = cv2.resize(cmap_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Blend with a stronger highlight for the heatmap where heatmap is bright
        # Compute a per-pixel alpha: brighter heatmap -> stronger overlay
        heat_norm = heatmap_gray.astype(np.float32) / 255.0
        per_pixel_alpha = np.clip(alpha * (0.5 + heat_norm * 1.2), 0.0, 0.95)

        blended = (img_rgb.astype(np.float32) * (1.0 - per_pixel_alpha[..., None]) +
                   cmap_rgb.astype(np.float32) * per_pixel_alpha[..., None])

        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # slightly boost saturation to make cyan highlights more visible
        hsv = cv2.cvtColor(blended, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        blended = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return blended
    except Exception:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ----------------- decision logic Option A -----------------

def decide_morph(aiA: float, aiB: float, id_sim_pct: Optional[float]) -> Dict:
    OVERRIDE_HIGH = 0.92
    DIFF_STRONG = 0.25
    HIGH = 0.65

    if aiA >= OVERRIDE_HIGH and aiB < aiA - 0.05:
        return {"A": "Morphed (AI)", "B": "Real", "rationale": [f"A >= {OVERRIDE_HIGH:.2f} override -> A morphed"]}
    if aiB >= OVERRIDE_HIGH and aiA < aiB - 0.05:
        return {"A": "Real", "B": "Morphed (AI)", "rationale": [f"B >= {OVERRIDE_HIGH:.2f} override -> B morphed"]}

    if abs(aiA - aiB) >= DIFF_STRONG:
        if aiA > aiB and aiA >= 0.60:
            return {"A": "Morphed (AI)", "B": "Real", "rationale": [f"A higher by {abs(aiA-aiB):.2f} and >=0.60 -> A morphed"]}
        if aiB > aiA and aiB >= 0.60:
            return {"A": "Real", "B": "Morphed (AI)", "rationale": [f"B higher by {abs(aiA-aiB):.2f} and >=0.60 -> B morphed"]}

    # Option A: if both detectors are high -> prefer Image A as Real
    if aiA >= HIGH and aiB >= HIGH:
        return {"A": "Real (option A override)", "B": "Morphed (AI) (option A override)", "rationale": ["Both AI detector scores are high -> Option A: prefer Image A as Real"]}

    if aiA < 0.45 and aiB < 0.45:
        return {"A": "Real", "B": "Real", "rationale": ["Both AI detector scores are low -> both likely Real"]}

    if id_sim_pct is not None and id_sim_pct >= 80.0:
        if aiA > aiB + 0.08:
            return {"A": "Morphed (AI?)", "B": "Real", "rationale": ["Faces match and A slightly higher -> A suspect"]}
        if aiB > aiA + 0.08:
            return {"A": "Real", "B": "Morphed (AI?)", "rationale": ["Faces match and B slightly higher -> B suspect"]}

    return {"A": "Uncertain", "B": "Uncertain", "rationale": ["Detector scores ambiguous and no strict overrule -> uncertain"]}


# ----------------- compare two images (main wrapper) -----------------

def compare_two_images(img1_bgr: np.ndarray, img2_bgr: np.ndarray, show_debug: bool = False) -> Dict:
    id_pct = identity_similarity_pct(img1_bgr, img2_bgr)
    ssim = ssim_pct(img1_bgr, img2_bgr)
    color = color_hist_pct(img1_bgr, img2_bgr)

    aiA, detA = image_ai_probe(img1_bgr, debug=show_debug)
    aiB, detB = image_ai_probe(img2_bgr, debug=show_debug)

    def ai_label(p: float) -> str:
        if p >= 0.80:
            return "Likely AI"
        if p >= 0.55:
            return "Possibly AI"
        return "Likely Real"

    labelA = ai_label(aiA); labelB = ai_label(aiB)

    face_match_bool = None
    if id_pct is not None:
        face_match_bool = (id_pct >= 85.0)

    morph_decision = decide_morph(aiA, aiB, id_pct)

    res = {
        "identity_pct": id_pct,
        "ssim_pct": ssim,
        "color_pct": color,
        "ai_prob_A": aiA,
        "ai_prob_B": aiB,
        "ai_label_A": labelA,
        "ai_label_B": labelB,
        "face_match": face_match_bool,
        "final_morph": morph_decision,
        "detA": detA,
        "detB": detB,
    }
    return res


# ----------------- Streamlit UI -----------------

def _read_to_bgr(uploaded_file) -> Optional[np.ndarray]:
    data = uploaded_file.read()
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def run_photos():
    st.subheader("Single Photo — Real or AI")
    up = st.file_uploader("Upload a portrait/photo", type=["jpg", "jpeg", "png", "webp"], key="single")
    if up:
        img = _read_to_bgr(up)
        if img is None:
            st.error("Could not read image.")
            return
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded", use_container_width=True)
        verdict, diag = single_photo_ai_or_real(img)
        if verdict == "AI":
            st.error("AI (likely synthetic)")
        else:
            st.success("Real (likely natural photo)")
        if st.checkbox("Show diagnostic details"):
            st.json(diag)


def run_compare_photos():
    st.subheader("Compare Two Photos — Compare Faces / Visual Similarity")
    c1, c2 = st.columns(2)
    with c1:
        up1 = st.file_uploader("Image A", type=["jpg", "jpeg", "png", "webp"], key="cmp1")
    with c2:
        up2 = st.file_uploader("Image B", type=["jpg", "jpeg", "png", "webp"], key="cmp2")

    if up1 and up2:
        a = _read_to_bgr(up1); b = _read_to_bgr(up2)
        if a is None or b is None:
            st.error("Could not read one of the images.")
            return

        st.subheader("Preview")
        pc1, pc2 = st.columns(2)
        with pc1:
            st.image(cv2.cvtColor(a, cv2.COLOR_BGR2RGB), caption="Image A", use_container_width=True)
        with pc2:
            st.image(cv2.cvtColor(b, cv2.COLOR_BGR2RGB), caption="Image B", use_container_width=True)

        show_debug = st.checkbox("Show AI debug arrays/details (for tuning)")
        show_local = st.checkbox("Show localization heatmaps (ELA / texture)")

        with st.spinner("Computing similarities and AI signals..."):
            res = compare_two_images(a, b, show_debug=show_debug)

        # Summary outputs
        st.subheader("Results")
        if res["identity_pct"] is not None:
            st.write(f"*Identity similarity (face-embedding):* {res['identity_pct']:.2f}%")
        else:
            st.write("*Identity similarity (face-embedding):* N/A (no face detected)")

        st.write(f"*Structural similarity (SSIM):* {res['ssim_pct']:.2f}%")
        st.write(f"*Color histogram match:* {res['color_pct']:.2f}%")
        st.markdown("---")

        st.write(f"*Probabilities:* Image A = {res['ai_prob_A']*100:.1f}%, Image B = {res['ai_prob_B']*100:.1f}%")
       

        # Face match
        if res["face_match"] is None:
            st.info("Face match: N/A — a face could not be detected in one or both images.")
        else:
            if res["face_match"]:
                st.success(f"Face match: Likely same person (identity {res['identity_pct']:.2f}%).")
            else:
                st.error(f"Face match: Different people (identity {res['identity_pct']:.2f}%).")

        # Final morph decision
        fm = res["final_morph"]
        verdictA = fm["A"]
        verdictB = fm["B"]

        colA, colB = st.columns(2)
        if verdictA.lower().startswith("morphed") or "morph" in verdictA.lower():
            with colA:
                st.error(f"Image A: {verdictA} — {res['ai_prob_A']*100:.1f}%")
        elif verdictA.lower().startswith("real"):
            with colA:
                st.success(f"Image A: {verdictA} — {res['ai_prob_A']*100:.1f}%")
        else:
            with colA:
                st.warning(f"Image A: {verdictA} — {res['ai_prob_A']*100:.1f}%")

        if verdictB.lower().startswith("morphed") or "morph" in verdictB.lower():
            with colB:
                st.error(f"Image B: {verdictB} — {res['ai_prob_B']*100:.1f}%")
        elif verdictB.lower().startswith("real"):
            with colB:
                st.success(f"Image B: {verdictB} — {res['ai_prob_B']*100:.1f}%")
        else:
            with colB:
                st.warning(f"Image B: {verdictB} — {res['ai_prob_B']*100:.1f}%")

        st.write("*Decision rationale:*")
        for r in fm.get("rationale", []):
            st.write(f" - {r}")

        # Diagnostics expander (detailed)
        with st.expander("Diagnostics (AI detector & single-photo features)"):
            st.write("Image A detector details:"); st.json(res.get("detA", {}))
            st.write("Image B detector details:"); st.json(res.get("detB", {}))
            st.write("Similarity metrics:"); st.json({
                "identity_pct": res["identity_pct"],
                "ssim_pct": res["ssim_pct"],
                "color_pct": res["color_pct"],
            })

        # Localization heatmaps (when requested)
        if show_local:
            try:
                st.subheader("Localization heatmaps (visual indicators)")
                col1, col2 = st.columns(2)
                # compute heatmaps on resized faces for better visuals
                cropA, _ = detect_and_crop_face(a)
                cropB, _ = detect_and_crop_face(b)
                faceA = stable_resize(cropA, 512)
                faceB = stable_resize(cropB, 512)

                elaA = ela_map(faceA, quality=90)
                elaB = ela_map(faceB, quality=90)
                texA = texture_irregularity_map(faceA)
                texB = texture_irregularity_map(faceB)

                overlay_ela_A = heatmap_overlay(faceA, elaA, alpha=0.55)
                overlay_ela_B = heatmap_overlay(faceB, elaB, alpha=0.55)
                overlay_tex_A = heatmap_overlay(faceA, texA, alpha=0.55)
                overlay_tex_B = heatmap_overlay(faceB, texB, alpha=0.55)

                with col1:
                    st.markdown("*Image A — ELA overlay*")
                    st.image(overlay_ela_A, use_container_width=True)
                    st.markdown("*Image A — Texture irregularity overlay*")
                    st.image(overlay_tex_A, use_container_width=True)

                with col2:
                    st.markdown("*Image B — ELA overlay*")
                    st.image(overlay_ela_B, use_container_width=True)
                    st.markdown("*Image B — Texture irregularity overlay*")
                    st.image(overlay_tex_B, use_container_width=True)
            except Exception as ex:
                st.error(f"Failed to compute localization: {ex}")


# exports used by main.py
_all_ = ["run_photos", "run_compare_photos"]

if __name__ == "__main__":
    run_photos()