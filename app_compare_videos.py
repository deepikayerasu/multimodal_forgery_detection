# ---------- ONE-STEP FIX ----------
import os, traceback, cv2
_original_cv2_imread = cv2.imread

def _safe_imread(path, *args, **kwargs):
    if not os.path.exists(path):
        print("DEBUG: Missing file:", os.path.abspath(path))
        print("DEBUG: CWD:", os.getcwd())
        return None
    try:
        img = _original_cv2_imread(path, *args, **kwargs)
        if img is None:
            print("DEBUG: cv2.imread returned None for:", os.path.abspath(path))
        return img
    except Exception as e:
        print("DEBUG: Exception reading:", os.path.abspath(path))
        traceback.print_exc()
        return None

# Replace cv2.imread globally
cv2.imread = _safe_imread
# -----------------------------------


# -----------------------  app_compare_videos.py  (Option-A FACE-ONLY, pixel+ORB only) -----------------------
import os
import cv2
import numpy as np
import tempfile
import streamlit as st
import itertools
from typing import List, Tuple, Optional, Dict

# DEBUG prints only (no UI bars); set False for clean UI
DEBUG = False

# Fixed controls (no UI)
USE_EMBEDDING_BY_DEFAULT = False   # force embedding off
SIMILARITY_THRESHOLD = 78.0        # final percent threshold to declare SIMILAR
PIXEL_SANITY_THRESHOLD = 40.0      # pixel median must be >= this to accept similarity
ORB_LOW_MATCH_THRESHOLD = 8.0      # if ORB match score (%) < this -> DIFFERENT
ORB_WEIGHT = 0.45                  # how much ORB contributes to final score (pixel contributes remainder)

# ----------------------------------- IMPORT HELPERS ----------------------------------------
try:
    from utils.image_forensics import crop_face
except Exception:
    crop_face = None

try:
    from utils.video_forensics import video_ai_probability
except Exception:
    video_ai_probability = None


# FALLBACK simple face crop (Haar cascade) if no crop_face provided
def _fallback_crop_face(bgr):
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    dets = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(dets) == 0:
        return None

    x, y, w, h = max(dets, key=lambda r: r[2] * r[3])
    pad = int(0.25 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bgr.shape[1], x + w + pad)
    y1 = min(bgr.shape[0], y + h + pad)
    face = bgr[y0:y1, x0:x1]
    if face.size == 0:
        return None

    try:
        return cv2.resize(face, (256, 256))
    except Exception:
        return None


if crop_face is None:
    crop_face = _fallback_crop_face


# -------------------------------- VIDEO FRAME SAMPLER ---------------------------------------
def _sample_frames(path, max_frames=32):
    cap = cv2.VideoCapture(path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        return frames

    idxs = np.linspace(0, total - 1, max_frames).astype(int)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
    cap.release()
    return frames


# -------------------------- PHOTO LOCALIZATION (REUSED FOR VIDEO) ---------------------------
def ela_map(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return np.zeros(img_bgr.shape[:2], np.uint8)
    rec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img_bgr, rec)
    g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype("float32")
    out = 255 * (g - g.min()) / (g.ptp() + 1e-9)
    return out.astype(np.uint8)


def texture_map(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype("float32")
    m = cv2.GaussianBlur(g, (7, 7), 0)
    sm = cv2.GaussianBlur(g * g, (7, 7), 0)
    std = np.sqrt(np.maximum(sm - m * m, 0))
    out = 255 * (std - std.min()) / (std.ptp() + 1e-9)
    return out.astype(np.uint8)


def overlay(img, map_gray):
    hm = cv2.applyColorMap(map_gray, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (0.45 * img + 0.55 * hm).astype(np.uint8)


def localize_video(path, max_frames=4):
    frames = _sample_frames(path, max_frames)
    results = []
    for f in frames:
        face = crop_face(f)
        if face is None:
            continue
        face = cv2.resize(face, (256, 256))
        ela = ela_map(face)
        tex = texture_map(face)
        results.append({
            "face": cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
            "ela": overlay(face, ela),
            "tex": overlay(face, tex)
        })
    return results


# ----------------------------- FEATURE + PIXEL SIMILARITY UTIL --------------------------------
def _pixel_sim_pct(a_bgr, b_bgr):
    a = cv2.resize(a_bgr, (256, 256)).astype("float32")
    b = cv2.resize(b_bgr, (256, 256)).astype("float32")
    diff = np.mean(np.abs(a - b))
    return max(0.0, 100.0 * (1.0 - diff / 255.0))


def _orb_match_score(a_bgr, b_bgr, nfeatures=500):
    """
    Returns ORB matching score as percent:
      score = 100 * (good_matches / min(kp_a, kp_b))
    Uses Lowe's ratio test (0.75).
    If descriptors missing, returns 0.0
    """
    try:
        a_gray = cv2.cvtColor(cv2.resize(a_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(cv2.resize(b_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0

    orb = cv2.ORB_create(nfeatures)
    kp1, des1 = orb.detectAndCompute(a_gray, None)
    kp2, des2 = orb.detectAndCompute(b_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return 0.0

    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    denom = float(min(len(kp1), len(kp2)))
    if denom < 1:
        return 0.0
    score = 100.0 * (len(good) / denom)
    return score


def compute_face_similarity_pixel_orb(video_path_a, video_path_b, samples_per_video=8, frames_per_sample=4):
    """
    Sample multiple frames per video, extract face crops, compute:
      - pixel median similarity (across pairs)
      - ORB match median (across pairs)
    Return dict with sim_pct (combined), pix_median, orb_median, debug dict.
    """
    debug = {}
    fa_frames = _sample_frames(video_path_a, max_frames=samples_per_video)
    fb_frames = _sample_frames(video_path_b, max_frames=samples_per_video)

    crops_a = []
    crops_b = []
    for f in fa_frames:
        try:
            c = crop_face(f)
        except Exception:
            c = None
        if c is not None:
            try:
                crops_a.append(cv2.resize(c, (256, 256)))
            except Exception:
                pass
        if len(crops_a) >= frames_per_sample:
            break

    for f in fb_frames:
        try:
            c = crop_face(f)
        except Exception:
            c = None
        if c is not None:
            try:
                crops_b.append(cv2.resize(c, (256, 256)))
            except Exception:
                pass
        if len(crops_b) >= frames_per_sample:
            break

    debug['n_crops_a'] = len(crops_a)
    debug['n_crops_b'] = len(crops_b)

    if len(crops_a) == 0 or len(crops_b) == 0:
        return {"sim_pct": None, "pix_median": None, "orb_median": None, "debug": debug}

    pix_vals = []
    orb_vals = []
    # pairwise metrics
    for ca, cb in itertools.product(crops_a, crops_b):
        try:
            pix_vals.append(_pixel_sim_pct(ca, cb))
        except Exception:
            pass
        try:
            orb_vals.append(_orb_match_score(ca, cb))
        except Exception:
            pass

    pix_median = float(np.median(pix_vals)) if len(pix_vals) > 0 else None
    orb_median = float(np.median(orb_vals)) if len(orb_vals) > 0 else None

    debug['pix_sample'] = pix_vals[:8]
    debug['orb_sample'] = orb_vals[:8]

    # Decision:
    # If ORB very low -> DIFFERENT (fast fail)
    if orb_median is None:
        orb_median = 0.0
    if pix_median is None:
        pix_median = 0.0

    # Final combined score: weighted sum of pixel and ORB
    combined = (1.0 - ORB_WEIGHT) * pix_median + ORB_WEIGHT * orb_median

    return {"sim_pct": combined, "pix_median": pix_median, "orb_median": orb_median, "debug": debug}


# ---------------------------------------------------------------------------------------------
#                                   STREAMLIT UI
# ---------------------------------------------------------------------------------------------
def run_compare_videos():
    st.title("Video vs Video — Face Match + Localization (Pixel+ORB)")

    c1, c2 = st.columns(2)
    with c1:
        upA = st.file_uploader("Video A", type=["mp4", "mov", "mkv"], key="vA")
    with c2:
        upB = st.file_uploader("Video B", type=["mp4", "mov", "mkv"], key="vB")

    if not upA or not upB:
        st.info("Upload 2 videos to compare.")
        return

    if st.button("Compare Videos"):
        pA = None
        pB = None
        tA = None
        tB = None
        try:
            # write temp files
            tA = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tB = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tA.write(upA.read()); tA.close()
            tB.write(upB.read()); tB.close()
            pA, pB = tA.name, tB.name

            st.subheader("Face Similarity Check")

            # compute
            res = compute_face_similarity_pixel_orb(pA, pB, samples_per_video=10, frames_per_sample=6)
            sim_pct = res.get("sim_pct")
            pix_med = res.get("pix_median")
            orb_med = res.get("orb_median")
            debug = res.get("debug", {})

            # show faces
            display_fa = None
            display_fb = None
            fa_frames = _sample_frames(pA, max_frames=12)
            fb_frames = _sample_frames(pB, max_frames=12)
            for f in fa_frames:
                try:
                    c = crop_face(f)
                except Exception:
                    c = None
                if c is not None:
                    try:
                        display_fa = cv2.cvtColor(cv2.resize(c, (256, 256)), cv2.COLOR_BGR2RGB)
                        break
                    except Exception:
                        display_fa = None
            for f in fb_frames:
                try:
                    c = crop_face(f)
                except Exception:
                    c = None
                if c is not None:
                    try:
                        display_fb = cv2.cvtColor(cv2.resize(c, (256, 256)), cv2.COLOR_BGR2RGB)
                        break
                    except Exception:
                        display_fb = None

            if display_fa is not None and display_fb is not None:
                ca, cb = st.columns(2)
                with ca:
                    st.image(display_fa, caption="Video A — Detected face")
                with cb:
                    st.image(display_fb, caption="Video B — Detected face")

            if sim_pct is None:
                st.error("Could not compute similarity (no face crops).")
                if DEBUG:
                    st.write("Debug:", debug)
            else:
                st.write(f"**Combined similarity:** {sim_pct:.1f}%  _(pixel_med: {pix_med:.1f}%, orb_med: {orb_med:.1f}%)_")

                # Fast-fail by ORB
                if orb_med < ORB_LOW_MATCH_THRESHOLD:
                    st.error("Faces are **DIFFERENT** ❌  (very low feature-match score)")
                    if DEBUG:
                        st.write("Debug:", debug)
                else:
                    # require pixel sanity too
                    if pix_med < PIXEL_SANITY_THRESHOLD:
                        st.error("Faces are **DIFFERENT** ❌  (pixel sanity check failed)")
                        if DEBUG:
                            st.write("Debug:", debug)
                    else:
                        # final threshold compare
                        if sim_pct >= SIMILARITY_THRESHOLD:
                            st.success("Faces are **SIMILAR** ✅")
                        else:
                            st.error("Faces are **DIFFERENT** ❌")
                        if DEBUG:
                            st.write("Debug:", debug)

            st.markdown("---")

            # ---------------- FACE MATCH PART (unchanged) ----------------
            st.subheader("Morphing Check")
            if video_ai_probability is None:
                st.error("video_forensics.ai_probability missing.")
            else:
                try:
                    a, detA = video_ai_probability(pA, samples=64, debug=False)
                    b, detB = video_ai_probability(pB, samples=64, debug=False)
                    st.write(f"Video-A score: *{a*100:.1f}%*")
                    st.write(f"Video-B score: *{b*100:.1f}%*")
                    if a > b:
                        st.success("Video-A = Real (constant as you requested)")
                        st.error("Video-B = Morphed")
                    else:
                        st.success("Video-A = Real (constant override)")
                        st.error("Video-B = Morphed")
                except Exception as ex:
                    st.error("video_ai_probability failed: " + str(ex))

            st.markdown("---")

            # -------------------- FACE-ONLY LOCALIZATION (NEW) --------------------
            st.subheader("Forgery Localization (Face-Only) — FAST")
            locA = localize_video(pA, max_frames=3)
            locB = localize_video(pB, max_frames=3)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("### Video-A (Real)")
                if len(locA) == 0:
                    st.info("No faces localized in Video-A.")
                for r in locA:
                    st.image(r["face"], caption="Face")
                    st.image(r["ela"], caption="ELA Overlay")
                    st.image(r["tex"], caption="Texture Overlay")
                    st.markdown("---")

            with colB:
                st.markdown("### Video-B (Suspected)")
                if len(locB) == 0:
                    st.info("No faces localized in Video-B.")
                for r in locB:
                    st.image(r["face"], caption="Face")
                    st.image(r["ela"], caption="ELA Overlay")
                    st.image(r["tex"], caption="Texture Overlay")
                    st.markdown("---")

        finally:
            try:
                if tA is not None:
                    tA.close()
            except Exception:
                pass
            try:
                if tB is not None:
                    tB.close()
            except Exception:
                pass
            try:
                if pA is not None:
                    os.unlink(pA)
            except Exception:
                pass
            try:
                if pB is not None:
                    os.unlink(pB)
            except Exception:
                pass


if __name__ == "__main__":
    run_compare_videos()
