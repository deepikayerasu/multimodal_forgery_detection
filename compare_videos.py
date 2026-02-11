# compare_videos.py
# -----------------------------------------------------------
# Video-vs-Video face identity similarity using face_recognition
# No fusion. Pure face-embedding cosine similarity over frames.
# -----------------------------------------------------------

import os
import tempfile
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import face_recognition  # dlib-based
except Exception as e:
    raise RuntimeError(
        "face_recognition is required. Install with: pip install face-recognition dlib"
    ) from e


# ---------- helpers ----------

def _as_path(file_or_path: Union[str, bytes]) -> str:
    """
    Accepts a Streamlit UploadedFile, bytes, or str path and returns a filesystem path.
    If it's bytes or an UploadedFile, writes to a temp file.
    """
    # Streamlit's UploadedFile has .read and .name; treat like bytes
    if hasattr(file_or_path, "read"):
        data = file_or_path.read()
        suffix = os.path.splitext(getattr(file_or_path, "name", "video.mp4"))[-1] or ".mp4"
        f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.write(data)
        f.flush()
        f.close()
        return f.name

    if isinstance(file_or_path, (bytes, bytearray)):
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        f.write(file_or_path)
        f.flush()
        f.close()
        return f.name

    # assume string path
    return str(file_or_path)


def _iter_frames(path: str, every_n: int = 10, max_frames: int = 80) -> Iterable[np.ndarray]:
    """
    Read frames from a video file, return BGR frames every N frames (to reduce work).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return

    total = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            total += 1
            yield frame
            if total >= max_frames:
                break
        idx += 1
    cap.release()


def _face_embed_bgr(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns a 128-D face embedding for the largest face in the frame (if any).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")  # fast; use "cnn" if GPU
    if not boxes:
        return None
    # pick largest box
    hws = [(b[2] - b[0]) * (b[1] - b[3]) for b in boxes]  # (top,right,bottom,left)
    box = boxes[int(np.argmax(hws))]
    encs = face_recognition.face_encodings(rgb, [box])
    return encs[0] if len(encs) else None


def _mean_embedding_from_video(path: str) -> Optional[np.ndarray]:
    """
    Average the face embeddings over sampled frames. Returns None if no face found.
    """
    embs: List[np.ndarray] = []
    for fr in _iter_frames(path, every_n=10, max_frames=80):
        e = _face_embed_bgr(fr)
        if e is not None:
            embs.append(e)
    if not embs:
        return None
    return np.mean(np.stack(embs, axis=0), axis=0)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


# ---------- public API ----------

def compute_face_similarity(
    original_video: Union[str, bytes],
    suspected_video: Union[str, bytes],
) -> float:
    """
    Returns face identity similarity in percent [0..100].
    If a face is not found in either clip, returns 0.0.
    """
    path_a = _as_path(original_video)
    path_b = _as_path(suspected_video)

    emb_a = _mean_embedding_from_video(path_a)
    emb_b = _mean_embedding_from_video(path_b)

    # clean up temp files if we created them
    for p in (path_a, path_b):
        # only remove temp files we created (in tmp)
        if os.path.dirname(p).startswith(tempfile.gettempdir()):
            try:
                os.remove(p)
            except Exception:
                pass

    if emb_a is None or emb_b is None:
        return 0.0

    sim = _cosine_sim(emb_a, emb_b)
    # cosine [-1..1] -> clamp to [0..1] just in case of tiny negatives
    sim01 = max(0.0, min(1.0, (sim + 1.0) / 2.0))  # or simply clamp(sim)
    # empirical: face_recognition encodings usually produce sim in ~0.2..0.9 (dot product)
    # Using a direct clamp is fine for display.
    return sim01 * 100.0
