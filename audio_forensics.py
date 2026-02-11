# utils/audio_forensics.py
# Permissive pitch extraction + spectral fallback TTS heuristics.
# Added pitch-quant hard rule to catch highly-quantized TTS like ElevenLabs.

from typing import Tuple, Dict
import numpy as np
import librosa
import soundfile as sf

def _safe_load(path: str, sr: int = 22050):
    y, orig_sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if orig_sr != sr:
        try:
            y = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
        except Exception:
            try:
                import resampy
                y = resampy.resample(y.astype(np.float32), orig_sr, sr)
            except Exception:
                import numpy as _np
                duration = y.shape[0] / float(orig_sr)
                new_len = int(round(duration * sr))
                if new_len <= 0:
                    return y.astype(np.float32), sr
                old_idx = _np.linspace(0, 1, num=y.shape[0])
                new_idx = _np.linspace(0, 1, num=new_len)
                y = _np.interp(new_idx, old_idx, y).astype(_np.float32)
    return y.astype(np.float32), sr


def _try_extract_f0(y: np.ndarray, sr: int, frame_length: int = 4096, hop_length: int = 128):
    """
    More permissive pitch extraction:
      - larger frame_length + smaller hop for better resolution,
      - wider fmin/fmax to cover many voices,
      - fallback to an energy-based proxy if YIN/PYIN fail.
    Returns: (f0_array_or_empty, voiced_ratio, method_name)
    """
    # Try YIN (permissive voiced threshold)
    try:
        f0 = librosa.yin(y=y, fmin=40, fmax=800, sr=sr,
                         frame_length=frame_length, hop_length=hop_length)
        voiced = ~np.isnan(f0)
        voiced_ratio = float(np.mean(voiced)) if f0.size else 0.0
        if voiced_ratio > 0.03:
            return f0, voiced_ratio, "yin"
    except Exception:
        f0 = None

    # Try PYIN
    try:
        f0_pyin, voiced_flag, voiced_probs = librosa.pyin(y=y, fmin=40, fmax=800, sr=sr,
                                                          frame_length=frame_length, hop_length=hop_length)
        if f0_pyin is not None:
            voiced = ~np.isnan(f0_pyin)
            voiced_ratio = float(np.mean(voiced)) if f0_pyin.size else 0.0
            if voiced_ratio > 0.02:
                return f0_pyin, voiced_ratio, "pyin"
    except Exception:
        pass

    # Energy-based / centroid proxy on strong frames
    try:
        S = np.abs(librosa.stft(y=y, n_fft=frame_length, hop_length=hop_length))
        # compute rms per-frame from waveform (safe) using same frame/hop
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).ravel()
        strong = rms_frames > (np.median(rms_frames) * 0.6) if rms_frames.size else np.array([])
        if strong.size and float(strong.mean()) > 0.02:
            centroid = librosa.feature.spectral_centroid(S=S, sr=sr).ravel()
            f0_est = np.clip(centroid, 40.0, 800.0)
            f0 = np.full((len(f0_est),), np.nan)
            # ensure boolean indexing length matches f0_est length
            if strong.shape[0] == f0.shape[0]:
                f0[strong] = f0_est[strong]
            else:
                # if shapes mismatch, map via minimum length
                m = min(strong.shape[0], f0.shape[0])
                if m > 0:
                    f0[:m][strong[:m]] = f0_est[:m][strong[:m]]
            voiced_ratio = float(strong.mean())
            return f0, voiced_ratio, "energy_proxy"
    except Exception:
        pass

    return np.array([]), 0.0, "none"


def analyze_audio_for_ai(path: str, sr: int = 22050) -> Tuple[float, Dict]:
    """
    Returns (prob_ai_0_1, details). Uses permissive f0 extraction and a spectral fallback rule.
    """
    y, sr = _safe_load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=40)

    if y.size < sr // 2:
        return 0.0, {"err": "audio_too_short"}

    # STFT params
    hop = 128
    frame_length = 4096

    # --- pitch extraction with permissive settings
    f0, voiced_ratio, f0_method = _try_extract_f0(y, sr, frame_length=frame_length, hop_length=hop)
    if f0 is None or f0.size == 0:
        f0_mean = 0.0
        f0_std = 0.0
        jitter = 1.0
        pitch_quant_ratio = 1.0
    else:
        f0_clean = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_clean)) if f0_clean.size else 0.0
        f0_std = float(np.std(f0_clean)) if f0_clean.size else 0.0
        if f0_clean.size > 2:
            diffs = np.abs(np.diff(f0_clean))
            jitter = float(np.mean(diffs / (f0_clean[:-1] + 1e-9)))
        else:
            jitter = 1.0
        # pitch quantization: ratio of unique 1Hz bins to voiced frames
        if f0_clean.size >= 6:
            bins = np.round(f0_clean)
            unique_ratio = float(np.unique(bins).size) / float(bins.size)
            pitch_quant_ratio = float(unique_ratio)
        else:
            pitch_quant_ratio = 1.0

    # --- STFT-based features (use same frame/hop)
    S = np.abs(librosa.stft(y=y, n_fft=frame_length, hop_length=hop))

    # FIX: compute RMS from waveform with same frame_length/hop to avoid n_fft/frame_length mismatch
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop).ravel()
    rms_std = float(np.std(rms)) if rms.size else 0.0
    rms_mean = float(np.mean(rms)) if rms.size else 0.0

    flat = librosa.feature.spectral_flatness(S=S).ravel()
    flat_mean = float(np.mean(flat)) if flat.size else 0.0

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).ravel()
    centroid_std = float(np.std(centroid)) if centroid.size else 0.0
    centroid_mean = float(np.mean(centroid)) if centroid.size else 0.0

    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop).ravel()
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
    zcr_std = float(np.std(zcr)) if zcr.size else 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    mfcc_std = float(np.mean(np.std(mfcc, axis=1))) if mfcc.size else 0.0
    mfcc_mean = float(np.mean(np.mean(mfcc, axis=1))) if mfcc.size else 0.0

    # harmonic-to-noise ratio (approx)
    try:
        y_harm = librosa.effects.harmonic(y=y)
        eps = 1e-9
        energy_h = np.sum(y_harm ** 2) + eps
        energy_res = np.sum((y - y_harm) ** 2) + eps
        hnr_db = 10.0 * np.log10(energy_h / energy_res + eps)
    except Exception:
        hnr_db = 0.0

    # Map to AI-like signals (0..1 where higher -> more TTS-like)
    ai_jitter = 1.0 - float(np.clip(jitter / 0.03, 0.0, 1.0))
    ai_f0 = 1.0 - float(np.clip(f0_std / 30.0, 0.0, 1.0))
    ai_hnr = float(np.clip((hnr_db + 5.0) / 30.0, 0.0, 1.0))
    ai_flat = float(np.clip(flat_mean / 0.55, 0.0, 1.0))
    ai_rms = 1.0 - float(np.clip(rms_std / 0.08, 0.0, 1.0))
    ai_mfcc = 1.0 - float(np.clip(mfcc_std / 18.0, 0.0, 1.0))
    ai_zcr = 1.0 - float(np.clip(zcr_std / 0.02, 0.0, 1.0))
    ai_centroid = 1.0 - float(np.clip(centroid_std / (sr * 0.01), 0.0, 1.0))
    ai_pitch_quant = 1.0 - float(np.clip(pitch_quant_ratio, 0.0, 1.0))

    feats = {
        "ai_jitter": float(np.clip(ai_jitter, 0.0, 1.0)),
        "ai_f0": float(np.clip(ai_f0, 0.0, 1.0)),
        "ai_hnr": float(np.clip(ai_hnr, 0.0, 1.0)),
        "ai_flat": float(np.clip(ai_flat, 0.0, 1.0)),
        "ai_rms": float(np.clip(ai_rms, 0.0, 1.0)),
        "ai_mfcc": float(np.clip(ai_mfcc, 0.0, 1.0)),
        "ai_zcr": float(np.clip(ai_zcr, 0.0, 1.0)),
        "ai_centroid": float(np.clip(ai_centroid, 0.0, 1.0)),
        "ai_pitch_quant": float(np.clip(ai_pitch_quant, 0.0, 1.0))
    }

    # Fusion weights: emphasize jitter/harmony/pitch-quant but keep others
    w = {
        "ai_jitter": 0.46,
        "ai_f0": 0.06,
        "ai_hnr": 0.30,
        "ai_flat": 0.06,
        "ai_rms": 0.02,
        "ai_mfcc": 0.02,
        "ai_zcr": 0.02,
        "ai_centroid": 0.02,
        "ai_pitch_quant": 0.04
    }

    raw = sum(feats[k] * w[k] for k in w)

    # conditional boost
    signals = sum(1 for v in feats.values() if v >= 0.60)
    amp = 1.0
    if signals >= 2:
        amp = 1.35
    elif signals == 1:
        amp = 1.08

    prob = float(np.clip(raw * amp, 0.0, 1.0))

    # --- spectral fallback: stable centroid + decent HNR => likely TTS
    try:
        spectral_stability = centroid_std  # low -> more stable -> more TTS-like
        spec_stab_score = 1.0 - float(np.clip(spectral_stability / (sr * 0.01 + 1e-9), 0.0, 1.0))
        if spec_stab_score >= 0.55 and feats.get("ai_hnr", 0.0) >= 0.45:
            prob = max(prob, 0.90)
    except Exception:
        pass

    # Strong pitch-quant trigger: many modern TTS voices produce very quantized f0 (few unique bins).
    try:
        if feats.get("ai_pitch_quant", 0.0) >= 0.80 and voiced_ratio >= 0.60 and feats.get("ai_hnr", 0.0) >= 0.18:
            prob = max(prob, 0.98)
    except Exception:
        pass

    # Aggressive hard rules (still safe-guarded)
    try:
        if feats["ai_jitter"] >= 0.60 and feats["ai_hnr"] >= 0.50 and voiced_ratio >= 0.45:
            prob = max(prob, 0.95)
        if feats["ai_pitch_quant"] >= 0.50 and (feats["ai_jitter"] >= 0.55 or feats["ai_hnr"] >= 0.5):
            prob = max(prob, 0.93)
        strong_count = sum(1 for k in ["ai_jitter", "ai_hnr", "ai_pitch_quant", "ai_f0"] if feats.get(k, 0.0) >= 0.60)
        if strong_count >= 3:
            prob = max(prob, 0.98)
    except Exception:
        pass

    details = {
        "f0_method": f0_method,
        "f0_mean": f0_mean, "f0_std": f0_std, "voiced_ratio": voiced_ratio, "jitter": jitter,
        "pitch_quant_ratio": float(pitch_quant_ratio),
        "rms_mean": rms_mean, "rms_std": rms_std,
        "flat_mean": flat_mean, "centroid_mean": centroid_mean, "centroid_std": centroid_std,
        "zcr_mean": zcr_mean, "zcr_std": zcr_std,
        "mfcc_mean": mfcc_mean, "mfcc_std": mfcc_std,
        "hnr_db": hnr_db,
        "feats": feats,
        "weights": w,
        "raw": raw, "signals": int(signals), "amp": float(amp), "prob": prob
    }
    return prob, details

__all__ = ["analyze_audio_for_ai"]
