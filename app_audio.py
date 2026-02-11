import streamlit as st
import tempfile
import os
import traceback

from utils.extract_audio import extract_wav_mono
from utils.audio_forensics import analyze_audio_for_ai

def run_audio():
    st.subheader("🎤 Single Audio — AI Voice or Real Speech")

    up = st.file_uploader("Upload WAV / MP3 / M4A", type=["wav", "mp3", "m4a"], key="aud_single")
    if not up:
        st.info("Upload audio to analyze.")
        return

    fd, tmp_in = tempfile.mkstemp(suffix=os.path.splitext(up.name)[1])
    os.close(fd)
    with open(tmp_in, "wb") as f:
        f.write(up.read())

    # Ensure conversion to 22050 Hz for analysis (matches analyzer SR)
    tmp_wav = None
    try:
        tmp_wav = extract_wav_mono(tmp_in, target_sr=22050)
    except Exception as e:
        print("app_audio: extract_wav_mono raised:", repr(e))
        traceback.print_exc()

    try:
        os.remove(tmp_in)
    except Exception:
        pass

    if tmp_wav is None:
        st.error("Could not decode audio. Extraction failed before analysis.")
        st.markdown(
            """
            **Troubleshooting (choose one):**
            - Install **ffmpeg** and add it to PATH (recommended). Then restart the app.
            - Ensure `librosa` and `soundfile` (and their native backends) are installed in this Python environment.
            - Try converting the file manually with ffmpeg:
              `ffmpeg -y -i "yourfile.mp3" -ac 1 -ar 22050 converted.wav`
              then upload `converted.wav`.
            """
        )
        # server-side debug hint
        print("app_audio: extract_wav_mono returned None for input (uploaded file).")
        return

    prob_ai, details = analyze_audio_for_ai(tmp_wav, sr=22050)

    try:
        os.remove(tmp_wav)
    except Exception:
        pass

    if prob_ai is None:
        st.error("Audio analysis failed.")
        return

    # Decision thresholds (tuned to be sensitive to modern TTS)
    THRESH_AI = 0.18
    THRESH_REAL = 0.14

    signals = details.get("signals", 0)
    if prob_ai >= THRESH_AI:
        st.error(f"AI Generated Speech — AI Probability {prob_ai*100:.1f}% (signals={signals})")
    elif prob_ai <= THRESH_REAL:
        st.success(f"Real Human Speech — AI Probability {prob_ai*100:.1f}% (signals={signals})")
    else:
        if signals >= 2 or prob_ai >= 0.42:
            st.warning(f"Uncertain — leans AI (prob {prob_ai*100:.1f}%, signals={signals})")
        else:
            st.warning(f"Uncertain — leans Real (prob {prob_ai*100:.1f}%, signals={signals})")

    # Show diagnostics for debugging/tuning (remove or replace with checkbox later)
    st.expander("Show audio diagnostic details", expanded=False).write(details)
    # Also print a short server-side log for easier terminal debugging
    print("app_audio: prob_ai=", prob_ai, "signals=", signals)

if __name__ == "__main__":
    st.set_page_config(page_title="Audio Fake Check", page_icon="🎤")
    run_audio()
