# app_video.py
import streamlit as st
from utils.video_face import video_face_embeddings, temporal_flicker_score

def run():
    st.subheader("🎞 Single Video — Fake Check (No Fusion)")

    v = st.file_uploader("Upload a video (face visible)", type=["mp4","mov","avi","mpeg4"], key="sv1")
    if st.button("Analyze") and v:
        tmp = "single_check.mp4"
        with open(tmp, "wb") as f: f.write(v.read())

        with st.spinner("Analyzing temporal consistency..."):
            embeds, sharp = video_face_embeddings(tmp, max_frames=150, step=3)

        if len(embeds) == 0:
            st.error("No face detected in sampled frames. Please upload a clearer, front-facing video.")
            return

        susp = temporal_flicker_score(embeds, sharp)   # 0..100 (higher = more suspicious)
        st.metric("Suspiciousness (heuristic)", f"{susp:.1f}")

        # Simple verdict thresholds (tune if needed)
        if susp >= 65:
            st.error("LIKELY AI / HEAVILY EDITED ❌")
        elif susp >= 45:
            st.warning("UNCERTAIN — needs human review ⚠️")
        else:
            st.success("APPEARS REAL ✅")
