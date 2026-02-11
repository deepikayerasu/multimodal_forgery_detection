import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from main import detect

st.set_page_config(page_title="Multimodal Deepfake Detector", page_icon="🕵️")
st.title("🕵️ Multimodal Deepfake Detection (Single Video)")

video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mpeg4"])

if st.button("Analyze") and video is not None:
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(video.read())

        with st.spinner("Analyzing..."):
            label, conf = detect("temp_video.mp4")

        if label.startswith("ERROR"):
            st.error(label)
        else:
            st.success(f"Result: {label}")
            st.info(f"Confidence: {conf:.2f}%")
    except Exception as e:
        st.exception(e)  # shows the traceback in the page instead of killing the server
