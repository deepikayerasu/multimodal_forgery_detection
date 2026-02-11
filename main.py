# main.py
import streamlit as st

from app_photos_main import run_photos, run_compare_photos
from app_video import run_video
from app_compare_videos import run_compare_videos
from app_audio import run_audio

# MUST be first Streamlit call
st.set_page_config(page_title="Multimodal Forgery Detection and Localization", page_icon="🛡", layout="wide")

st.title("Multimodal Forgery Detection and Localization")

option = st.radio(
    "Choose Tool",
    [
        "Photo (AI or Real)",
        "Photo vs Photo (Compare Faces)",
        "Single Video (Fake Check)",
        "Video vs Video (Compare Faces)",
        "Single Audio (Fake Check)"
    ]
)

if option == "Photo (AI or Real)":
    run_photos()

elif option == "Photo vs Photo (Compare Faces)":
    run_compare_photos()

elif option == "Single Video (Fake Check)":
    run_video()

elif option == "Video vs Video (Compare Faces)":
    run_compare_videos()

elif option == "Single Audio (Fake Check)":
    run_audio()

