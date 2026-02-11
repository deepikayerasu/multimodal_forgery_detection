import streamlit as st
from compare_videos import face_similarity

def run():
    st.header("🎥 Video vs Video — Face Match (%)")

    v1 = st.file_uploader("Upload Video 1", type=["mp4","mov","avi","mkv"])
    v2 = st.file_uploader("Upload Video 2", type=["mp4","mov","avi","mkv"])

    if st.button("Compare"):
        if not v1 or not v2:
            st.error("Upload both videos first.")
            return

        open("cmp1.mp4","wb").write(v1.read())
        open("cmp2.mp4","wb").write(v2.read())

        score = face_similarity("cmp1.mp4","cmp2.mp4")
        if score is None:
            st.error("No face detected in one or both videos.")
        else:
            st.success(f"Face Similarity: **{score}%**")
