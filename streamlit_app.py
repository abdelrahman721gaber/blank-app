import streamlit as st
import whisper
import tempfile
import os

# Load Whisper model (large)
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

st.title("🎙️ Arabic Speech-to-Text with Whisper")
st.write("Upload an audio/video file and get the transcription in Arabic.")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "mp4", "m4a"])

if uploaded_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(tmp_path, format="audio/wav")

    st.write("⏳ Transcribing with Whisper large model... This may take a while.")

    # Run Whisper transcription
    result = model.transcribe(tmp_path, language="ar")

    st.subheader("📝 Transcription")
    st.write(result["text"])

    # Download button
    st.download_button(
        "⬇️ Download Transcription",
        data=result["text"],
        file_name="transcription.txt"
    )

    os.remove(tmp_path)
