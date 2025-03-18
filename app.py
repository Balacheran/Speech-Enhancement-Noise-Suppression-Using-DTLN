import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import tempfile
import os
from DTLN_model import DTLN_model  # Make sure this is imported correctly from your project files

# Set page configuration
st.set_page_config(
    page_title="Audio Noise Suppression",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme detection
is_dark_theme = st.get_option("theme.base") == "dark"

# Apply different CSS styles based on theme
if is_dark_theme:
    st.markdown("""
        <style>
        .main {
            background-color: #333;
            color: white;
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main {
            background-color: #fff;
            color: black;
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #008CBA;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load and cache the DTLN model"""
    model_path = 'pretrained_model/DTLN_norm_500h.h5'  # Update path if needed
    norm_stft = True if 'norm' in model_path else False
    
    model_class = DTLN_model()
    model_class.build_DTLN_model(norm_stft=norm_stft)
    model_class.model.load_weights(model_path)
    return model_class.model

def process_audio(input_path):
    """Process audio file using the loaded model"""
    model = load_model()
    
    # Load audio file
    audio, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # Process audio
    processed_audio = model.predict(np.expand_dims(audio, axis=0).astype(np.float32))
    processed_audio = np.squeeze(processed_audio)
    
    return audio, processed_audio, sr

def main():
    st.title("🎧 Deep Noise Suppression using DTLN")
    st.markdown("Upload a WAV audio file to remove background noise using state-of-the-art AI processing")

    # File upload section
    with st.container():
        st.header("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a WAV file", 
            type=["wav"],
            accept_multiple_files=False,
            key="file_uploader"
        )

    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Process audio
        with st.spinner('Processing audio... This may take a few moments'):
            try:
                original_audio, processed_audio, sr = process_audio(tmp_path)
                
                # Create two columns layout
                col1, col2 = st.columns(2)

                # Original audio column
                with col1:
                    st.subheader("Original Audio")
                    st.audio(tmp_path, format="audio/wav")
                    
                    # Save original to bytes for download
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as orig_tmp:
                        sf.write(orig_tmp.name, original_audio, sr)
                        original_bytes = open(orig_tmp.name, "rb").read()
                    
                    st.download_button(
                        label="Download Original",
                        data=original_bytes,
                        file_name="original_audio.wav",
                        mime="audio/wav"
                    )

                # Processed audio column
                with col2:
                    st.subheader("Processed Audio")
                    
                    # Create temporary file for processed audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as proc_tmp:
                        sf.write(proc_tmp.name, processed_audio, sr)
                        processed_bytes = open(proc_tmp.name, "rb").read()
                    
                    st.audio(processed_bytes, format="audio/wav")
                    
                    st.download_button(
                        label="Download Processed",
                        data=processed_bytes,
                        file_name="processed_audio.wav",
                        mime="audio/wav"
                    )

                # Cleanup temporary files
                os.unlink(tmp_path)
                os.unlink(proc_tmp.name)
                os.unlink(orig_tmp.name)

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                os.unlink(tmp_path)

if __name__ == "__main__":
    main()
