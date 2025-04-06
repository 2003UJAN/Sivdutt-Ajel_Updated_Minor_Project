import streamlit as st
from utils.model_utils import load_models, detect_cyberbullying, rephrase_text

# Load models once
@st.cache_resource
def initialize_models():
    return load_models()

bert_model, bert_tokenizer, t5_model, t5_tokenizer = initialize_models()

# Page config
st.set_page_config(page_title="Cyberbullying Detector & Rephraser", layout="wide")

# Title section
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #6A1B9A;">💬 AI-Powered Cyberbullying Detector & Text Rephraser</h1>
        <p style="font-size: 18px;">Detect harmful language and generate safe, rephrased alternatives using BERT and T5 models</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🧠 Choose a Task")
task = st.sidebar.radio("Select what you want to do:", ["Detect Cyberbullying", "Rephrase Text"])

# Main area
st.markdown("---")
st.markdown("### ✏️ Input Your Text")

text = st.text_area("Paste or type text here...", height=200)

col1, col2 = st.columns([1, 2])

if task == "Detect Cyberbullying":
    with col1:
        if st.button("🔍 Detect"):
            if text.strip():
                result = detect_cyberbullying(text, bert_model, bert_tokenizer)
                st.success(f"🛡️ **Detection Result:** {result}")
            else:
                st.warning("⚠️ Please enter some text.")

elif task == "Rephrase Text":
    with col1:
        if st.button("🔁 Rephrase"):
            if text.strip():
                result = rephrase_text(text, t5_model, t5_tokenizer)
                st.success(f"✨ **Rephrased Text:** {result}")
            else:
                st.warning("⚠️ Please enter some text to rephrase.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: grey;">
        Developed with ❤️ using Hugging Face Transformers and Streamlit<br>
        <a href="https://github.com/your-github" target="_blank">GitHub</a> | 
        <a href="https://huggingface.co/models" target="_blank">Hugging Face</a>
    </div>
    """,
    unsafe_allow_html=True
)
