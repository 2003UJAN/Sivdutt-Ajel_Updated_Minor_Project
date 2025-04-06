import streamlit as st
from utils.model_utils import load_models, detect_cyberbullying, rephrase_text

# Load models once
@st.cache_resource
def initialize_models():
    return load_models()

bert_model, bert_tokenizer, t5_model, t5_tokenizer = initialize_models()

# Streamlit UI
st.title("Cyberbullying Detection & Text Rephrasing")

menu = st.sidebar.selectbox("Choose Task", ["Detect Cyberbullying", "Rephrase Text"])

if menu == "Detect Cyberbullying":
    text = st.text_area("Enter text to check for cyberbullying:")
    if st.button("Detect"):
        result = detect_cyberbullying(text, bert_model, bert_tokenizer)
        st.success(f"Result: {result}")

elif menu == "Rephrase Text":
    text = st.text_area("Enter text to rephrase:")
    if st.button("Rephrase"):
        result = rephrase_text(text, t5_model, t5_tokenizer)
        st.success(f"Rephrased: {result}")
