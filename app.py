import streamlit as st
from utils.model_utils import load_models, detect_cyberbullying, rephrase_text

# Load models
st.title("🚫 Cyberbullying Detector & 🤖 Rephraser")
st.write("Detect offensive content and suggest neutral alternatives.")

with st.spinner("Loading models..."):
    bert_model, tokenizer, t5_model, t5_tokenizer = load_models()

# Input
user_input = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label = detect_cyberbullying(user_input, bert_model, tokenizer)

        if label == 1:
            st.error("⚠️ Offensive content detected!")
            rephrased = rephrase_text(user_input, t5_model, t5_tokenizer)
            st.subheader("Suggested Rephrasing:")
            st.success(rephrased)
        else:
            st.success("✅ No offensive content detected!")
