import streamlit as st
from utils.model_utils import load_models, detect_cyberbullying, rephrase_text

# Page configuration
st.set_page_config(page_title="Cyberbullying Detection & Rephrasing", page_icon="🛡️")

st.title("🛡️ Cyberbullying Detection & Rephrasing")
st.caption("🔍 Powered by BERT (Detection) + T5 (Rephrasing) | No External APIs Used")

# Load models
with st.spinner("🔄 Loading models..."):
    bert_model, tokenizer, t5_model, t5_tokenizer = load_models()

# User input
user_input = st.text_area("✍️ Enter a message to check for cyberbullying and get a rephrased suggestion if needed:")

# Analyze button
if st.button("🚀 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        with st.spinner("Analyzing message..."):
            label = detect_cyberbullying(user_input, bert_model, tokenizer)
            
            if label == 1:
                st.error("🚫 Cyberbullying Detected!")
                rephrased = rephrase_text(user_input, t5_model, t5_tokenizer)
                st.markdown("---")
                st.subheader("💡 Suggested Rewrite:")
                st.success(rephrased)
            else:
                st.success("✅ No Cyberbullying Detected!")

# Footer
st.markdown("---")
st.caption("🔒 This app runs entirely on-device. No messages are stored or sent to external APIs.")

