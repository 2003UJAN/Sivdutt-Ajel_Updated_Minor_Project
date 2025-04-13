from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

def load_models():
    bert_dir = os.path.join("models", "bert")
    t5_dir = os.path.join("models", "t5")

    # Load BERT model and tokenizer
    try:
        bert_model = BertForSequenceClassification.from_pretrained(bert_dir, local_files_only=True)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_dir, local_files_only=True)
        bert_model.eval()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load BERT model: {e}")

    # Load T5 model and tokenizer
    try:
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_dir, local_files_only=True)
        t5_tokenizer = AutoTokenizer.from_pretrained(t5_dir, local_files_only=True)
        t5_model.eval()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load T5 model: {e}")

    return bert_model, bert_tokenizer, t5_model, t5_tokenizer

def detect_cyberbullying(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()

    label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
    return label_map.get(label, "Unknown")

def rephrase_text(text, model, tokenizer):
    input_text = f"paraphrase: {text} </s>"
    inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
