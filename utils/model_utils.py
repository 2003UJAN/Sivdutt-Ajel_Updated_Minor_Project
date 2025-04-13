from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_models():
    bert_dir = "models/bert"
    t5_dir = "models/t5"

    bert_model = BertForSequenceClassification.from_pretrained(bert_dir, local_files_only=True)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_dir, local_files_only=True)

    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_dir, local_files_only=True)
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_dir, local_files_only=True)

    return bert_model, bert_tokenizer, t5_model, t5_tokenizer

def detect_cyberbullying(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    return "Cyberbullying" if label == 1 else "Not Cyberbullying"

def rephrase_text(text, model, tokenizer):
    input_text = "paraphrase: " + text + " </s>"
    inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
