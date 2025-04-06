from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_models():
    # Load BERT model for cyberbullying detection
    bert_model = BertForSequenceClassification.from_pretrained(
        "./models/bert", local_files_only=True
    )
    bert_tokenizer = BertTokenizer.from_pretrained(
        "./models/bert", local_files_only=True
    )

    # Load T5 model for paraphrasing
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        "./models/t5", local_files_only=True
    )
    t5_tokenizer = AutoTokenizer.from_pretrained(
        "./models/t5", local_files_only=True
    )

    return bert_model, bert_tokenizer, t5_model, t5_tokenizer

def detect_cyberbullying(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id  # 1 = Cyberbullying, 0 = Non-offensive

def rephrase_text(text, t5_model, t5_tokenizer):
    input_text = f"paraphrase: {text} </s>"
    encoding = t5_tokenizer.encode_plus(
        input_text, padding="longest", return_tensors="pt", max_length=128, truncation=True
    )
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outputs = t5_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=5,
        num_return_sequences=1,
        temperature=1.5,
    )
    rephrased = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rephrased
