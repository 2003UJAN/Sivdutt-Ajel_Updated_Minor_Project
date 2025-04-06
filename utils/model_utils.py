from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_models():
    # Load BERT model and tokenizer from local folder
    bert_model = BertForSequenceClassification.from_pretrained("./models/bert")
    bert_tokenizer = BertTokenizer.from_pretrained("./models/bert")

    # Load T5 model and tokenizer from local folder
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("./models/t5")
    t5_tokenizer = AutoTokenizer.from_pretrained("./models/t5")

    return bert_model, bert_tokenizer, t5_model, t5_tokenizer

def detect_cyberbullying(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id  # 1 = cyberbullying, 0 = safe

def rephrase_text(text, model, tokenizer):
    input_text = f"paraphrase: {text} </s>"
    inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
