from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch

def load_models():
    bert_model = BertForSequenceClassification.from_pretrained("model/bert_model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

    return bert_model, tokenizer, t5_model, t5_tokenizer

def detect_cyberbullying(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return pred

def rephrase_text(text, model, tokenizer):
    input_ids = tokenizer.encode("rephrase: " + text, return_tensors='pt', max_length=128, truncation=True)
    outputs = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
