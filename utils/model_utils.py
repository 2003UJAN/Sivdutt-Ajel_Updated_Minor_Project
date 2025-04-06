from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_models():
    # Load BERT for Cyberbullying Detection
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForSequenceClassification.from_pretrained("s-nlp/bert-base-uncased-cyberbullying")

    # Load T5 for Rephrasing
    t5_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    return bert_model, bert_tokenizer, t5_model, t5_tokenizer

def detect_cyberbullying(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction  # 1 = cyberbullying, 0 = not cyberbullying

def rephrase_text(text, model, tokenizer):
    input_text = "paraphrase: " + text + " </s>"
    encoding = tokenizer.encode_plus(input_text, return_tensors="pt", padding="longest")
    outputs = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
