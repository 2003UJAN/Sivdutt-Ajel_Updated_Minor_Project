from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def load_models():
    bert_dir = "./models/bert"
    t5_dir = "./models/t5"

    if not os.path.exists(os.path.join(bert_dir, "pytorch_model.bin")):
        raise FileNotFoundError(f"Missing BERT model in {bert_dir}")

    if not os.path.exists(os.path.join(t5_dir, "pytorch_model.bin")):
        raise FileNotFoundError(f"Missing T5 model in {t5_dir}")

    bert_model = BertForSequenceClassification.from_pretrained(
        bert_dir, local_files_only=True
    )
    bert_tokenizer = BertTokenizer.from_pretrained(
        bert_dir, local_files_only=True
    )

    t5_model = AutoModelForSeq2SeqLM.from_pretrained(
        t5_dir, local_files_only=True
    )
    t5_tokenizer = AutoTokenizer.from_pretrained(
        t5_dir, local_files_only=True
    )

    return bert_model, bert_tokenizer, t5_model, t5_tokenizer
