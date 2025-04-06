from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

def load_models():
    # Load and save a working cyberbullying BERT model
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "SSEF-HG-AC/distilbert-uncased-finetuned-cyberbullying"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "SSEF-HG-AC/distilbert-uncased-finetuned-cyberbullying"
    )

    # Load and save a working T5 paraphrasing model
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    t5_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    return bert_model, tokenizer, t5_model, t5_tokenizer
