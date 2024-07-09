from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

def load_ner_model(model_name):
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return pipeline("ner", model=model, tokenizer=tokenizer, framework="pt")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to default model...")
        return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def evaluate_ner(text):
    models = [
        "bert-base-NER",  # BERT base version
        "dbmdz/bert-large-cased-finetuned-conll03-english",  # BERT large version
        "Jean-Baptiste/roberta-large-ner-english",  # RoBERTa
        "elastic/distilbert-base-cased-finetuned-conll03-english",  # DistilBERT
        "bhadresh-savani/albert-base-v2-ner",  # ALBERT
        "xlnet-base-cased",  # XLNet
        "google/t5-small-ner",  # T5 small version
        "google/t5-base-ner"  # T5 base version
    ]
    results = {}
    
    for model_name in models:
        try:
            ner_pipeline = load_ner_model(model_name)
            result = ner_pipeline(text)
            results[model_name] = result
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            results[model_name] = [{"error": str(e)}]
    
    return results

# Test the function
text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."

results = evaluate_ner(text)
print(results)