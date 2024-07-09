from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def load_model(model_name):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to default model...")
        return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def evaluate(text):
    models = [
        "bert-base-uncased",  # BERT base version
        "bert-large-uncased",  # BERT large version
        "roberta-base",
        "distilbert-base-uncased",
        "albert-base-v2",
        "xlnet-base-cased",
        "t5-small",  # T5 small version
        "t5-base"    # T5 base version
    ]
    results = {}
    
    for model_name in models:
        try:
            model = load_model(model_name)
            result = model(text)
            results[model_name] = result
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            results[model_name] = [{"error": str(e)}]
    
    return results

# Test the function
text = "This movie was fantastic! I really enjoyed it."

results = evaluate(text)
print(results)