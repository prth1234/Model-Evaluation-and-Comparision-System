from transformers import pipeline
from huggingface_hub import login

# Uncomment and use this if you need to authenticate
# login(token="your_token_here")

def load_qa_model(model_name):
    try:
        return pipeline("question-answering", model=model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to default model...")
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def evaluate(context, question):
    models = [
        "bert-base-uncased",  # BERT base version
        "bert-large-uncased-whole-word-masking-finetuned-squad",  # BERT large version
        "deepset/roberta-base-squad2",  # RoBERTa
        "distilbert-base-cased-distilled-squad",  # DistilBERT
        "twmkn9/albert-base-v2-squad2",  # ALBERT
        "deepset/xlnet-base-cased-squad2",  # XLNet
        "google/t5-small-qa-qg-hl",  # T5 small version
        "google/t5-base-qa-qg-hl"  # T5 base version
    ]
    results = {}
    
    for model_name in models:
        try:
            qa_pipeline = load_qa_model(model_name)
            result = qa_pipeline(question=question, context=context)
            results[model_name] = [
                {
                    "answer": result["answer"],
                    "score": result["score"]
                }
            ]
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            results[model_name] = [{"error": str(e)}]
    
    return results

# Test the function
context = "OpenAI is an artificial intelligence research laboratory consisting of the for-profit corporation OpenAI LP and its parent company, the non-profit OpenAI Inc."
question = "What is OpenAI?"

results = evaluate(context, question)
print(results)