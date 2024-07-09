from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_cache = {}

def load_model(model_name):
    if model_name not in model_cache:
        model_cache[model_name] = pipeline('text-classification', model=model_name)
    return model_cache[model_name]
