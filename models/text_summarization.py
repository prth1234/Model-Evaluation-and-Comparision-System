from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None, None

def get_sentence_embeddings(sentences, tokenizer, model):
    try:
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return None

def evaluate(text):
    models = [
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "distilbert-base-uncased",
        "albert-base-v2",
        "xlnet-base-cased",
        "t5-small",
        "t5-base"
    ]
    results = {}
    
    sentences = [sent.strip() for sent in text.split('.') if sent.strip()]
    
    for model_name in models:
        tokenizer, model = load_model(model_name)
        if tokenizer is None or model is None:
            continue
        
        sentence_embeddings = get_sentence_embeddings(sentences, tokenizer, model)
        if sentence_embeddings is None:
            continue
        
        # Calculate similarity between each sentence and the whole text
        text_embedding = sentence_embeddings.mean(dim=0, keepdim=True)
        similarities = cosine_similarity(sentence_embeddings.numpy(), text_embedding.numpy())
        
        # Select top 3 sentences as summary
        top_indices = np.argsort(similarities.flatten())[-3:][::-1]
        summary = '. '.join([sentences[i] for i in top_indices])
        
        results[model_name] = summary
    
    return results

