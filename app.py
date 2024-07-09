from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from models import (
    model_loader,
    text_classification,
    named_entity_recognition,
    question_answering,
    text_summarization,
)
from collections import Counter
import time
from utils.caching import cache_output
from utils.logging import setup_logging
from models.text_classification import evaluate
import hashlib
from functools import wraps

import models.text_classification

# Global variables to track usage statistics
request_count = Counter()
last_request_time = {}
start_time = time.time()

# API key storage
API_KEYS = {
    "admin1": "user1"
}
def calculate_metrics(true_labels, predicted_labels, task):
    if task in ['text_classification', 'ner']:
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        return {'accuracy': accuracy, 'f1_score': f1}
    elif task == 'text_summarization':
        # For text summarization, we'll use BLEU score
        # Assuming true_labels and predicted_labels are lists of summaries
        true_labels = [nltk.word_tokenize(t.lower()) for t in true_labels]
        predicted_labels = [nltk.word_tokenize(p.lower()) for p in predicted_labels]
        bleu = corpus_bleu([[t] for t in true_labels], predicted_labels)
        return {'bleu_score': bleu}
    elif task == 'question_answering':
        # For QA, we could use exact match and F1 score
        # This is a simplified version, you might want to use more sophisticated metrics
        exact_match = sum([1 if t == p else 0 for t, p in zip(true_labels, predicted_labels)]) / len(true_labels)
        return {'exact_match': exact_match}

def benchmark_models(dataset, task):
    models = {
        'text_classification': [text_classification.evaluate],
        'ner': [named_entity_recognition.evaluate_ner],
        'question_answering': [question_answering.evaluate],
        'text_summarization': [text_summarization.evaluate]
    }

    results = {}
    for model_name, model_func in models[task]:
        predictions = []
        for item in dataset:
            if task == 'question_answering':
                pred = model_func(item['context'], item['question'])
            else:
                pred = model_func(item['text'])
            predictions.append(pred)
        
        metrics = calculate_metrics([item['label'] for item in dataset], predictions, task)
        results[model_name] = metrics

    return results
def make_cache_key():
    """Create a cache key based on the request URL and body."""
    return hashlib.md5(
        f"{request.path}:{request.data.decode('utf-8')}".encode('utf-8')
    ).hexdigest()

# Custom cache implementation
class CustomCache:
    def __init__(self):
        self.cache = {}

    def get(self, key, api_key):
        if key in self.cache and self.cache[key]['api_key'] == api_key:
            if self.cache[key]['expiry'] > time.time():
                return self.cache[key]['data']
        return None

    def set(self, key, data, api_key, timeout=300):
        self.cache[key] = {
            'data': data,
            'api_key': api_key,
            'expiry': time.time() + timeout
        }

    def clean(self):
        current_time = time.time()
        self.cache = {k: v for k, v in self.cache.items() if v['expiry'] > current_time}

custom_cache = CustomCache()

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key in API_KEYS:
            return f(*args, **kwargs)
        else:
            return jsonify({"error": "Invalid or missing API Key"}), 401
    return decorated_function


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder  # Set the custom JSON encoder

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

CORS(app)
limiter = Limiter(key_func=get_remote_address, app=app)

setup_logging()

def convert_floats(obj):
    """
    Recursively convert numpy float32 to native Python float.
    """
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj

def make_cache_key():
    """Create a cache key based on the request URL and body."""
    return hashlib.md5(
        f"{request.path}:{request.data.decode('utf-8')}".encode('utf-8')
    ).hexdigest()

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key in API_KEYS:
            return f(*args, **kwargs)
        else:
            return jsonify({"error": "Invalid or missing API Key"}), 401
    return decorated_function

@app.before_request
def before_request():
    """Update request statistics before each request."""
    api_key = request.headers.get('X-API-Key')
    if api_key in API_KEYS:  # Only count authenticated requests
        endpoint = request.endpoint
        if endpoint:
            request_count[endpoint] += 1
            last_request_time[endpoint] = time.time()

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint with basic usage statistics.

    Returns:
        JSON response with status, uptime, and usage statistics.

    Example:
        curl -X GET http://localhost:5000/health
    """
    current_time = time.time()
    uptime = current_time - start_time

    stats = {
        "status": "healthy",
        "message": "Service is running",
        "uptime_seconds": int(uptime),
        "total_requests": sum(request_count.values()),
        "requests_per_endpoint": dict(request_count),
        "last_request_time": {endpoint: f"{current_time - t:.2f} seconds ago" 
                              for endpoint, t in last_request_time.items()},
        "active_api_keys": len(API_KEYS)
    }

    return jsonify(stats), 200

@app.route('/evaluate', methods=['POST'])
# @limiter.limit("10 per minute")
@require_api_key
def evaluate():
    """
    Evaluate text using a specific NLP task.
    """
    api_key = request.headers.get('X-API-Key')
    cache_key = make_cache_key()
    
    custom_cache.clean()  # Clean expired cache entries
    cached_response = custom_cache.get(cache_key, api_key)
    if cached_response:
        print("Returning cached response")  # Debug print
        return jsonify(cached_response), 200

    print("Cache miss, evaluating...")  # Debug print

    data = request.json
    text = data.get("text")
    task = data.get("task")

    if not text or not task:
        return jsonify({"error": "Text and task type are required"}), 400
    
    if task == "text_classification":
        results = text_classification.evaluate(text)
    elif task == "ner":
        results = named_entity_recognition.evaluate_ner(text)
    elif task == "question_answering":
        question = data.get("question")
        if not question:
            return jsonify({"error": "Question is required for question answering task"}), 400
        results = question_answering.evaluate(text, question)
    elif task == "text_summarization":
        results = text_summarization.evaluate(text)
    else:
        return jsonify({"error": "Invalid task type"}), 400

    results = convert_floats(results)  # Ensure all float32s are converted to float
    cache_output(task, text, results)
    
    custom_cache.set(cache_key, results, api_key)
    print(f"Cached result for key: {cache_key}")  # Debug print
    
    return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)