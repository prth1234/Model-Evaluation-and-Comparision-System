import redis
import json
import numpy as np

r = redis.Redis(host='localhost', port=6379, db=0)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def cache_output(task, text, result):
    key = f"{task}:{text}"
    r.set(key, json.dumps(result, cls=NumpyEncoder), ex=3600)  # Cache for 1 hour