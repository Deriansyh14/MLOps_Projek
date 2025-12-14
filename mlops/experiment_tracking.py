import json
from datetime import datetime

def log_experiment(params, metrics):
    log = {
        "timestamp": str(datetime.now()),
        "params": params,
        "metrics": metrics
    }

    with open("data/logs/experiment.json", "a") as f:
        f.write(json.dumps(log) + "\n")
