import joblib, json, os, time

BASE_DIR = "model_registry"
os.makedirs(BASE_DIR, exist_ok=True)

def register_model(model, info):
    version = f"bertopic_{int(time.time())}"
    path = f"{BASE_DIR}/{version}.joblib"

    joblib.dump(model, path)

    with open(path.replace(".joblib", ".json"), "w") as f:
        json.dump({
            "topics": len(info),
            "timestamp": version
        }, f)

    return version
