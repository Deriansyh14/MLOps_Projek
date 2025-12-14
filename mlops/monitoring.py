def monitor_coherence(df):
    best = df.sort_values("coherence", ascending=False).iloc[0]
    return {
        "best_min_cluster": int(best["min_cluster_size"]),
        "best_coherence": float(best["coherence"])
    }
