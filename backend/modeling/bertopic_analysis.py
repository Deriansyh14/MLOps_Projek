import functools
import os
import requests
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.pyfunc
import tempfile

from backend.modeling.text_cleaning import simple_tokenizer

# ============= 1. LAZY LOADING HELPERS =============
@functools.lru_cache(maxsize=1)
def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

@functools.lru_cache(maxsize=1)
def _load_embedding_model(path: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device=device)

@functools.lru_cache(maxsize=3)
def _load_joblib(path: str):
    import joblib
    return joblib.load(path)

def _determine_min_cluster_range(n_docs: int):
    if n_docs < 500: return range(4, 10)
    if n_docs < 1000: return range(8, 25)
    if n_docs < 1500: return range(12, 30)
    if n_docs < 2500: return range(15, 35)
    if n_docs < 3500: return range(18, 42)
    return range(20, 50)

# ============= 2. EVALUATION & TUNING =============
def _evaluate_min_cluster(min_cluster_size, docs, embeddings, embedding_model, umap_model, vectorizer_model, ctfidf_model, docs_tokenized, dictionary):
    try:
        from hdbscan import HDBSCAN
        from bertopic import BERTopic

        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom", prediction_data=False, core_dist_n_jobs=-2)
        
        topic_model = BERTopic(
            embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, verbose=False
        )
        topic_model.fit(docs, embeddings)
        
        topic_freq = topic_model.get_topic_freq()
        topic_ids = topic_freq[(topic_freq["Count"] >= 5) & (topic_freq["Topic"] != -1)]["Topic"].tolist()
        
        topic_words = []
        for topic_id in topic_ids:
            words = topic_model.get_topic(topic_id)
            if isinstance(words, list):
                topic_words.append([word for word, _ in words])
        
        if len(topic_words) < 2: return (min_cluster_size, np.nan, None)
        
        coherence_model = CoherenceModel(topics=topic_words, texts=docs_tokenized, dictionary=dictionary, coherence="c_v", processes=1, topn=15)
        return (min_cluster_size, coherence_model.get_coherence(), topic_model)
    except Exception as e:
        print(f"[Warn] Error evaluating min_cluster_size={min_cluster_size}: {e}")
        return (min_cluster_size, np.nan, None)

def bertopic_analysis(docs, model_dir="save_models/all-MiniLM-L6-v2", umap_path="save_models/umap_model.joblib", vectorizer_path="save_models/vectorizer_model.joblib", ctfidf_path="save_models/ctfidf_model.joblib", max_trials=None):
    try:
        device = _get_device()
        n_docs = len(docs)
        if n_docs < 5: return {"error": "Dokumen terlalu sedikit (<5)."}

        embedding_model = _load_embedding_model(model_dir, device)
        embeddings = embedding_model.encode(docs, batch_size=64, show_progress_bar=False, device=device)
        umap_model = _load_joblib(umap_path)
        vectorizer_model = _load_joblib(vectorizer_path)
        ctfidf_model = _load_joblib(ctfidf_path)
        
        docs_tokenized = simple_tokenizer(docs)
        dictionary = Dictionary(docs_tokenized)
        
        min_cluster_range = list(_determine_min_cluster_range(n_docs))
        if max_trials: min_cluster_range = min_cluster_range[:max_trials]
        
        results = []
        for m in min_cluster_range:
            results.append(_evaluate_min_cluster(m, docs, embeddings, embedding_model, umap_model, vectorizer_model, ctfidf_model, docs_tokenized, dictionary))
        
        best_score, best_size = -1, None
        valid_clusters = []
        for m, c, _ in results:
            if not np.isnan(c):
                valid_clusters.append(m)
                if c > best_score: best_score, best_size = c, m
        
        # Plotting
        plot_html = None
        filtered = [(m, c) for m, c, _ in results if not np.isnan(c)]
        if filtered:
            import plotly.express as px
            import plotly.io as pio
            df_plot = pd.DataFrame(filtered, columns=["min_cluster_size", "coherence_score"])
            fig = px.line(df_plot, x="min_cluster_size", y="coherence_score", markers=True, title="Coherence Score Optimization")
            if best_size: fig.add_vline(x=best_size, line_dash="dash", line_color="red")
            plot_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

        cache_data = {
            "docs": docs, "embeddings": embeddings, "embedding_model": embedding_model,
            "umap_model": umap_model, "vectorizer_model": vectorizer_model,
            "ctfidf_model": ctfidf_model, "docs_tokenized": docs_tokenized, "dictionary": dictionary,
        }

        # MLflow Logging
        mlflow.set_tracking_uri("file:./data/logs")

        mlflow.set_experiment("BERTopic-Analysis")
        with mlflow.start_run(run_name="hyperparameter-tuning"):
            mlflow.log_param("n_documents", len(docs))
            mlflow.log_metric("best_coherence_score", best_score)
            if plot_html:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    f.write(plot_html)
                    mlflow.log_artifact(f.name, artifact_path="plots")
                try: os.unlink(f.name)
                except: pass

        return {
            "plot_html": plot_html,
            "best_params": {"min_cluster_size": best_size or min_cluster_range[0], "coherence_score": best_score},
            "cluster_options": sorted(valid_clusters),
            "cache_data": cache_data
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

# ============= 3. LABELING HELPERS (GROQ) =============

def _extract_words_from_representation(rep) -> List[str]:
    if not rep: return []
    if isinstance(rep, str): words = [w.strip() for w in rep.split(",")]
    elif isinstance(rep, list):
        words = []
        for item in rep:
            if isinstance(item, tuple) and len(item) >= 1: words.append(str(item[0]))
            else: words.append(str(item))
    else: words = [str(rep)]
    
    seen = set()
    deduplicated = []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            deduplicated.append(w.strip())
    return deduplicated

def generate_simple_labels(topic_info: pd.DataFrame) -> Dict[int, str]:
    labels = {}
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1: continue
        words = _extract_words_from_representation(row.get("Representation", []))
        if words:
            stopwords = {"research", "study", "analysis", "method", "based", "data", "system"}
            filtered = [w for w in words[:5] if w.lower() not in stopwords]
            labels[tid] = " ".join(filtered[:4]).title() if len(filtered) >= 3 else " ".join(words[:4]).title()
        else:
            labels[tid] = f"Topic {tid}"
    return labels

# --- PERBAIKAN UTAMA: GANTI NAMA MODEL ---
def generate_labels_with_groq(topic_info: pd.DataFrame, api_key: str, model: str = "llama-3.3-70b-versatile") -> Optional[Dict[int, str]]:
    if not api_key: return None

    base_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    labels = {}
    
    print(f"[INFO] Connecting to Groq with model: {model}...")

    try:
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1: continue

            words = _extract_words_from_representation(row["Representation"])
            unique_keywords = " ".join(words[:8])

            prompt = f"""Generate a specific, descriptive academic topic label based on these keywords: {unique_keywords}
            Rules:
            1. Maximum 5 words.
            2. Must be professional/academic.
            3. Do NOT use prefixes like "Study of", "Research on".
            4. Return ONLY the label string.
            """

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 50,
            }

            r = requests.post(base_url, headers=headers, json=payload, timeout=10)
            
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                labels[topic_id] = content.replace('"', '').replace("'", "")
            else:
                print(f"[ERROR Groq] Topic {topic_id} Failed. Status: {r.status_code}. Msg: {r.text}")
        
        return labels if labels else None

    except Exception as e:
        print(f"[CRITICAL] Groq Connection Failed: {e}")
        return None

# ============= 4. MAIN FUNCTION =============

def generate_topics_with_label(docs, embeddings, embedding_model, umap_model, vectorizer_model, ctfidf_model, min_cluster_size, groq_api_key: Optional[str] = None):
    try:
        from hdbscan import HDBSCAN
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired

        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        topic_model = BERTopic(
            embedding_model=embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, representation_model=KeyBERTInspired(),
            calculate_probabilities=True, verbose=False
        )

        topics, probs = topic_model.fit_transform(docs, embeddings)
        topic_info = topic_model.get_topic_info()

        # Labeling Logic
        labels = None
        if groq_api_key:
            labels = generate_labels_with_groq(topic_info, groq_api_key)
        
        if labels:
            print("[INFO] ✓ Menggunakan Label dari Groq AI.")
        else:
            print("[INFO] ⚠ Menggunakan Fallback (Simple Keywords) karena Groq kosong/gagal.")
            labels = generate_simple_labels(topic_info)

        for tid, label in labels.items():
            topic_info.loc[topic_info["Topic"] == tid, "Name"] = label

        return topic_model, topic_info, topics, probs

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

# ============= 5. RESEARCH GROUPS =============

def _topic_matrix_from_model(topic_model, topics, use_ctfidf=True):
    topic_order = topic_model.get_topic_info()["Topic"].tolist()
    topic_indices = [topic_order.index(t) for t in topics if t in topic_order]
    return topic_model.c_tf_idf_[topic_indices] if use_ctfidf else topic_model.topic_embeddings_[topic_indices]

def _name_group_with_groq(labels: List[str], api_key: str, model: str = "llama-3.3-70b-versatile") -> Optional[str]:
    if not api_key: return None
    prompt = f"""Name this research group based on these topics: {', '.join(labels[:10])}
    Requirements:
    1. Maximum 4 words.
    2. Must sound like a department/lab name (e.g., "AI Research Group").
    3. Return ONLY the name.
    """
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 20},
            timeout=10
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip().replace('"', '')
    except: pass
    return None

def make_research_groups(topic_model, topic_info_df: pd.DataFrame, use_ctfidf=True, color_threshold=1.0, linkage_method="ward", distance="cosine", groq_api_key: Optional[str] = None):
    topic_info = topic_info_df[topic_info_df["Topic"] != -1].copy()
    topics = topic_info["Topic"].tolist()
    if not topics: raise ValueError("No topics for grouping")
    
    X = _topic_matrix_from_model(topic_model, topics, use_ctfidf)
    if distance == "cosine": D = 1.0 - cosine_similarity(X)
    else:
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X.toarray() if hasattr(X, "toarray") else X, metric="euclidean")
    
    Z = sch.linkage(squareform(D, checks=False), method=linkage_method)
    cluster_labels = sch.fcluster(Z, t=color_threshold, criterion="distance")
    
    df_clusters = pd.DataFrame({"topic": topics, "cluster": cluster_labels})
    dfm = df_clusters.merge(topic_info[["Topic", "Name", "Count"]], left_on="topic", right_on="Topic", how="left").sort_values(["cluster", "Count"], ascending=[True, False])
    
    groups = []
    for cid, sub in dfm.groupby("cluster"):
        t_labels = [str(r["Name"]).strip() if pd.notna(r["Name"]) else f"Topic {r['topic']}" for _, r in sub.iterrows()]
        
        g_name = None
        if groq_api_key:
            g_name = _name_group_with_groq(t_labels, groq_api_key)
        
        if not g_name:
            from collections import Counter
            toks = []
            for lbl in t_labels: toks += [t.lower() for t in str(lbl).split()]
            stopwords = {"research", "group", "study", "analysis", "of", "and", "in"}
            common = [w for w, _ in Counter(toks).most_common(10) if w not in stopwords]
            g_name = " ".join(common[:3]).title() + " Group" if common else "General Group"
            
        groups.append({
            "research_group": int(cid), "group_name": g_name,
            "n_topics": len(sub), "topics": sub["topic"].tolist(), "topic_labels": t_labels
        })
        
    return pd.DataFrame(groups).sort_values("research_group").reset_index(drop=True)