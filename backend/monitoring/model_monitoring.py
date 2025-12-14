# backend/monitoring/model_monitoring.py
"""
Model Monitoring Module - Track Topic Drift and Model Performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path
import mlflow
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


class TopicDriftDetector:
    """Detect topic drift by comparing current topics with baseline."""
    
    def __init__(self, baseline_topics: Dict[int, List[str]], baseline_timestamp: str = None):
        """
        Initialize drift detector with baseline topics.
        
        Args:
        - baseline_topics: Dict mapping topic_id to list of top words
        - baseline_timestamp: When baseline was created
        """
        self.baseline_topics = baseline_topics
        self.baseline_timestamp = baseline_timestamp or datetime.now().isoformat()
        self.drift_history = []
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def detect_drift(self, current_topics: Dict[int, List[str]], top_n: int = 10) -> Dict:
        """
        Detect topic drift by comparing current topics with baseline.
        
        Args:
        - current_topics: Dict mapping topic_id to list of top words
        - top_n: Number of top words to compare
        
        Returns:
        - Drift metrics (similarity scores, drift flag)
        """
        drift_scores = {}
        similarities = []
        
        # Compare topics
        for topic_id, baseline_words in self.baseline_topics.items():
            if topic_id not in current_topics:
                # Topic disappeared
                drift_scores[topic_id] = 0.0
                similarities.append(0.0)
            else:
                current_words = current_topics[topic_id]
                baseline_set = set(baseline_words[:top_n])
                current_set = set(current_words[:top_n])
                
                similarity = self._jaccard_similarity(baseline_set, current_set)
                drift_scores[topic_id] = similarity
                similarities.append(similarity)
        
        # Calculate aggregate metrics
        mean_similarity = np.mean(similarities) if similarities else 0.0
        min_similarity = np.min(similarities) if similarities else 0.0
        
        # Determine drift flag (if mean similarity < 0.6, there's drift)
        has_drift = mean_similarity < 0.6
        
        drift_result = {
            "timestamp": datetime.now().isoformat(),
            "baseline_timestamp": self.baseline_timestamp,
            "mean_similarity": float(mean_similarity),
            "min_similarity": float(min_similarity),
            "topic_similarities": drift_scores,
            "has_drift": has_drift,
            "drift_level": "HIGH" if mean_similarity < 0.4 else "MEDIUM" if mean_similarity < 0.6 else "LOW"
        }
        
        self.drift_history.append(drift_result)
        return drift_result
    
    def get_drift_history(self) -> List[Dict]:
        """Get all drift detection results."""
        return self.drift_history


class CoherenceMonitor:
    """Monitor model coherence score on new data."""
    
    def __init__(self):
        self.coherence_history = []
    
    def calculate_coherence(
        self,
        topics: List[List[str]],
        texts_tokenized: List[List[str]],
        dictionary: Dictionary,
        coherence_measure: str = "c_v"
    ) -> float:
        """
        Calculate coherence score on new data.
        
        Returns:
        - Coherence score (0-1)
        """
        try:
            if len(topics) < 2:
                return 0.0
            
            coherence_model = CoherenceModel(
                topics=topics,
                texts=texts_tokenized,
                dictionary=dictionary,
                coherence=coherence_measure,
                processes=1,
                topn=10
            )
            
            score = coherence_model.get_coherence()
            
            self.coherence_history.append({
                "timestamp": datetime.now().isoformat(),
                "coherence_score": float(score),
                "n_topics": len(topics)
            })
            
            return score
        
        except Exception as e:
            print(f"[WARN] Failed to calculate coherence: {e}")
            return 0.0
    
    def get_coherence_trend(self) -> pd.DataFrame:
        """Get coherence scores over time."""
        if not self.coherence_history:
            return pd.DataFrame()
        return pd.DataFrame(self.coherence_history)


class ModelPerformanceMonitor:
    """Track overall model performance and health."""
    
    def __init__(self, model_name: str = "BERTopic-Model"):
        self.model_name = model_name
        self.performance_log = []
    
    def log_inference_metrics(
        self,
        n_documents: int,
        n_topics: int,
        inference_time: float,
        coherence_score: float = None,
        drift_detected: bool = False,
        drift_level: str = "LOW"
    ) -> Dict:
        """
        Log inference metrics to history.
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "n_documents": n_documents,
            "n_topics": n_topics,
            "inference_time_sec": float(inference_time),
            "coherence_score": float(coherence_score) if coherence_score else None,
            "drift_detected": drift_detected,
            "drift_level": drift_level,
            "model_name": self.model_name
        }
        
        self.performance_log.append(metric)
        
        # Log to MLflow
        try:
            mlflow.log_metric("inference_documents", n_documents)
            mlflow.log_metric("inference_topics", n_topics)
            mlflow.log_metric("inference_time_sec", float(inference_time))
            if coherence_score:
                mlflow.log_metric("inference_coherence", float(coherence_score))
            mlflow.log_param("drift_detected", drift_detected)
            mlflow.log_param("drift_level", drift_level)
        except Exception:
            pass  # MLflow not initialized
        
        return metric
    
    def get_performance_report(self) -> pd.DataFrame:
        """Get performance metrics over time."""
        if not self.performance_log:
            return pd.DataFrame()
        return pd.DataFrame(self.performance_log)
    
    def save_monitoring_log(self, filepath: str):
        """Save monitoring log to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
        print(f"[INFO] ✓ Monitoring log saved to {filepath}")
    
    def load_monitoring_log(self, filepath: str):
        """Load monitoring log from JSON file."""
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                self.performance_log = json.load(f)
            print(f"[INFO] ✓ Loaded {len(self.performance_log)} monitoring records")


class InferenceMonitor:
    """Monitor inference on new data and detect anomalies."""
    
    def __init__(self, topic_model, baseline_topics: Dict[int, List[str]]):
        """
        Initialize with trained model and baseline topics.
        """
        self.topic_model = topic_model
        self.drift_detector = TopicDriftDetector(baseline_topics)
        self.coherence_monitor = CoherenceMonitor()
        self.performance_monitor = ModelPerformanceMonitor()
    
    def run_inference(
        self,
        new_docs: List[str],
        new_texts_tokenized: List[List[str]] = None,
        dictionary: Dictionary = None,
        calculate_coherence: bool = True
    ) -> Dict:
        """
        Run inference on new data and compute monitoring metrics.
        
        Returns:
        - Inference results with drift metrics
        """
        import time
        
        start_time = time.time()
        
        try:
            # Run inference
            topics, probs = self.topic_model.transform(new_docs)
            inference_time = time.time() - start_time
            
            # Get current topic words
            current_topics = {}
            for topic_id in range(len(self.topic_model.get_topic_info())):
                try:
                    words = self.topic_model.get_topic(topic_id)
                    if isinstance(words, list):
                        current_topics[topic_id] = [w for w, _ in words]
                except:
                    pass
            
            # Detect drift
            drift_result = self.drift_detector.detect_drift(current_topics)
            
            # Calculate coherence if requested
            coherence_score = None
            if calculate_coherence and new_texts_tokenized and dictionary:
                unique_topics = list(set(topics))
                if len(unique_topics) > 1:
                    topic_words = [current_topics.get(t, []) for t in unique_topics]
                    coherence_score = self.coherence_monitor.calculate_coherence(
                        topic_words,
                        new_texts_tokenized,
                        dictionary
                    )
            
            # Log metrics
            perf_metric = self.performance_monitor.log_inference_metrics(
                n_documents=len(new_docs),
                n_topics=len(current_topics),
                inference_time=inference_time,
                coherence_score=coherence_score,
                drift_detected=drift_result["has_drift"],
                drift_level=drift_result["drift_level"]
            )
            
            result = {
                "status": "success",
                "n_documents": len(new_docs),
                "n_topics": len(current_topics),
                "topics_assigned": topics.tolist(),
                "probabilities": probs.tolist() if hasattr(probs, 'tolist') else probs,
                "inference_time_sec": inference_time,
                "coherence_score": coherence_score,
                "drift": drift_result,
                "performance_metric": perf_metric
            }
            
            return result
        
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            inference_time = time.time() - start_time
            
            return {
                "status": "error",
                "error_message": str(e),
                "inference_time_sec": inference_time
            }
    
    def get_monitoring_dashboard_data(self) -> Dict:
        """Get all monitoring data for dashboard visualization."""
        return {
            "latest_drift": self.drift_detector.drift_history[-1] if self.drift_detector.drift_history else None,
            "drift_history": self.drift_detector.drift_history,
            "coherence_trend": self.coherence_monitor.get_coherence_trend().to_dict('records'),
            "performance_report": self.performance_monitor.get_performance_report().to_dict('records'),
            "model_name": self.performance_monitor.model_name
        }
