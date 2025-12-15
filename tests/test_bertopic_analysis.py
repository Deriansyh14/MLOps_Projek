import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
# Import modul backend yang diperlukan
from backend.monitoring.model_monitoring import TopicDriftDetector, CoherenceMonitor, ModelPerformanceMonitor


class TestExtractWordsFromRepresentation:
    """Test word extraction from representation"""
    
    def test_extract_from_tuple_list(self):
        """Test extracting words from list of tuples"""
        from backend.modeling.bertopic_analysis import _extract_words_from_representation
        
        rep = [("word1", 0.9), ("word2", 0.8), ("word3", 0.7)]
        result = _extract_words_from_representation(rep)
        
        assert isinstance(result, list)
        assert "word1" in result
        assert "word2" in result
    
    def test_extract_from_string(self):
        """Test extracting words from string"""
        from backend.modeling.bertopic_analysis import _extract_words_from_representation
        
        rep = "word1, word2, word3"
        result = _extract_words_from_representation(rep)
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_extract_empty(self):
        """Test extracting from empty representation"""
        from backend.modeling.bertopic_analysis import _extract_words_from_representation
        
        result = _extract_words_from_representation(None)
        assert result == []
    
    def test_deduplication(self):
        """Test that deduplication works"""
        from backend.modeling.bertopic_analysis import _extract_words_from_representation
        
        rep = [("word", 0.9), ("word", 0.8), ("other", 0.7)]
        result = _extract_words_from_representation(rep)
        
        # Should have deduplicated
        word_count = sum(1 for w in result if w.lower() == "word")
        assert word_count == 1


class TestGenerateSimpleLabels:
    """Test simple label generation"""
    
    def test_generate_simple_labels(self):
        """Test simple label generation"""
        from backend.modeling.bertopic_analysis import generate_simple_labels
        
        topic_info = pd.DataFrame({
            "Topic": [0, 1, 2, -1],
            "Representation": [
                ["machine", "learning", "ai"],
                ["natural", "language", "nlp"],
                ["deep", "neural", "networks"],
                ["outliers"]
            ]
        })
        
        result = generate_simple_labels(topic_info)
        
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert -1 not in result  # Outliers should be excluded
    
    def test_label_format(self):
        """Test that labels are properly formatted"""
        from backend.modeling.bertopic_analysis import generate_simple_labels
        
        topic_info = pd.DataFrame({
            "Topic": [0],
            "Representation": [["word1", "word2", "word3"]]
        })
        
        result = generate_simple_labels(topic_info)
        label = result[0]
        
        assert isinstance(label, str)
        assert len(label) > 0


class TestDriftDetector:
    """Test drift detection"""
    
    def test_drift_detector_init(self):
        """Test drift detector initialization"""
        # Di-import di atas: TopicDriftDetector
        
        baseline = {0: ["word1", "word2"], 1: ["word3", "word4"]}
        detector = TopicDriftDetector(baseline)
        
        assert detector.baseline_topics == baseline
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation"""
        # Di-import di atas: TopicDriftDetector
        
        baseline = {0: ["word1", "word2"], 1: ["word3", "word4"]}
        detector = TopicDriftDetector(baseline)
        
        set1 = {"word1", "word2"}
        set2 = {"word1", "word3"}
        
        similarity = detector._jaccard_similarity(set1, set2)
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        # PERBAIKAN: Mengubah 0.5 menjadi 1/3 (1 irisan / 3 gabungan)
        assert similarity == 1/3 
    
    def test_detect_drift_no_drift(self):
        """Test drift detection when no drift occurs"""
        # Di-import di atas: TopicDriftDetector
        
        baseline = {0: ["machine", "learning"], 1: ["nlp", "text"]}
        detector = TopicDriftDetector(baseline)
        
        current = {0: ["machine", "learning"], 1: ["nlp", "text"]}
        
        result = detector.detect_drift(current)
        
        assert result["has_drift"] == False
        assert result["mean_similarity"] == 1.0


class TestCoherenceMonitor:
    """Test coherence monitoring"""
    
    def test_coherence_monitor_init(self):
        """Test coherence monitor initialization"""
        # Di-import di atas: CoherenceMonitor
        
        monitor = CoherenceMonitor()
        assert monitor.coherence_history == []
    
    def test_coherence_history(self):
        """Test coherence score history tracking"""
        # Di-import di atas: CoherenceMonitor
        
        monitor = CoherenceMonitor()
        
        # Simulate logging coherence scores
        monitor.coherence_history.append({
            "timestamp": "2024-01-01",
            "coherence_score": 0.6
        })
        
        history = monitor.get_coherence_trend()
        assert len(history) > 0


class TestPerformanceMonitor:
    """Test performance monitoring"""
    
    def test_performance_monitor_init(self):
        """Test performance monitor initialization"""
        # Di-import di atas: ModelPerformanceMonitor
        
        monitor = ModelPerformanceMonitor(model_name="TestModel")
        assert monitor.model_name == "TestModel"
    
    def test_log_inference_metrics(self):
        """Test logging inference metrics"""
        # Di-import di atas: ModelPerformanceMonitor
        
        monitor = ModelPerformanceMonitor()
        metric = monitor.log_inference_metrics(
            n_documents=10,
            n_topics=5,
            inference_time=0.5,
            coherence_score=0.7
        )
        
        assert metric["n_documents"] == 10
        assert metric["n_topics"] == 5
        assert metric["inference_time_sec"] == 0.5