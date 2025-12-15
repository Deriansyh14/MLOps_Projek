# tests/test_preprocessing.py
"""Unit tests for text preprocessing"""

import pytest
import pandas as pd
from backend.modeling.text_cleaning import (
    clean_text,
    preprocess_dataframe,
    combine_docs,
    simple_tokenizer
)


class TestTextCleaning:
    """Test text cleaning functions"""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "Hello WORLD! This is a TEST."
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_clean_text_empty(self):
        """Test cleaning empty text"""
        result = clean_text("")
        assert result == "" or len(result) == 0
    
    def test_clean_text_special_chars(self):
        """Test cleaning text with special characters"""
        text = "Machine@Learning#AI & NLP!!!"
        result = clean_text(text)
        assert isinstance(result, str)


class TestPreprocessing:
    """Test data preprocessing"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """
        Create sample dataframe for testing.
        DITAMBAH menjadi 5 dokumen untuk memenuhi batasan 'Minimal 5 dokumen' dari backend.
        """
        return pd.DataFrame({
            "Title": ["Document 1", "Document 2", "Document 3", "Document 4", "Document 5"], 
            "Abstract": [
                "This is the first abstract about ML.",
                "This is the second abstract about NLP.",
                "This is the third abstract about Big Data.",
                "This is the fourth abstract about Python.", 
                "This is the fifth abstract about data science." 
            ]
        })
    
    def test_preprocess_dataframe(self, sample_dataframe):
        """Test dataframe preprocessing"""
        result = preprocess_dataframe(sample_dataframe)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5 
        assert "Title" in result.columns
        assert "Abstract" in result.columns
    
    def test_combine_docs(self, sample_dataframe):
        """Test document combination"""
        df = preprocess_dataframe(sample_dataframe) 
        docs = combine_docs(df)
        
        assert isinstance(docs, list)
        
        assert len(docs) == 5 
        assert all(isinstance(doc, str) for doc in docs)
    
    def test_simple_tokenizer(self):
        """Test tokenization"""
        docs = [
            "Hello world machine learning",
            "Natural language processing",
            "Deep learning neural networks"
        ]
        
        result = simple_tokenizer(docs)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(tokens, list) for tokens in result)


class TestDataValidation:
    """Test data validation"""
    
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        df = pd.DataFrame({
            "Title": ["Doc 1", "Doc 2"],
            "Content": ["Content 1", "Content 2"]
        })
        

        try:
            result = preprocess_dataframe(df)
            assert result is not None
        except KeyError:
            pass
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        df = pd.DataFrame({"Title": [], "Abstract": []})
        try:
            preprocess_dataframe(df)
            result = combine_docs(df)
            assert len(result) == 0
        except ValueError as e:
            assert "Minimal 5 dokumen" in str(e)
            pass

#coba:  pytest tests/test_preprocessing.py