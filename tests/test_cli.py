# tests/test_cli.py
"""Tests for CLI commands with Blue/Green Strategy"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from cli.bertopic_cli import app

runner = CliRunner()

class TestCLIVersion:
    """Test version command"""
    
    def test_version_command(self):
        """Test version command output"""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "BERTopic MLOps" in result.stdout
        # Memastikan versi yang muncul sesuai update terakhir
        assert "v1.0.0" in result.stdout or "Blue/Green" in result.stdout

class TestCLIHelp:
    """Test help functionality"""
    
    def test_help_main(self):
        """Test main help output"""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "train" in result.stdout
        assert "predict" in result.stdout
        # Memastikan deskripsi Blue/Green muncul
        assert "Blue/Green" in result.stdout or "Train" in result.stdout
    
    def test_train_help(self):
        """Test train command help"""
        result = runner.invoke(app, ["train", "--help"])
        
        assert result.exit_code == 0
        assert "stage" in result.stdout.lower() or "blue" in result.stdout.lower()

class TestCLIValidation:
    """Test CLI input validation"""
    
    def test_train_missing_file(self):
        """Test train with missing input file"""
        result = runner.invoke(app, ["train", "--input", "nonexistent.csv"])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout or "not found" in result.stdout
    
    def test_predict_missing_file(self):
        """Test predict with missing input file"""
        result = runner.invoke(app, ["predict", "--input", "nonexistent.csv"])
        
        assert result.exit_code != 0

class TestBlueGreenFlow:
    """
    Test Blue/Green Logic using Mocks.
    We mock the heavy backend functions so tests run instantly.
    """
    
    @pytest.fixture
    def sample_data(self):
        """Create a temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                "Title": ["Doc 1", "Doc 2", "Doc 3"],
                "Abstract": ["AI research", "Machine learning study", "Deep learning paper"]
            })
            df.to_csv(f.name, index=False)
            fname = f.name
        yield fname
        # Cleanup
        if os.path.exists(fname):
            os.unlink(fname)

    @patch("cli.bertopic_cli.mlflow")
    @patch("cli.bertopic_cli.MlflowClient")
    @patch("cli.bertopic_cli.bertopic_analysis")
    @patch("cli.bertopic_cli.generate_topics_with_label")
    @patch("cli.bertopic_cli.register_model_to_mlflow")
    def test_train_blue_green_pipeline(self, mock_reg, mock_gen, mock_analysis, mock_client, mock_mlflow, sample_data):
        """
        Test if 'train' command triggers both Blue and Green phases correctly.
        """
        # 1. Setup Mock Returns (Agar backend pura-pura berhasil)
        # Mock result tuning (Green)
        mock_analysis.return_value = {
            "best_params": {"min_cluster_size": 20, "coherence_score": 0.85},
            "cache_data": {
                "docs": [], "embeddings": [], "embedding_model": None,
                "umap_model": None, "vectorizer_model": None, "ctfidf_model": None
            }
        }
        
        # Mock model generation result (Blue & Green)
        mock_topic_info = pd.DataFrame({"Topic": [0, 1], "Name": ["T0", "T1"]})
        # Return: (model, info, topics, probs)
        mock_gen.return_value = (MagicMock(), mock_topic_info, [0, 1], [0.9, 0.8])
        
        # 2. Run CLI Command
        result = runner.invoke(app, ["train", "--input", sample_data, "--trials", "1"])
        
        # 3. Assertions
        assert result.exit_code == 0, f"Training failed with: {result.stdout}"
        
        # Cek apakah output mencerminkan strategi Blue/Green
        assert "Blue/Green" in result.stdout
        assert "[BLUE]" in result.stdout
        assert "[GREEN]" in result.stdout
        
        # Cek apakah register model dipanggil 2 kali (sekali Blue, sekali Green)
        assert mock_reg.call_count == 2
        
        # Cek apakah transisi stage dipanggil (Production & Staging)
        # Kita perlu akses mock instance dari MlflowClient
        mock_client_instance = mock_client.return_value
        assert mock_client_instance.transition_model_version_stage.call_count >= 2

    @patch("cli.bertopic_cli.mlflow.pyfunc.load_model")
    def test_predict_blue_green_flags(self, mock_load_model, sample_data):
        """
        Test if 'predict' command accepts --stage flag and loads correct URI.
        """
        # 1. Setup Mock Model
        mock_model = MagicMock()
        mock_model.predict.return_value = ([0, 1, 0]) # Mock output topics
        mock_load_model.return_value = mock_model
        
        # 2. Test Stage: Production (Blue)
        result_blue = runner.invoke(app, [
            "predict", 
            "--input", sample_data, 
            "--stage", "Production"
        ])
        
        assert result_blue.exit_code == 0
        assert "(🔵 BLUE)" in result_blue.stdout
        # Pastikan URI yang dipanggil benar
        mock_load_model.assert_called_with("models:/BERTopic-Model/Production")
        
        # 3. Test Stage: Staging (Green)
        result_green = runner.invoke(app, [
            "predict", 
            "--input", sample_data, 
            "--stage", "Staging"
        ])
        
        assert result_green.exit_code == 0
        assert "(🟢 GREEN)" in result_green.stdout
        # Pastikan URI yang dipanggil benar
        mock_load_model.assert_called_with("models:/BERTopic-Model/Staging")