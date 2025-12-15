import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from cli.bertopic_cli import app

runner = CliRunner()

class TestBlueGreenFlow:
    """
    Test Blue/Green Logic using Mocks.
    """
    
    @pytest.fixture
    def sample_data(self):
        """Create a temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Data harus minimal 5 baris agar tidak gagal di validasi backend
            df = pd.DataFrame({
                "Title": ["D1", "D2", "D3", "D4", "D5"],
                "Abstract": ["AI research", "Machine learning study", "Deep learning paper", "Robotics", "Data Science"]
            })
            df.to_csv(f.name, index=False)
            fname = f.name
        yield fname
        # Cleanup
        if os.path.exists(fname):
            os.unlink(fname)

    # Mocking kuat untuk semua MLflow components
    @patch("cli.bertopic_cli.mlflow") 
    @patch("cli.bertopic_cli.MlflowClient")
    @patch("cli.bertopic_cli.bertopic_analysis")
    @patch("cli.bertopic_cli.generate_topics_with_label")
    @patch("cli.bertopic_cli.register_model_to_mlflow") 
    def test_train_blue_green_pipeline(self, mock_reg, mock_gen, mock_analysis, mock_client, mock_mlflow, sample_data):
        """
        Test if 'train' command triggers both Blue and Green phases correctly (Exit Code 0).
        """
        # 1. Setup Mock Returns (Agar backend pura-pura berhasil)
        
        # Mocking mlflow.start_run
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
        
        # Mocking MlflowClient methods
        mock_client_instance = mock_client.return_value
        # Pastikan get_latest_versions mengembalikan sesuatu yang bisa diakses .version
        mock_client_instance.get_latest_versions.return_value = [MagicMock(version=1)] 
        
        # Mock result tuning (Green)
        mock_analysis.return_value = {
            "best_params": {"min_cluster_size": 20, "coherence_score": 0.85},
            "cache_data": {
                "docs": ["D1", "D2", "D3", "D4", "D5"], 
                "embeddings": [0]*5, # Dummy embeddings
                "embedding_model": None, "umap_model": None, 
                "vectorizer_model": None, "ctfidf_model": None
            }
        }
        
        # Mock model generation result (Blue & Green)
        mock_topic_info = pd.DataFrame({"Topic": [0, 1], "Name": ["T0", "T1"]})
        # Return: (model, info, topics, probs)
        mock_gen.return_value = (MagicMock(), mock_topic_info, [0, 1, 0, 1, 0], [0.9]*5)
        
        # Mock register_model_to_mlflow agar mengembalikan sukses
        mock_reg.return_value = {"model_version": 1, "run_id": "dummy"}
        
        # 2. Run CLI Command
        result = runner.invoke(app, ["train", "--input", sample_data, "--trials", "1"])
        
        # 3. Assertions
        # Jika semua mock sukses, exit_code harus 0 (Perbaikan #2)
        assert result.exit_code == 0, f"Training failed with: {result.stdout}"
        
        # Cek apakah register model dipanggil 2 kali (sekali Blue, sekali Green)
        assert mock_reg.call_count == 2
        
        # Cek apakah transisi stage dipanggil setidaknya 2 kali (Production & Staging)
        assert mock_client_instance.transition_model_version_stage.call_count >= 2

    @patch("cli.bertopic_cli.mlflow.pyfunc.load_model")
    def test_predict_blue_green_flags(self, mock_load_model, sample_data):
        """
        Test if 'predict' command accepts --stage flag and loads correct URI (Exit Code 0).
        """
        # 1. Setup Mock Model
        mock_model = MagicMock()
        mock_model.predict.return_value = ([0, 1, 0, 1, 0]) # Mock output topics (sesuai 5 baris data)
        # Pastikan mock_load_model selalu mengembalikan model yang berfungsi
        mock_load_model.return_value = mock_model
        
        # 2. Test Stage: Production (Blue)
        result_blue = runner.invoke(app, [
            "predict", 
            "--input", sample_data, 
            "--stage", "Production"
        ])
        
        # Assert exit_code 0 (Perbaikan #3)
        assert result_blue.exit_code == 0, f"Predict Blue failed with: {result_blue.stdout}"
        mock_load_model.assert_called_with("models:/BERTopic-Model/Production")
        
        # 3. Test Stage: Staging (Green)
        result_green = runner.invoke(app, [
            "predict", 
            "--input", sample_data, 
            "--stage", "Staging"
        ])
        
        # Assert exit_code 0 (Perbaikan #3)
        assert result_green.exit_code == 0, f"Predict Green failed with: {result_green.stdout}"
        mock_load_model.assert_called_with("models:/BERTopic-Model/Staging")