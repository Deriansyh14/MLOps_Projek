# 🚀 BERTopic MLOps - Complete Production MLOps Pipeline

A comprehensive MLOps project implementing topic modeling using BERTopic with production-ready infrastructure including model tracking, monitoring, CLI, testing, and deployment capabilities.

## 📋 Project Overview

This project demonstrates a complete MLOps pipeline for topic modeling with:
- ✅ **Streamlit Web UI** for interactive analysis
- ✅ **Typer CLI** for command-line operations
- ✅ **MLflow Tracking & Registry** for experiment tracking and model versioning
- ✅ **Comprehensive Monitoring** with drift detection and coherence tracking
- ✅ **Full Test Suite** with pytest
- ✅ **CI/CD Pipeline** with GitHub Actions
- ✅ **Docker Support** for containerization

## 🎯 Quick Start

### 1. Setup Environment

```bash
# Clone/download project
cd MLOps

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install typer CLI (if not in requirements)
pip install typer
```

### 2. Run Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` with 3 tabs:
- 🎯 **Training & Analysis** - Upload CSV and train model
- 📊 **Model Monitoring** - Inference and drift detection
- 📈 **MLflow Tracking** - View experiment tracking info

### 3. Use CLI

```bash
# Show help
python -m cli.bertopic_cli --help

# Show version
python -m cli.bertopic_cli version

# Train model
python -m cli.bertopic_cli train --input data/raw/data.csv --max-trials 10

# Run inference
python -m cli.bertopic_cli predict --input data/test.csv

# Monitor performance
python -m cli.bertopic_cli monitor --input data/test.csv

# Evaluate experiments
python -m cli.bertopic_cli evaluate --metrics data/logs/experiment.json
```

### 4. Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=backend --cov=cli

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run only unit tests
pytest tests/ -k "test_" -v
```

### 5. View MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --host 127.0.0.1 --port 5000

# Open http://localhost:5000 in browser
```

## 📁 Project Structure

```
MLOps/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration
├── README.md                       # This file
├── DEPLOYMENT_CHECKLIST.md         # Compliance checklist
│
├── backend/
│   ├── eda/
│   │   └── eda_report.py          # EDA report generation
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── bertopic_analysis.py   # Core BERTopic analysis & MLflow tracking
│   │   └── text_cleaning.py       # Text preprocessing
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── model_monitoring.py    # Drift detection & monitoring
│   └── registry/
│       └── model_registry.py      # Model registry functions
│
├── cli/
│   ├── __init__.py
│   ├── run_bertopic.py            # Legacy CLI (deprecated)
│   └── bertopic_cli.py            # NEW: Typer-based CLI
│
├── mlops/
│   ├── experiment_tracking.py     # Experiment logging
│   └── monitoring.py              # Basic monitoring utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py      # Text cleaning tests
│   ├── test_bertopic_analysis.py  # Analysis & monitoring tests
│   └── test_cli.py                # CLI command tests
│
├── data/
│   ├── raw/                       # Input data
│   ├── processed/                 # Processed data
│   └── logs/                      # Experiment logs & monitoring data
│
├── save_models/
│   ├── *.joblib                   # Saved model components
│   ├── all-MiniLM-L6-v2/         # Pre-trained embeddings
│   └── nltk_data/                 # NLTK resources
│
├── ui/
│   ├── script.js                  # (Future) Custom JS
│   └── style.css                  # (Future) Custom CSS
│
└── .github/
    └── workflows/
        └── ml-ci.yml              # GitHub Actions CI/CD
```

## 🔧 Key Components

### 1. **Model Training & Analysis** (`backend/modeling/bertopic_analysis.py`)

```python
from backend.modeling.bertopic_analysis import bertopic_analysis, generate_topics_with_label

# Automated hyperparameter tuning with coherence evaluation
result = bertopic_analysis(docs, max_trials=None)

# Generate final model with AI labels via Groq
topic_model, topic_info, topics, probs = generate_topics_with_label(
    docs=docs,
    embeddings=embeddings,
    embedding_model=embedding_model,
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    min_cluster_size=best_size,
    groq_api_key=None  # Uses fallback word-based labels if API unavailable
)
```

**Features:**
- ✅ Lazy-loaded models (efficient memory usage)
- ✅ Coherence-based hyperparameter tuning
- ✅ MLflow tracking of metrics & artifacts
- ✅ Automatic model registry with versioning
- ✅ Groq API for intelligent topic labeling (with fallback)
- ✅ Deduplication of keywords

### 2. **Monitoring & Drift Detection** (`backend/monitoring/model_monitoring.py`)

```python
from backend.monitoring.model_monitoring import InferenceMonitor

# Initialize with trained model
monitor = InferenceMonitor(topic_model, baseline_topics)

# Run inference with drift detection
result = monitor.run_inference(
    new_docs=new_documents,
    new_texts_tokenized=tokenized_docs,
    dictionary=gensim_dictionary,
    calculate_coherence=True
)

# Get monitoring dashboard data
dashboard = monitor.get_monitoring_dashboard_data()
```

**Metrics Tracked:**
- Topic Drift (Jaccard similarity: 0-1)
- Coherence Score (c_v metric)
- Inference Time
- Document/Topic Counts
- Drift Level (LOW/MEDIUM/HIGH)

### 3. **CLI with Typer** (`cli/bertopic_cli.py`)

```bash
# Train model with options
bertopic train \
  --input data/documents.csv \
  --output save_models \
  --trials 10 \
  --experiment "experiment-name"

# Run inference
bertopic predict --input new_data.csv --model-version latest

# Monitor performance
bertopic monitor --input data/test.csv --output report.json

# View performance
bertopic evaluate --metrics data/logs/experiment.json

# Show version
bertopic version
```

### 4. **Comprehensive Testing** (`tests/`)

Test suites for:
- ✅ Text preprocessing & cleaning
- ✅ BERTopic analysis functions
- ✅ Monitoring & drift detection
- ✅ CLI commands
- ✅ Data validation

Run tests:
```bash
pytest tests/ -v --cov=backend --cov=cli
```

### 5. **Streamlit Application** (`app.py`)

**Tab 1: Training & Analysis**
- Upload CSV with Title & Abstract columns
- Automatic hyperparameter tuning
- Coherence score visualization
- Topic labels display
- Document explorer by topic
- CSV export

**Tab 2: Model Monitoring**
- New data inference
- Topic drift detection visualization
- Coherence trend analysis
- Performance metrics
- Real-time monitoring data

**Tab 3: MLflow Tracking**
- Experiment info
- Instructions for MLflow UI access

## 📊 Data Format

Input CSV must contain:
- `Title` (string) - Document title
- `Abstract` (string) - Document abstract

Example:
```csv
Title,Abstract
Antioxidant Compounds in Cancer Prevention,This study investigates...
Machine Learning Applications in Healthcare,Recent advances in ML...
```

## 🔐 API Keys

**Groq API (optional for topic labeling):**
1. Get key from https://console.groq.com
2. Provide via:
   - Streamlit sidebar input
   - Environment variable: `GROQ_API_KEY`
   - Fallback: Auto-use word-based labels if API unavailable

## 📈 MLflow Integration

All training runs are automatically logged to MLflow:

```bash
# Start MLflow UI
mlflow ui --host 127.0.0.1 --port 5000
```

View:
- Experiment metrics (coherence, min_cluster_size, etc.)
- Artifacts (coherence plots, topic info CSV)
- Model versions (BERTopic-Model-v1, v2, etc.)
- Run parameters and timestamps

## ✅ Production Readiness Checklist

- ✅ **Model Serving**: Online inference via Streamlit + monitoring
- ✅ **CLI Interface**: Full Typer-based command-line support
- ✅ **Testing**: Comprehensive pytest suite with 20+ tests
- ✅ **Monitoring**: Topic drift detection, coherence tracking, performance metrics
- ✅ **Tracking**: MLflow experiment tracking & model registry
- ✅ **Documentation**: Complete README and deployment guide
- ✅ **Error Handling**: Graceful fallbacks for API failures
- ✅ **Logging**: Console + JSON logging + MLflow tracking

## 🚀 Deployment Options

### Option 1: Local Development
```bash
streamlit run app.py
```

### Option 2: Docker Container
```bash
docker build -t bertopic-mlops .
docker run -p 8501:8501 bertopic-mlops
```

### Option 3: Cloud Deployment
- **AWS EC2** - Host on EC2 instance with Streamlit
- **Heroku** - Deploy Streamlit app
- **Google Cloud Run** - Containerized deployment
- **Azure App Service** - Host on Azure

### Option 4: MLflow Model Registry
```python
import mlflow.pyfunc

# Load model from registry
model_uri = "models:/BERTopic-Model/latest"
model = mlflow.pyfunc.load_model(model_uri)
```

## 📖 Documentation

- `README.md` - This file
- `DEPLOYMENT_CHECKLIST.md` - Compliance verification
- `MLOPS_ASSESSMENT.md` - Feature completeness assessment
- `pytest.ini` - Test configuration

## 🤝 Contributing

1. Create feature branch
2. Add tests in `tests/`
3. Run test suite: `pytest tests/`
4. Submit PR

## 📝 License

This project is provided as-is for educational purposes.

## ❓ Support

For issues or questions:
1. Check DEPLOYMENT_CHECKLIST.md for requirements
2. Review test cases in `tests/` for usage examples
3. Check MLflow UI for experiment tracking

---

**Status**: ✅ Production Ready (v1.0.0)
**Last Updated**: December 2024
