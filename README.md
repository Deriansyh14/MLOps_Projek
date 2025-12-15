#  Pipeline MLOps untuk BERTopic Modeling 

Sebuah proyek MLOps komprehensif yang mengimplementasikan topic modeling menggunakan BERTopic dengan infrastruktur, mencakup pelacakan model, monitoring, CLI, pengujian, dan kemampuan deployment

## Latar Belakang Proyek
Di era Big Data, analisis volume data teks yang masif telah menjadi kebutuhan mendesak. Topic Modeling, khususnya menggunakan BERTopic yang memanfaatkan kekuatan transformer embeddings, adalah solusi mutakhir untuk mengekstrak wawasan terstruktur. Namun, keberhasilan model machine learning (ML) di dunia nyata sangat bergantung pada implementasi operasional yang solid—sebuah disiplin yang dikenal sebagai MLOps.

Proyek ini bertujuan untuk menjembatani kesenjangan antara pengembangan model BERTopic dan implementasi produksi yang tangguh. Kami menyediakan pipeline MLOps ujung-ke-ujung (end-to-end) yang dirancang untuk mengatasi tantangan utama, seperti Topic Drift (pergeseran topik), reproduktifitas yang buruk, dan kesulitan memantau kinerja model setelah deployment.

## Arsitektur MLOps
[cite_start]Proyek ini mengikuti alur kerja MLOps standar yang terdiri dari tiga fase utama:

1. **Build (Pipeline Pelatihan):**
   - Preprocessing data teks otomatis.
   - Pelatihan model BERTopic.
   - Hyperparameter tuning dengan logging ke **MLflow**.
   - Model registering untuk versi terbaik.
2. **Deploy (Pipeline Produksi):**
   - Otomatisasi testing dengan **GitHub Actions (CI/CD)**.
   - Deployment aplikasi antarmuka menggunakan **Streamlit**.
3. **Monitor (Feedback Loop):**
   - Pemantauan berkala terhadap *Topic Drift* dan *Data Drift*.
   - Evaluasi kualitas model (*Coherence Score*) pada data baru.

## Quick Start

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

Aplikasi akan terbuka di `http://localhost:8501` dengan 3 tab:
- **Training & Analysis** - Unggah CSV dan latih model
- **Model Monitoring** - Inferensi dan deteksi drift
- **MLflow Tracking** - Lihat info pelacakan eksperimen

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

## Project Structure

```
MLOps/
├── app.py                      # Aplikasi utama Streamlit
├── requirements.txt            # Dependensi Python
├── pytest.ini                  # Konfigurasi Pytest
├── README.md                   # Dokumentasi Proyek
├── DEPLOYMENT_CHECKLIST.md     # Daftar periksa kepatuhan
│
├── backend/
│   ├── eda/                    # Skrip Exploratory Data Analysis
│   ├── modeling/               # Analisis BERTopic & Preprocessing
│   ├── monitoring/             # Deteksi drift & monitoring
│   └── registry/               # Fungsi registry model
│
├── cli/
│   └── bertopic_cli.py         # CLI berbasis Typer
│
├── mlops/
│   ├── experiment_tracking.py  # Logging eksperimen MLflow
│   └── monitoring.py           # Utilitas monitoring dasar
│
├── tests/                      # Unit & Integration Tests
│   ├── test_preprocessing.py
│   ├── test_bertopic_analysis.py
│   └── test_cli.py
│
├── data/                       # Penyimpanan Data
│   ├── raw/                    # Data input
│   ├── processed/              # Data bersih
│   └── logs/                   # Log sistem
│
├── save_models/                # Artefak Model Tersimpan
└── .github/
    └── workflows/
        └── ml-ci.yml           # GitHub Actions CI/CD
```

## Key Components

### 1. **Model Training & Analysis** (`backend/modeling/bertopic_analysis.py`)

```python
from backend.modeling.bertopic_analysis import bertopic_analysis, generate_topics_with_label

# Hyperparameter tuning otomatis dengan evaluasi koherensi
result = bertopic_analysis(docs, max_trials=None)

# Hasilkan model akhir dengan label AI via Groq
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

**Fitur:**
- Model Lazy-loaded (penggunaan memori yang efisien)
- Hyperparameter tuning berbasis skor koherensi
- Pelacakan metrik & artefak menggunakan MLflow
- Registry model otomatis dengan sistem versi (versioning)
- API Groq untuk pelabelan topik cerdas (dengan metode cadangan/fallback)
- Deduplikasi kata kunci

### 2. **Monitoring & Drift Detection** (`backend/monitoring/model_monitoring.py`)

```python
from backend.monitoring.model_monitoring import InferenceMonitor

# Inisialisasi dengan model yang sudah dilatih
monitor = InferenceMonitor(topic_model, baseline_topics)

# Jalankan inferensi dengan deteksi drift
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
- Waktu Inferensi
- Jumlah Dokumen/Topik
- Tingkat Drift (LOW/MEDIUM/HIGH)

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

Rangkaian tes untuk:
- Pra-pemrosesan & pembersihan teks
- Fungsi analisis BERTopic
- Monitoring & deteksi drift
- Perintah CLI
- Validasi data

Run tests:
```bash
pytest tests/ -v --cov=backend --cov=cli
```

### 5. **Streamlit Application** (`app.py`)

**Tab 1: Pelatihan & Analisis**
- Unggah CSV dengan kolom Title & Abstract
- Hyperparameter tuning otomatis
- Visualisasi skor koherensi
-Tampilan label topik
-Menjelajah dokumen berdasarkan topik
-Ekspor CSV

**Tab 2: Model Monitoring**
-Inferensi data baru
- Visualisasi deteksi topic drift
- Analisis tren koherensi
- Metrik kinerja
- Data monitoring real-time

**Tab 3: MLflow Tracking**
- Info eksperimen
- Instruksi akses UI MLflow

## Data Format

Input CSV must contain:
- `Title` (string) - Judul Dokumen
- `Abstract` (string) - Abstrak Dokumen

Example:
```csv
Title,Abstract
"Analisis Sentimen", "Penelitian ini membahas..."
"Deep Learning untuk Visi", "Model CNN digunakan untuk..."
```

## API Keys

**Groq API (optional for topic labeling):**
1. Get key from https://console.groq.com
2. Provide via:
   - Streamlit sidebar input
   - Environment variable: `GROQ_API_KEY`
   - Fallback: Auto-use word-based labels if API unavailable

**Groq API (opsional untuk pelabelan topik):**
1. Dapatkan kunci dari https://console.groq.com
2. Masukkan melalui:
    - Input sidebar di Streamlit
    - Variabel lingkungan: GROQ_API_KEY
    - Fallback: Menggunakan label berbasis kata secara otomatis jika API tidak tersedia

## MLflow Integration

All training runs are automatically logged to MLflow:

```bash
# Start MLflow UI
mlflow ui --host 127.0.0.1 --port 5000
```

View:
- Metrik eksperimen (koherensi, min_cluster_size, dll.)
- Artefak (plot koherensi, CSV info topik)
- Versi model (BERTopic-Model-v1, v2, dll.)
- Parameter proses dan stempel waktu (timestamps)

## Kesiapan Produksi
- **Model Serving**: Inferensi online via Streamlit + monitoring
- **Antarmuka CLI**: Dukungan baris perintah penuh berbasis Typer
- **Pengujian**: Paket pytest komprehensif dengan 20+ tes
- **Monitoring**: Deteksi topic drift, pelacakan koherensi, metrik kinerja
- **Pelacakan**: Pelacakan eksperimen & registry model MLflow
- **Dokumentasi**: README lengkap dan panduan deployment
- **Penanganan Kesalahan**: Fallback yang aman untuk kegagalan API
- **Logging**: Konsol + Logging JSON + Pelacakan MLflow
## Deployment Options

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

- **AWS EC2** - Hosting di instance EC2 dengan Streamlit
- **Heroku** - Deploy aplikasi Streamlit
- **Google Cloud Run** - Deployment terkontainerisasi
- **Azure App Service** - Hosting di Azure

### Option 4: MLflow Model Registry
```python
import mlflow.pyfunc

# Load model from registry
model_uri = "models:/BERTopic-Model/latest"
model = mlflow.pyfunc.load_model(model_uri)
```

##  Documentation

- `README.md` - This file
- `DEPLOYMENT_CHECKLIST.md` - Compliance verification
- `MLOPS_ASSESSMENT.md` - Feature completeness assessment
- `pytest.ini` - Test configuration

## Contributing

1. Create feature branch
2. Add tests in `tests/`
3. Run test suite: `pytest tests/`
4. Submit PR

##  License

This project is provided as-is for educational purposes.

##  Support

Untuk masalah atau pertanyaan:
1. Periksa DEPLOYMENT_CHECKLIST.md untuk persyaratan
2. Tinjau kasus uji di tests/ untuk contoh penggunaan
3. Periksa UI MLflow untuk pelacakan eksperimen

## Acknowledgment
Pengembangan proyek ini didukung oleh Program Studi Sains Data, Institut Teknologi Sumatera (ITERA).

Institut Teknologi Sumatera menyediakan lingkungan akademik yang mendorong penelitian lintas disiplin dan memungkinkan implementasi praktis sains data, pembelajaran mesin, dan metodologi MLOps dalam proyek terapan.


## Contributors 
This project was developed by the MLOps Team (Group 8):
1. M. Deriansyah Okutra (122450101)
2. Danang Hilal Kurniawan (122450085)
3. Irvan Alfaritzi (122450093)
4. Chevando Daffa Pramanda (122450096)
5. Smertniki Javid Ahmedthian (122450115)
---

**Last Updated**: December 2025
