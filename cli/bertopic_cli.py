# Imports
import typer
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import json

mlflow.set_tracking_uri("data/logs")

# Mengimport fungsi inti BERTopic
from backend.modeling.bertopic_analysis import (
    bertopic_analysis,
    generate_topics_with_label,
    # register_model_to_mlflow # <-- DIHAPUS DARI SINI
)

# Mengimport fungsi MLflow dari modul yang seharusnya
from backend.monitoring.model_monitoring import register_model_to_mlflow 


app = typer.Typer(
    help="BERTopic MLOps CLI - Train (Blue/Green) & Predict",
    pretty_exceptions_enable=True
)

@app.command()
def train(
    input_file: str = typer.Option(
        "data/raw/data.csv",
        "--input", "-i",
        help="Path to input CSV file (requires Title, Abstract columns)"
    ),
    output_dir: str = typer.Option(
        "save_models",
        "--output", "-o",
        help="Directory to save local artifacts"
    ),
    max_trials: Optional[int] = typer.Option(
        10, 
        "--trials", "-t",
        help="Max hyperparameter trials for Green Model (None = full evaluation)"
    ),
    experiment_name: str = typer.Option(
        "BERTopic-Analysis",
        "--experiment", "-e",
        help="MLflow experiment name"
    ),
):
    """
    Train using Blue/Green Strategy:
    1. Blue: Model Standar (Fixed Params) -> Production
    2. Green: Model Tuned (Optuna/Search) -> Staging
    """
    try:
        typer.echo(f" Starting Blue/Green Training Pipeline...")
        
        # 1. Validate & Load Data
        if not Path(input_file).exists():
            typer.echo(f" Error: File not found: {input_file}", err=True)
            raise typer.Exit(code=1)
            
        typer.echo(" Loading & Preprocessing data...")
        df = pd.read_csv(input_file)
        
        from backend.modeling.text_cleaning import preprocess_dataframe, combine_docs
        df_clean = preprocess_dataframe(df)
        docs = combine_docs(df_clean)
        typer.echo(f"✓ Loaded {len(docs)} documents")

        # 2. Setup MLflow
        mlflow.set_experiment(experiment_name)
        client = MlflowClient()
        
        # Hapus import berulang, karena sudah diimport secara global di awal file ini.
        # from backend.modeling.bertopic_analysis import (
        #     bertopic_analysis,
        #     generate_topics_with_label,
        #     register_model_to_mlflow 
        # )

        # PHASE 1: MODEL BLUE (BASELINE / STANDARD)
        typer.echo("\n [BLUE] Training Model Standar (Baseline)...")
        
        with mlflow.start_run(run_name="train-blue-baseline"):
            # Hardcode parameter standar (misal 15)
            STANDARD_MIN_CLUSTER = 15
            
            # Generate Model Blue - will auto-load embeddings/models inside
            result_blue = generate_topics_with_label(
                docs=docs,
                min_cluster_size=STANDARD_MIN_CLUSTER,
                groq_api_key=None
            )
            
            # Check if result is dict with error
            if isinstance(result_blue, dict) and "error" in result_blue:
                typer.echo(f" Blue model training failed: {result_blue['error']}", err=True)
                raise typer.Exit(code=1)
            
            model_blue, info_blue, topics_blue, probs_blue = result_blue
            
            # Register ke MLflow
            blue_reg = register_model_to_mlflow(
                topic_model=model_blue,
                topic_info=info_blue,
                coherence_score=0.5,  # Score dummy/baseline
                min_cluster_size=STANDARD_MIN_CLUSTER,
                n_topics=len(info_blue),
                model_name="BERTopic-Model"
            )
            
            # Transition to PRODUCTION
            try:
                latest_versions = client.get_latest_versions("BERTopic-Model", stages=["None"])
                if latest_versions:
                    latest_blue = latest_versions[0].version
                    client.transition_model_version_stage(
                        name="BERTopic-Model", version=latest_blue, stage="Production"
                    )
                    typer.echo(f"✓ Model Blue (v{latest_blue}) -> PRODUCTION")
            except Exception as e:
                typer.echo(f"  Transisi Blue model: {e}") 

            # Save Local CSV (Optional)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            info_blue.to_csv(f"{output_dir}/blue_topics.csv", index=False)
            typer.echo(f"✓ Blue model artifacts saved")

        # PHASE 2: MODEL GREEN (TUNED / OPTIMIZED)
        typer.echo("\n [GREEN] Hyperparameter Tuning...")
        
        # Cari parameter terbaik
        tuning_result = bertopic_analysis(docs, max_trials=max_trials)
        
        if "error" in tuning_result:
             typer.echo(f" Tuning failed: {tuning_result['error']}", err=True)
             raise typer.Exit(code=1)

        best_size = tuning_result["best_params"]["min_cluster_size"]
        best_score = tuning_result["best_params"]["coherence_score"]
        cache = tuning_result["cache_data"]
        typer.echo(f"   -> Best min_cluster_size: {best_size} | Coherence: {best_score:.4f}")
        
        with mlflow.start_run(run_name="train-green-tuned"):
            # Generate Model Green pakai Best Params - pass cached data
            result_green = generate_topics_with_label(
                docs=cache["docs"],
                embeddings=cache["embeddings"],
                embedding_model=cache["embedding_model"],
                umap_model=cache["umap_model"],
                vectorizer_model=cache["vectorizer_model"],
                ctfidf_model=cache["ctfidf_model"],
                min_cluster_size=best_size,
                groq_api_key=None
            )
            
            # Check if result is dict with error
            if isinstance(result_green, dict) and "error" in result_green:
                typer.echo(f" Green model training failed: {result_green['error']}", err=True)
                raise typer.Exit(code=1)
            
            model_green, info_green, topics_green, probs_green = result_green
            
            # Register ke MLflow
            green_reg = register_model_to_mlflow(
                topic_model=model_green,
                topic_info=info_green,
                coherence_score=best_score,
                min_cluster_size=best_size,
                n_topics=len(info_green),
                model_name="BERTopic-Model"
            )
            
            # Transition to STAGING
            try:
                latest_versions = client.get_latest_versions("BERTopic-Model", stages=["None"])
                if latest_versions:
                    latest_green = latest_versions[0].version
                    client.transition_model_version_stage(
                        name="BERTopic-Model", version=latest_green, stage="Staging"
                    )
                    typer.echo(f"✓ Model Green (v{latest_green}) -> STAGING")
            except Exception as e:
                typer.echo(f"  Transisi Green model: {e}")

            # Save Local CSV
            info_green.to_csv(f"{output_dir}/green_topics.csv", index=False)
            typer.echo(f"✓ Green model artifacts saved")

        typer.echo("\n Blue/Green Pipeline Completed Successfully!")
        
    except Exception as e:
        typer.echo(f" Error: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def predict(
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input CSV"),
    stage: str = typer.Option(
        "Production", 
        "--stage", "-s", 
        help="Select Stage: 'Production' (Blue) or 'Staging' (Green)"
    ),
    model_name: str = typer.Option("BERTopic-Model", "--name", "-n", help="MLflow Model Name"),
    output_file: str = typer.Option("results/predictions.csv", "--output", "-o", help="Output file"),
):
    """
    Run inference using Blue (Production) or Green (Staging) model.
    """
    try:
        typer.echo(f" Loading Data: {input_file}")
        
        # 1. Load Data
        if not Path(input_file).exists():
            typer.echo(f" File not found: {input_file}", err=True)
            raise typer.Exit(code=1)
        df = pd.read_csv(input_file)
        
        from backend.modeling.text_cleaning import preprocess_dataframe, combine_docs
        df_clean = preprocess_dataframe(df)
        docs = combine_docs(df_clean)
        typer.echo(f"✓ Loaded {len(docs)} documents")
        
        # 2. Load Model from MLflow Stage
        model_uri = f"models:/{model_name}/{stage}"
        typer.echo(f" Loading Model: {model_name}/{stage}")
        typer.echo(f"  Stage: {stage} " + ("(🔵 BLUE)" if stage == "Production" else "(🟢 GREEN)"))
        
        try:
            loaded_model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            typer.echo(f" Failed to load model from stage '{stage}': {e}", err=True)
            # Fallback: try to load the latest version
            typer.echo(f"  Attempting to use latest model version...")
            try:
                loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
            except:
                raise typer.Exit(code=1)

        # 3. Predict
        typer.echo(" Running Inference...")
        try:
            predictions = loaded_model.predict(docs)
            topics = predictions[0] if isinstance(predictions, tuple) else predictions
        except Exception as e:
            typer.echo(f" Inference failed: {e}", err=True)
            raise typer.Exit(code=1)

        # 4. Save Results
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df_result = pd.DataFrame({"Document": docs, "Topic": topics})
        df_result.to_csv(output_file, index=False)
        
        typer.echo(f" Results saved to {output_file}")
        typer.echo(f" Topics found: {len(set(topics))}")
        
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f" Error: {str(e)}", err=True)
        raise typer.Exit(code=1)


@app.command()
def monitor(
    input_file: str = typer.Option("data/raw/data.csv", "--input", "-i"),
    output_file: str = typer.Option("data/logs/monitoring_report.json", "--output", "-o"),
):
    """
    Monitor model performance (Placeholder).
    """
    typer.echo("  Please use Streamlit App (app.py) for Visual Monitoring.")


@app.command()
def evaluate(
    metrics_file: str = typer.Option("data/logs/experiment.json", "--metrics", "-m"),
):
    """
    Evaluate model from logs (Placeholder).
    """
    typer.echo("Please check MLflow UI ('mlflow ui') for detailed evaluation.")


@app.command()
def version():
    typer.echo(" BERTopic MLOps CLI v1.0.0 (Blue/Green Enabled)")

if __name__ == "__main__":
    app()