#!/usr/bin/env python3
"""
BERTopic MLOps CLI Entry Point
Run: python cli/run_bertopic.py train --input data/raw/data.csv
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.bertopic_cli import app

if __name__ == "__main__":
    app()
