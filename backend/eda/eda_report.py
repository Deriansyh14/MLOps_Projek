# backend/eda/eda_report.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

def generate_eda_report(df, output_dir="data/logs"):
    """
    Generate comprehensive EDA report (Optimized for NLP tasks)
    Includes: Basic stats, Word counts, Top N-grams, and WordCloud.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Basic Info
    report = {
        "timestamp": timestamp,
        "basic_info": {
            "total_records": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum())
        },
        "text_statistics": {},
        "top_words": []
    }
    
    # 2. Text Statistics (Word Count & Char Count)
    # Fokus pada kolom Abstract jika ada, jika tidak pakai Title
    target_col = "Abstract" if "Abstract" in df.columns else "Title"
    
    if target_col in df.columns:
        # Hitung jumlah kata (Word Count) - Lebih relevan untuk NLP
        # astype(str) untuk handle jika ada data non-string/NaN
        df["word_count"] = df[target_col].astype(str).apply(lambda x: len(x.split()))
        df["char_count"] = df[target_col].astype(str).str.len()
        
        report["text_statistics"][target_col] = {
            "mean_word_count": float(df["word_count"].mean()),
            "max_word_count": int(df["word_count"].max()),
            "min_word_count": int(df["word_count"].min()),
            "mean_char_count": float(df["char_count"].mean())
        }

    # 3. Top Words Extraction (Optimized using CountVectorizer)
    try:
        # Gabungkan Title dan Abstract untuk analisis kata menyeluruh
        text_corpus = df["Title"].fillna("") + " " + df.get("Abstract", "").fillna("")
        
        # Menggunakan CountVectorizer (C-optimized) jauh lebih cepat daripada loop manual
        vec = CountVectorizer(stop_words='english', max_features=20)
        X = vec.fit_transform(text_corpus)
        
        sum_words = X.sum(axis=0) 
        words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        report["top_words"] = words_freq # Simpan list top words ke JSON
        
    except Exception as e:
        print(f"Warning: Could not generate Top Words: {e}")
        words_freq = []

    # 4. Save Report JSON
    report_path = output_dir / f"eda_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # 5. Generate Visualizations (Combined Plot)
    # Buat Grid Layout: 2 Baris. Atas (Histogram & Bar Chart), Bawah (WordCloud)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot A: Histogram Distribusi Panjang Kata
    ax1 = fig.add_subplot(gs[0, 0])
    if "word_count" in df.columns:
        sns.histplot(df["word_count"], bins=30, kde=True, ax=ax1, color='skyblue')
        ax1.set_title(f"Distribution of Word Counts ({target_col})")
        ax1.set_xlabel("Number of Words")
    
    # Plot B: Bar Chart Top 15 Words
    ax2 = fig.add_subplot(gs[0, 1])
    if words_freq:
        df_freq = pd.DataFrame(words_freq[:15], columns=['Term', 'Count'])
        sns.barplot(x='Count', y='Term', data=df_freq, ax=ax2, palette='viridis')
        ax2.set_title("Top 15 Most Frequent Words")
    
    # Plot C: Word Cloud
    ax3 = fig.add_subplot(gs[1, :]) # Span seluruh baris bawah
    try:
        # Generate wordcloud (Sample max 10k docs to save memory if dataset huge)
        text_sample = " ".join(text_corpus.sample(n=min(len(text_corpus), 5000), random_state=42).tolist())
        wordcloud = WordCloud(width=800, height=300, background_color='white', max_words=100).generate(text_sample)
        ax3.imshow(wordcloud, interpolation='bilinear')
        ax3.axis("off")
        ax3.set_title("Word Cloud Visualization")
    except Exception as e:
        ax3.text(0.5, 0.5, "Could not generate WordCloud", ha='center')
    
    plt.tight_layout()
    plot_path = output_dir / f"eda_plots_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"✓ EDA report saved to: {report_path}")
    print(f"✓ EDA visualizations saved to: {plot_path}")
    
    return report