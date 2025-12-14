
# TEXT CLEANING & TOKENIZATION

import os
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# NLTK SETUP (MODEL REGISTERED LOCALLY)
NLTK_PATH = os.path.join(os.getcwd(), "save_models", "nltk_data")
nltk.data.path.append(NLTK_PATH)

for res in ["tokenizers/punkt", "corpora/stopwords", "corpora/wordnet"]:
    nltk.data.find(res)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# CLEANING
def clean_text(text):
    if pd.isnull(text):
        return ""

    text = re.sub(r'©.*', '', text)
    text = re.sub(r'Keywords?:.*', '', text, flags=re.I)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def preprocess_dataframe(df):
    df = df.copy()

    df["Title"] = df["Title"].astype(str).apply(clean_text)
    df["Abstract"] = df["Abstract"].astype(str).apply(clean_text)

    df = df.dropna()
    df = df[(df["Title"] != "") & (df["Abstract"] != "")]
    df = df.drop_duplicates()

    if len(df) < 5:
        raise ValueError("Minimal 5 dokumen")

    return df


def combine_docs(df):
    return (df["Title"] + " " + df["Abstract"]).tolist()


# TOKENIZER FOR COHERENCE (EVALUATION)
def simple_tokenizer(docs):
    tokenized = []

    for doc in docs:
        doc = doc.lower()
        doc = re.sub(r'[^a-z\s]', '', doc)

        tokens = word_tokenize(doc)
        tokens = [
            lemmatizer.lemmatize(w, pos="v")
            for w in tokens
            if w not in stop_words and len(w) > 2
        ]
        tokenized.append(tokens)

    return tokenized
