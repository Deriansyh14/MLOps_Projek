import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# --- 1. CONFIG & CSS LOADER ---
st.set_page_config(page_title="BERTopic Analysis", layout="wide", initial_sidebar_state="expanded")

def load_css(file_name):
    """Fungsi untuk memanggil file CSS eksternal"""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File CSS tidak ditemukan di: {file_name}")

# Panggil file CSS dari folder assets
load_css("ui/style.css")

# --- 2. FUNGSI EDA (CACHED) ---
@st.cache_data
def calculate_eda_metrics(text_list):
    text_list = [str(x) for x in text_list]
    doc_lens = [len(d.split()) for d in text_list]
    
    try:
        vec = CountVectorizer(stop_words='english', max_features=20) 
        X = vec.fit_transform(text_list)
        sum_words = X.sum(axis=0) 
        words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        df_freq = pd.DataFrame(words_freq, columns=['Kata', 'Jumlah'])
    except ValueError:
        df_freq = pd.DataFrame(columns=['Kata', 'Jumlah'])
    
    return doc_lens, df_freq

@st.cache_data
def generate_wordcloud_img(text_list):
    text_sample = " ".join([str(t) for t in text_list][:5000]) 
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100, collocations=False).generate(text_sample)
    return wc

# --- 3. MAIN APPLICATION ---
def main():
    if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
    if "cache_data" not in st.session_state: st.session_state.cache_data = None
    if "titles" not in st.session_state: st.session_state.titles = []
    if "abstracts" not in st.session_state: st.session_state.abstracts = []

    st.title("📑 Topic Modeling Dashboard")
    st.caption("Upload -> Analisis Kualitas Data (EDA) -> Training Model")

    # --- SIDEBAR & API KEY ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Priority: Input Box > Environment Variable > None
        env_key = os.getenv("GROQ_API_KEY", "")
        user_key = st.text_input("Groq API Key (Optional)", value=env_key, type="password", help="Untuk nama topik otomatis yang cerdas.")
        
        groq_api_key = user_key if user_key.strip() else None

    # --- TABS ---
    tab_process, tab_eval = st.tabs(["🚀 Data & Training", "📊 Evaluasi Model"])
    
    with tab_process:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Upload CSV (Min. kolom: 'Abstract')", type=["csv"])

        if uploaded_file is not None:
            if 'df_raw' not in st.session_state: st.session_state.df_raw = pd.read_csv(uploaded_file)
            df = st.session_state.df_raw
            st.success(f"File uploaded: {len(df)} documents loaded.")

            # --- EDA ---
            st.header("2. Exploratory Data Analysis (EDA)")
            possible_cols = [c for c in df.columns if c.lower() in ['abstract', 'text', 'content', 'body', 'title']]
            default_ix = possible_cols.index('Abstract') if 'Abstract' in possible_cols else 0
            text_col = st.selectbox("Pilih Kolom Teks:", possible_cols, index=default_ix)

            if text_col:
                docs_eda = df[text_col].dropna().tolist()
                doc_lens, df_freq = calculate_eda_metrics(docs_eda)

                c1, c2 = st.columns(2)
                with c1:
                    fig_hist = px.histogram(x=doc_lens, nbins=50, title="Distribusi Panjang Kalimat")
                    st.plotly_chart(fig_hist, width="stretch") 
                with c2:
                    if not df_freq.empty:
                        fig_bar = px.bar(df_freq.head(15), x='Jumlah', y='Kata', orientation='h', title="Top 15 Words")
                        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_bar, width="stretch")

                if st.checkbox("Show WordCloud"):
                    wc = generate_wordcloud_img(docs_eda)
                    st.image(wc.to_array(), width="stretch")

            st.divider()

            # --- TRAINING ---
            st.header("3. Mulai Training Model")
            if st.button("▶ Start BERTopic Analysis", type="primary"):
                with st.spinner("⏳ Sedang memproses..."):
                    # Import Logic dari Backend (Bersih dari UI)
                    from backend.modeling.text_cleaning import preprocess_dataframe, combine_docs
                    
                    df_clean = preprocess_dataframe(df)
                    docs = combine_docs(df_clean)
                    st.session_state.titles = df_clean["Title"].tolist() if "Title" in df_clean.columns else [""] * len(df)
                    st.session_state.abstracts = df_clean["Abstract"].tolist() if "Abstract" in df_clean.columns else [""] * len(df)

                    from backend.modeling.bertopic_analysis import bertopic_analysis, generate_topics_with_label
                    
                    # Backend hanya mengembalikan DATA, bukan UI
                    result = bertopic_analysis(docs, max_trials=None) 

                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                        st.stop()
                    
                    st.session_state.analysis_result = result
                    st.session_state.cache_data = result["cache_data"]
                    
                    # Modeling & Labeling
                    topic_result = generate_topics_with_label(
                        docs=result["cache_data"]["docs"],
                        embeddings=result["cache_data"]["embeddings"],
                        embedding_model=result["cache_data"]["embedding_model"],
                        umap_model=result["cache_data"]["umap_model"],
                        vectorizer_model=result["cache_data"]["vectorizer_model"],
                        ctfidf_model=result["cache_data"]["ctfidf_model"],
                        min_cluster_size=result["best_params"]["min_cluster_size"],
                        groq_api_key=groq_api_key 
                    )

                    if isinstance(topic_result, dict) and "error" in topic_result:
                        st.error(topic_result['error'])
                    else:
                        st.session_state.topic_model, st.session_state.topic_info, st.session_state.topics, _ = topic_result
                        st.success("✅ Analisis Selesai! Cek tab Evaluasi.")

    with tab_eval:
        st.header("Hasil Analisis")
        if st.session_state.analysis_result is None:
            st.info("Belum ada model.")
        else:
            result = st.session_state.analysis_result
            topic_info = st.session_state.topic_info
            
            # Metrics UI
            c1, c2 = st.columns(2)
            c1.metric("Coherence Score", f"{result['best_params']['coherence_score']:.4f}")
            c2.metric("Min Cluster Size", result['best_params']['min_cluster_size'])

            if result["plot_html"]:
                with st.expander("📈 Grafik Tuning"):
                    st.components.v1.html(result["plot_html"], height=400)

            st.subheader("Daftar Topik")
            clean_topics = topic_info[topic_info["Topic"] != -1].copy()
            st.dataframe(clean_topics[["Topic", "Count", "Name"]], width="stretch", hide_index=True)

            st.subheader("Lihat Isi Dokumen per Topik")
            topic_opts = clean_topics["Topic"].tolist()
            if topic_opts:
                sel_topic = st.selectbox("Pilih Topik:", topic_opts)
                indices = [i for i, t in enumerate(st.session_state.topics) if t == sel_topic]
                
                docs_display = []
                for idx in indices[:10]:
                    docs_display.append({
                        "Title": st.session_state.titles[idx], 
                        "Abstract": st.session_state.abstracts[idx][:200] + "..."
                    })
                st.dataframe(pd.DataFrame(docs_display), width="stretch", hide_index=True)

if __name__ == "__main__":
    main()