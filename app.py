import multiprocessing
try:
    # Memastikan multiprocessing spawn method diatur
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. CONFIG & CSS LOADER ---
st.set_page_config(page_title="BERTopic Analysis", layout="wide", initial_sidebar_state="expanded")

def load_css(file_name):
    """Fungsi untuk memanggil file CSS eksternal"""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File CSS tidak ditemukan di: {file_name}")

load_css("ui/style.css")

# --- 2. FUNGSI EDA (CACHED) ---
@st.cache_data
def calculate_eda_metrics(text_list):
    # Logika EDA Anda
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
    # Inisialisasi State yang dibutuhkan
    if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
    if "topic_info" not in st.session_state: st.session_state.topic_info = None
    if "topics" not in st.session_state: st.session_state.topics = None 
    if "titles" not in st.session_state: st.session_state.titles = []
    if "abstracts" not in st.session_state: st.session_state.abstracts = []

    st.title("Topic Modeling Dashboard")
    st.caption("Aplikasi pemodelan topik terintegrasi BERTopic dan LLM.")

    # --- SIDEBAR & API KEY ---
    with st.sidebar:
        st.header("Konfigurasi")
        env_key = os.getenv("GROQ_API_KEY", "")
        user_key = st.text_input("Groq API Key (Optional)", value=env_key, type="password", help="Untuk nama topik otomatis yang cerdas.")
        groq_api_key = user_key if user_key.strip() else None

    # --- TABS ---
    tab_process, tab_eval = st.tabs(["Data & Training", "Evaluasi Model"])
    
    with tab_process:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload CSV (Min. kolom: 'Abstract')", type=["csv"])

        if uploaded_file is not None:
            if 'df_raw' not in st.session_state: st.session_state.df_raw = pd.read_csv(uploaded_file)
            df = st.session_state.df_raw
            st.success(f"File uploaded: {len(df)} documents loaded.")

            # --- EDA ---
            st.header("Analisis Kualitas Data (EDA)")
            possible_cols = [c for c in df.columns if c.lower() in ['abstract', 'text', 'content', 'body', 'title']]
            # PERBAIKAN: Default index agar tidak error jika 'Abstract' tidak ada
            default_ix = 0
            if 'Abstract' in possible_cols:
                 default_ix = possible_cols.index('Abstract')
            elif 'abstract' in possible_cols:
                 default_ix = possible_cols.index('abstract')
            
            text_col = st.selectbox("Pilih Kolom Teks:", possible_cols, index=default_ix)

            if text_col:
                docs_eda = df[text_col].dropna().tolist()
                doc_lens, df_freq = calculate_eda_metrics(docs_eda)

                c1, c2 = st.columns(2)
                with c1:
                    fig_hist = px.histogram(x=doc_lens, nbins=50, title="Distribusi Panjang Kalimat")
                    # PERUBAHAN: use_container_width=True -> width='stretch'
                    st.plotly_chart(fig_hist, width='stretch') 
                with c2:
                    if not df_freq.empty:
                        fig_bar = px.bar(df_freq.head(15), x='Jumlah', y='Kata', orientation='h', title="Top 15 Kata Paling Sering Muncul")
                        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
                        # PERUBAHAN: use_container_width=True -> width='stretch'
                        st.plotly_chart(fig_bar, width='stretch')

                if st.checkbox("Tampilkan WordCloud"):
                    wc = generate_wordcloud_img(docs_eda)
                    fig_wc, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc.to_array(), interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)
            
            st.divider()

            # --- TRAINING ---
            st.header("Mulai Training Model")
            if st.button("Start BERTopic Analysis", type="primary"):
                with st.spinner("Sedang melakukan tuning dan analisis topik..."):
                    # Import Logic dari Backend
                    from backend.modeling.text_cleaning import preprocess_dataframe, combine_docs
                    from backend.modeling.bertopic_analysis import bertopic_analysis, generate_topics_with_label
                    
                    df_clean = preprocess_dataframe(df)
                    docs = combine_docs(df_clean)
                    
                    st.session_state.titles = df_clean["Title"].tolist() if "Title" in df_clean.columns else [""] * len(df)
                    st.session_state.abstracts = df_clean["Abstract"].tolist() if "Abstract" in df_clean.columns else [""] * len(df)

                    # Backend (termasuk MLflow logging dan plot generation)
                    result = bertopic_analysis(docs, max_trials=15)
                    
                    if "error" in result:
                        st.error(f"Terjadi Kesalahan: {result['error']}")
                        return
                        
                    st.session_state.analysis_result = result
                    
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
                        st.toast("Proses Analisis Selesai! Cek tab Evaluasi.")
                        st.success("Analisis Selesai! Hasil tersedia di tab **Evaluasi Model**.")

    with tab_eval:
        st.header("Hasil Analisis")
        if st.session_state.analysis_result is None or st.session_state.topic_info is None:
            st.info("Belum ada model yang dijalankan. Silakan jalankan analisis di tab Data & Training.")
        else:
            result = st.session_state.analysis_result
            topic_info = st.session_state.topic_info
            
            # --- METRICS ---
            st.subheader("Metrik Kualitas Model")
            c1, c2, c3 = st.columns(3)
            c1.metric("Skor Koherensi Terbaik", f"{result['best_params']['coherence_score']:.4f}")
            c2.metric("Min Cluster Size Terpilih", result['best_params']['min_cluster_size'])
            c3.metric("Jumlah Topik Ditemukan", len(topic_info[topic_info['Topic'] != -1]))
            st.divider()
            
            # --- DAFTAR TOPIK ---
            st.subheader("Daftar Topik")
            clean_topics = topic_info[topic_info["Topic"] != -1].copy()
            st.dataframe(
                clean_topics[["Topic", "Count", "Name"]].rename(columns={"Name": "Label Topik", "Count": "Jml Dokumen"}), 
                # PERUBAHAN: use_container_width=True -> width='stretch'
                width='stretch', 
                hide_index=True
            )

            st.divider()

            # --- DETAIL DOKUMEN PER TOPIK (Semua dokumen, tabel interaktif) ---
            st.subheader("Detail Isi Dokumen per Topik")
            topic_opts = clean_topics["Topic"].tolist()
            if topic_opts:
                # Tambahkan label topik ke selectbox untuk memudahkan identifikasi
                topic_map = clean_topics.set_index('Topic')['Name'].to_dict()
                topic_labels = [f"Topik {id}: {topic_map[id]}" for id in topic_opts]
                
                sel_topic_label = st.selectbox("Pilih Topik:", topic_labels)
                sel_topic = int(sel_topic_label.split(":")[0].split()[-1])
                
                # Mendapatkan SEMUA indeks dokumen untuk topik terpilih
                indices = [i for i, t in enumerate(st.session_state.topics) if t == sel_topic]
                
                docs_display = []
                for idx in indices: # TIDAK ADA BATAS [:10]
                    docs_display.append({
                        "Judul": st.session_state.titles[idx], 
                        # PERBAIKAN: Menggunakan label "Ringkasan (Abstract)"
                        "Ringkasan (Abstract)": st.session_state.abstracts[idx][:400] + "..." # Potongan 400 karakter
                    })
                
                df_docs = pd.DataFrame(docs_display)
                
                # Pesan info menunjukkan total dokumen
                st.info(f"Menampilkan **{len(indices)}** dokumen yang termasuk dalam {sel_topic_label}.")
                
                # Tampilkan tabel interaktif yang dapat di-scroll (default Streamlit height)
                st.dataframe(
                    df_docs, 
                    # PERUBAHAN: use_container_width=True -> width='stretch'
                    width='stretch', 
                    hide_index=True,
                    height=500 # Tinggi yang ditetapkan untuk memungkinkan scroll di dalam tabel
                )


if __name__ == "__main__":
    main()