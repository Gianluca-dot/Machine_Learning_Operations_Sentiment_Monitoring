import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
from transformers import pipeline

st.set_page_config(
    page_title="MLOps - Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

LOG_FILE = "predictions.csv"

if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=["timestamp", "text", "prediction", "confidence", "latency_ms"])
    df_init.to_csv(LOG_FILE, index=False)

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

classifier = load_sentiment_model()

st.title("📊 MLOps Sentiment Analysis & Live Monitoring")

tab_deploy, tab_monitoring = st.tabs(["🚀 Deploy & Inferenza", "📈 Dashboard Monitoraggio"])

with tab_deploy:
    st.header("Interfaccia di Predizione Sentiment")
    user_input = st.text_area("Testo di input (Inglese):", value="", height=120)
    
    if st.button("Analizza Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Inserisci un testo valido.")
        else:
            start_time = time.time()
            result = classifier(user_input)[0]
            latency = round((time.time() - start_time) * 1000, 2)
            
            label = result["label"].upper()
            confidence = round(float(result["score"]), 4)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"Esito Predizione: {label}")
            with col2:
                st.metric("Confidenza Modello", f"{confidence * 100:.2f}%")
            with col3:
                st.metric("Latenza Sistema", f"{latency} ms")
            
            new_log = pd.DataFrame([{
                "timestamp": timestamp,
                "text": user_input,
                "prediction": label,
                "confidence": confidence,
                "latency_ms": latency
            }])
            new_log.to_csv(LOG_FILE, mode="a", header=False, index=False)
            st.toast("Predizione registrata nei log di monitoraggio.")

with tab_monitoring:
    st.header("Dashboard di Monitoraggio Continuo")
    
    if os.path.exists(LOG_FILE):
        df_logs = pd.read_csv(LOG_FILE)
        
        if df_logs.empty:
            st.info("Nessun dato presente nel registro log.")
        else:
            total_requests = len(df_logs)
            avg_confidence = round(df_logs["confidence"].mean() * 100, 2)
            avg_latency = round(df_logs["latency_ms"].mean(), 2)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Totale Richieste", total_requests)
            m2.metric("Confidenza Media", f"{avg_confidence}%")
            m3.metric("Latenza Media", f"{avg_latency} ms")
            
            st.markdown("---")
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.subheader("Distribuzione Output (Concept Drift)")
                st.bar_chart(df_logs["prediction"].value_counts())
                
            with col_g2:
                st.subheader("Andamento Latenza (ms)")
                st.line_chart(df_logs["latency_ms"])
            
            st.subheader("📋 Registro Log di Inferenza")
            st.dataframe(df_logs.sort_values(by="timestamp", ascending=False), use_container_width=True)
