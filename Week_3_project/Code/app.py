import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from preprocessor import clean_text, prepare_input_text, TextCNN, text_to_sequence

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NewsPulse AI | DL News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Result Card */
    .result-card {
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .result-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-high {
        background-color: #10b981;
        color: white;
    }
    .badge-med {
        background-color: #f59e0b;
        color: white;
    }
    .badge-low {
        background-color: #ef4444;
        color: white;
    }
    
    /* Keyword Chips */
    .keyword-chip {
        display: inline-block;
        background: rgba(100, 116, 139, 0.15);
        color: #3b82f6;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching & Artifact Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_all_artifacts():
    # 1. Load Scikit-Learn ML Model & TF-IDF Vectorizer
    ml_model = joblib.load(os.path.join(BASE_DIR, "lr_news_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    
    # 2. Load PyTorch TextCNN DL Model & Vocab Config
    with open(os.path.join(BASE_DIR, "vocab.json"), "r", encoding="utf-8") as f:
        dl_cfg = json.load(f)
        
    dl_model = TextCNN(
        vocab_size=len(dl_cfg["vocab"]),
        embed_dim=dl_cfg["embed_dim"],
        num_classes=dl_cfg["num_classes"]
    )
    dl_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "textcnn_model.pt"), map_location=torch.device("cpu")))
    dl_model.eval()
    
    # 3. Load Training Metrics for Analytics Dashboard
    metrics = None
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    return ml_model, vectorizer, label_encoder, dl_model, dl_cfg, metrics

try:
    ml_model, tfidf_vec, label_enc, dl_model, dl_cfg, metrics_data = load_all_artifacts()
    classes = [c.capitalize() for c in label_enc.classes_]
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts: {e}")
    st.info("Please ensure all `.pkl`, `.pt`, and `.json` model files exist in the `Code/` directory.")
    st.stop()

# Category Icon Mapping
CAT_ICONS = {
    "Business": "💼",
    "World": "🌍",
    "Entertainment": "🎬",
    "Politics": "🏛️",
    "Automobile": "🚗",
    "National": "🇮🇳",
    "Science": "🔬",
    "Sports": "🏏",
    "Education": "🎓",
    "Technology": "💻"
}

# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/news.png", width=110)
    st.title("NewsPulse AI")
    st.caption("Deep Learning News Classification Engine")
    
    st.markdown("---")
    st.subheader("⚙️ Active Model Engine")
    model_choice = st.radio(
        "Select Classification Architecture:",
        ("🧠 PyTorch TextCNN (Deep Learning)", "⚡ Calibrated Linear Classifier (TF-IDF)"),
        index=0,
        help="Switch between Deep Learning (learned embeddings + 1D convolutions) and fast Calibrated Linear classification."
    )
    
    is_dl = "Deep Learning" in model_choice
    
    st.markdown("---")
    st.subheader("📌 Model Specifications")
    if is_dl:
        st.markdown("""
        - **Architecture:** PyTorch TextCNN (1D Multi-Filter CNN)
        - **Embedding Dim:** 128
        - **Filter Sizes:** [2, 3, 4] (Bi-gram, Tri-gram, 4-gram feature maps)
        - **Accuracy:** **99.69%** (Test Set)
        - **Inference Mode:** CPU Evaluated
        """)
    else:
        st.markdown("""
        - **Architecture:** Calibrated Linear Classifier (LinearSVC + Platt Scaling)
        - **Feature Space:** 15,000 sublinear TF-IDF N-grams (1, 2)
        - **Accuracy:** **99.71%** (Test Set)
        - **Latency:** < 5 milliseconds
        """)
        
    st.markdown("---")
    with st.expander("ℹ️ About This Project"):
        st.write("""
        **Decodelabs Data Science Internship**  
        *Week 3 Project: NLP & Deep Learning Document Classification*
        
        Trained on the Inshorts News Dataset (~66,000 articles) categorized across 10 distinct, clean news domains.
        """)

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">📰 NewsPulse AI Classifier</h1>
    <p class="hero-subtitle">High-accuracy document categorization powered by Deep Learning (PyTorch TextCNN) and Calibrated Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ Real-Time Classifier",
    "📁 Batch File Classifier",
    "⚖️ Side-by-Side Comparison",
    "📊 Training & Evaluation Analytics",
    "🔬 Model Architecture"
])

# ---------------------------------------------------------
# TAB 1: Real-Time Classifier
# ---------------------------------------------------------
with tab1:
    st.markdown("### 📝 Enter News Article Content")
    st.write("Type or paste an article below, or pick a preset headline to test live categorization.")
    
    # Presets for Quick Testing
    preset_samples = {
        "🚀 ISRO Moon Mission": {
            "headline": "ISRO successfully conducts landing thruster tests for upcoming lunar lander mission",
            "content": "The Indian Space Research Organisation completed critical hot tests of the propulsion system designed to control soft touchdown on the moon surface."
        },
        "💼 Central Bank Interest Rates": {
            "headline": "Federal Reserve maintains benchmark interest rates amid cooling inflation trends",
            "content": "Central bank officials decided to keep rates steady as monthly inflation figures showed continued deceleration across financial markets."
        },
        "🏏 Cricket World Cup": {
            "headline": "India secures dominant victory in World Cup semi-final against New Zealand",
            "content": "A stellar century from the top order accompanied by fiery bowling spells propelled the Indian squad into the finals of the ICC tournament."
        },
        "💻 AI Supercomputer Launch": {
            "headline": "OpenAI and Microsoft unveil next-generation supercomputing cluster for reasoning models",
            "content": "The new hardware architecture integrates thousands of liquid-cooled graphics processors optimized specifically for large-scale neural network training."
        },
        "🚗 Electric Sports Car": {
            "headline": "Porsche reveals all-electric high-performance GT vehicle with ultra-fast charging battery",
            "content": "The dual-motor electric powertrain produces over 800 horsepower and can recharge from ten to eighty percent in under fifteen minutes."
        },
        "🏛️ Supreme Court Verdict": {
            "headline": "Supreme Court constitution bench delivers unanimous ruling on parliamentary privileges",
            "content": "The seven-judge bench ruled that legislators cannot claim constitutional immunity from criminal prosecution for taking bribes in assembly votes."
        }
    }
    
    st.write("**⚡ Quick Test Presets:**")
    preset_cols = st.columns(len(preset_samples))
    selected_preset = None
    
    for i, (name, sample) in enumerate(preset_samples.items()):
        if preset_cols[i].button(name, key=f"preset_{i}", use_container_width=True):
            st.session_state["input_headline"] = sample["headline"]
            st.session_state["input_content"] = sample["content"]
            st.rerun()

    # Form inputs
    col_input_left, col_input_right = st.columns([1, 1.4])
    with col_input_left:
        input_headline = st.text_input(
            "Headline (High-Signal Context):",
            value=st.session_state.get("input_headline", ""),
            placeholder="e.g., Sensex surges 600 points as tech stocks rally..."
        )
    with col_input_right:
        input_content = st.text_area(
            "Article Body / Content:",
            value=st.session_state.get("input_content", ""),
            height=100,
            placeholder="e.g., Benchmark equity indices climbed today tracking positive global cues and sustained foreign fund inflows..."
        )
    
    predict_clicked = st.button("⚡ Classify News Article", type="primary", use_container_width=True)
    
    if predict_clicked:
        combined_text = prepare_input_text(input_headline, input_content)
        cleaned = clean_text(combined_text)
        
        if not cleaned:
            st.warning("⚠️ Please enter a headline or article content to classify.")
        else:
            t0 = time.time()
            
            if is_dl:
                # PyTorch Inference
                seq = text_to_sequence(cleaned, dl_cfg["vocab"], dl_cfg["max_len"])
                with torch.no_grad():
                    logits = dl_model(seq)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probabilities)
                pred_class = dl_cfg["classes"][pred_idx]
                pred_prob = probabilities[pred_idx]
                all_probs = probabilities
                class_names = dl_cfg["classes"]
            else:
                # Calibrated Linear Classifier Inference
                vec = tfidf_vec.transform([cleaned])
                probabilities = ml_model.predict_proba(vec)[0]
                pred_idx = np.argmax(probabilities)
                pred_class = label_enc.classes_[pred_idx].capitalize()
                pred_prob = probabilities[pred_idx]
                all_probs = probabilities
                class_names = [c.capitalize() for c in label_enc.classes_]
                
            latency_ms = (time.time() - t0) * 1000
            
            # Prediction Results Section
            icon = CAT_ICONS.get(pred_class, "📰")
            confidence_pct = pred_prob * 100
            
            if confidence_pct >= 85:
                badge_html = '<span class="result-badge badge-high">High Confidence</span>'
            elif confidence_pct >= 60:
                badge_html = '<span class="result-badge badge-med">Moderate Confidence</span>'
            else:
                badge_html = '<span class="result-badge badge-low">Low Confidence</span>'
                
            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; font-weight: 700;">PREDICTED CATEGORY</span>
                        <h2 style="margin: 4px 0 0 0; font-size: 2rem; color: #1e3a8a;">{icon} {pred_class}</h2>
                    </div>
                    <div style="text-align: right;">
                        {badge_html}
                        <h3 style="margin: 6px 0 0 0; font-size: 1.8rem; color: #0f766e;">{confidence_pct:.2f}%</h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric Columns
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Topic", pred_class, icon)
            m2.metric("Probability", f"{confidence_pct:.2f}%")
            m3.metric("Inference Time", f"{latency_ms:.1f} ms")
            m4.metric("Tokens Processed", len(cleaned.split()))
            
            # Interactive Probability Breakdown Chart
            st.markdown("#### 📊 Category Probability Distribution")
            
            prob_df = pd.DataFrame({
                "Category": class_names,
                "Probability (%)": [p * 100 for p in all_probs]
            }).sort_values(by="Probability (%)", ascending=True)
            
            fig = px.bar(
                prob_df,
                x="Probability (%)",
                y="Category",
                orientation="h",
                color="Probability (%)",
                color_continuous_scale="Tealgrn",
                text=prob_df["Probability (%)"].apply(lambda v: f"{v:.1f}%")
            )
            fig.update_layout(
                margin=dict(l=10, r=20, t=20, b=20),
                height=340,
                xaxis=dict(range=[0, 100], title="Confidence Score (%)"),
                yaxis=dict(title=""),
                coloraxis_showscale=False
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            
            # Keyword Attribution Chips
            st.markdown("#### 🔍 Salient Context Words Detected")
            tokens = cleaned.split()
            # Filter common stop-words and short tokens
            significant_words = [w for w in tokens if len(w) > 3 and w not in {"this", "that", "with", "from", "have", "been", "were", "they", "will", "what", "their"}][:15]
            if significant_words:
                chips_html = "".join([f'<span class="keyword-chip">{w}</span>' for w in set(significant_words)])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.caption("No salient keywords extracted.")

# ---------------------------------------------------------
# TAB 2: Batch File Classifier
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📁 Batch Document Categorization")
    st.write("Upload a CSV file with news headlines or articles to classify them in bulk.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with {len(batch_df)} rows.")
            st.write("Data Preview:", batch_df.head(3))
            
            col_opts = batch_df.columns.tolist()
            c1, c2 = st.columns(2)
            with c1:
                text_col = st.selectbox("Select Article Text / Content Column:", col_opts)
            with c2:
                headline_col = st.selectbox("Select Headline Column (Optional):", ["None"] + col_opts)
                
            if st.button("🚀 Run Batch Prediction", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results_cat = []
                results_conf = []
                
                total_rows = len(batch_df)
                t0_batch = time.time()
                
                for idx, row in batch_df.iterrows():
                    h_val = str(row[headline_col]) if headline_col != "None" else ""
                    c_val = str(row[text_col])
                    combo = prepare_input_text(h_val, c_val)
                    cl = clean_text(combo)
                    
                    if is_dl:
                        seq = text_to_sequence(cl, dl_cfg["vocab"], dl_cfg["max_len"])
                        with torch.no_grad():
                            l = dl_model(seq)
                            probs = torch.softmax(l, dim=1).cpu().numpy()[0]
                        best_idx = np.argmax(probs)
                        results_cat.append(dl_cfg["classes"][best_idx])
                        results_conf.append(float(probs[best_idx] * 100))
                    else:
                        v = tfidf_vec.transform([cl])
                        probs = ml_model.predict_proba(v)[0]
                        best_idx = np.argmax(probs)
                        results_cat.append(label_enc.classes_[best_idx].capitalize())
                        results_conf.append(float(probs[best_idx] * 100))
                        
                    if idx % max(1, total_rows // 20) == 0:
                        progress_bar.progress((idx + 1) / total_rows)
                        status_text.text(f"Processed {idx + 1}/{total_rows} articles...")
                        
                progress_bar.progress(1.0)
                elapsed = time.time() - t0_batch
                status_text.success(f"✅ Classified {total_rows} articles in {elapsed:.2f} seconds ({elapsed/total_rows*1000:.1f} ms/doc)!")
                
                batch_df["Predicted_Category"] = results_cat
                batch_df["Confidence_%"] = np.round(results_conf, 2)
                
                st.markdown("#### Batch Summary Distribution")
                pie_fig = px.pie(
                    batch_df, 
                    names="Predicted_Category", 
                    title="Class Distribution in Uploaded Dataset",
                    hole=0.35,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(pie_fig, use_container_width=True)
                
                st.dataframe(batch_df.head(20), use_container_width=True)
                
                # CSV Download Button
                csv_data = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Categorized CSV",
                    data=csv_data,
                    file_name="categorized_news_output.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ---------------------------------------------------------
# TAB 3: Side-by-Side Comparison
# ---------------------------------------------------------
with tab3:
    st.markdown("### ⚖️ Side-by-Side Model Comparison")
    st.write("Compare the PyTorch Deep Learning TextCNN and the Calibrated Linear Classifier on the same text input simultaneously.")
    
    test_text = st.text_input(
        "Enter text to evaluate across both models:",
        value="Tesla has revealed its next-generation electric battery with enhanced energy density for commercial trucks."
    )
    
    if test_text:
        cl_eval = clean_text(test_text)
        
        # 1. DL Inference
        t0 = time.time()
        seq = text_to_sequence(cl_eval, dl_cfg["vocab"], dl_cfg["max_len"])
        with torch.no_grad():
            l = dl_model(seq)
            dl_probs = torch.softmax(l, dim=1).cpu().numpy()[0]
        dl_time = (time.time() - t0) * 1000
        dl_top = dl_cfg["classes"][np.argmax(dl_probs)]
        
        # 2. ML Inference
        t0 = time.time()
        v = tfidf_vec.transform([cl_eval])
        ml_probs = ml_model.predict_proba(v)[0]
        ml_time = (time.time() - t0) * 1000
        ml_top = label_enc.classes_[np.argmax(ml_probs)].capitalize()
        
        comp1, comp2 = st.columns(2)
        with comp1:
            st.markdown(f"#### 🧠 PyTorch TextCNN (Deep Learning)")
            st.metric("Predicted Class", f"{CAT_ICONS.get(dl_top, '📰')} {dl_top}", f"{np.max(dl_probs)*100:.2f}%")
            st.caption(f"Latency: **{dl_time:.2f} ms** | Features: 128-dim dense embedding")
            
            fig_dl = px.bar(
                x=[p * 100 for p in dl_probs],
                y=dl_cfg["classes"],
                orientation="h",
                title="PyTorch DL Probabilities"
            )
            fig_dl.update_layout(xaxis=dict(range=[0, 100], title="Probability (%)"), yaxis=dict(title=""), height=300)
            st.plotly_chart(fig_dl, use_container_width=True)
            
        with comp2:
            st.markdown(f"#### ⚡ Calibrated Linear (TF-IDF)")
            st.metric("Predicted Class", f"{CAT_ICONS.get(ml_top, '📰')} {ml_top}", f"{np.max(ml_probs)*100:.2f}%")
            st.caption(f"Latency: **{ml_time:.2f} ms** | Features: 15,000 sublinear TF-IDF")
            
            fig_ml = px.bar(
                x=[p * 100 for p in ml_probs],
                y=[c.capitalize() for c in label_enc.classes_],
                orientation="h",
                title="Calibrated Linear Probabilities",
                color_discrete_sequence=["#10b981"]
            )
            fig_ml.update_layout(xaxis=dict(range=[0, 100], title="Probability (%)"), yaxis=dict(title=""), height=300)
            st.plotly_chart(fig_ml, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Training & Evaluation Analytics
# ---------------------------------------------------------
with tab4:
    st.markdown("### 📊 Comprehensive Model Benchmarks & Metrics")
    
    if metrics_data:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Dataset Sampled", f"{metrics_data['dataset_total']:,}")
        m_col2.metric("ML Test Accuracy", f"{metrics_data['ml_model']['accuracy']*100:.2f}%")
        m_col3.metric("PyTorch DL Accuracy", f"{metrics_data['dl_model']['accuracy']*100:.2f}%")
        m_col4.metric("Active Classes", len(metrics_data['classes']))
        
        st.markdown("---")
        
        # Confusion Matrices
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 🧠 PyTorch TextCNN: Confusion Matrix")
            cm_dl = np.array(metrics_data["dl_model"]["confusion_matrix"])
            fig_cm_dl = px.imshow(
                cm_dl,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=metrics_data["classes"],
                y=metrics_data["classes"],
                color_continuous_scale="Blues",
                text_auto=True
            )
            st.plotly_chart(fig_cm_dl, use_container_width=True)
            
        with c_right:
            st.markdown("#### ⚡ Calibrated Linear: Confusion Matrix")
            cm_ml = np.array(metrics_data["ml_model"]["confusion_matrix"])
            fig_cm_ml = px.imshow(
                cm_ml,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=metrics_data["classes"],
                y=metrics_data["classes"],
                color_continuous_scale="Teal",
                text_auto=True
            )
            st.plotly_chart(fig_cm_ml, use_container_width=True)
            
        # Per-Class Precision & Recall Breakdown
        st.markdown("#### 📋 Detailed Per-Class Classification Report (PyTorch DL)")
        report_data = []
        for cls_name in metrics_data["classes"]:
            if cls_name in metrics_data["dl_model"]["report"]:
                rep = metrics_data["dl_model"]["report"][cls_name]
                report_data.append({
                    "Category": cls_name,
                    "Precision": round(rep["precision"], 4),
                    "Recall": round(rep["recall"], 4),
                    "F1-Score": round(rep["f1-score"], 4),
                    "Test Support": rep["support"]
                })
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    else:
        st.info("Metrics file `model_metrics.json` not found. Please train models using `train_models.py`.")

# ---------------------------------------------------------
# TAB 5: Model Architecture & Methodology
# ---------------------------------------------------------
with tab5:
    st.markdown("### 🔬 Architecture & Deep Learning Methodology")
    
    st.markdown("""
    #### 1. The Deep Learning Pipeline: PyTorch TextCNN
    Traditional Bag-of-Words and TF-IDF models ignore word order, syntactic hierarchy, and multi-word phrases. 
    Our **PyTorch TextCNN** employs 1D Convolutional Neural Networks across learned word embeddings:
    
    1. **Embedding Layer (128-dim):** Transforms incoming word indices into dense continuous semantic vectors.
    2. **Multi-Scale Convolutional Banks:**
       - **Filter Size 2:** Captures local bi-gram representations (e.g., *"prime minister"*, *"interest rate"*).
       - **Filter Size 3:** Captures tri-grams (e.g., *"supreme court ruling"*, *"all-time high"*).
       - **Filter Size 4:** Captures 4-gram sentence clauses.
    3. **Global Max-Pooling:** Extracts the single most dominant feature from each kernel map, regardless of sentence length.
    4. **Dropout Regularization (0.40):** Prevents co-adaptation of hidden features.
    5. **Dense Linear Projection & Softmax:** Projects concatenated pooled feature vectors into calibrated class probabilities.
    
    ---
    
    #### 2. Why Accuracy Rose from 91.7% to 99.7%
    - **Headline + Content Fusion:** Inshorts headlines pack the most information-dense keywords. Merging them eliminated semantic ambiguity.
    - **Data Sanitization:** Fixed the stringified Python list labels (e.g. converting `['politics', 'national']` strings into structured atomic classes).
    - **Sublinear TF Scaling & Bi-grams:** Allowed the linear model to prevent repetitive words from swamping contextual bigrams.
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    Decodelabs Data Science Internship | Week 3 Project | Developed by @Prpandav<br>
    Powered by PyTorch Deep Learning & Streamlit
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
    Developed by @Prpandav<br>
    Powered by PyTorch Deep Learning & Streamlit
</div>
""", unsafe_allow_html=True)