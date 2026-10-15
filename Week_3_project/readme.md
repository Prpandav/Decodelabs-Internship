# NewsPulse AI: Deep Learning & NLP Document Classification
**Decodelabs Data Science Internship — Week 3 Project**  
*Developed by @Prpandav*

![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🌟 Executive Overview
This repository contains the complete Week 3 project for the **Decodelabs Data Science Remote Internship**. The project focuses on building an end-to-end, high-accuracy Natural Language Processing (NLP) and Deep Learning document classification system trained on the **Inshorts News Dataset** (~307,000 articles).

It features a dual-model architecture:
1. **Deep Learning Model (PyTorch TextCNN):** 1D Multi-Scale Convolutional Neural Network with learned 128-dimensional word embeddings (**99.69% Test Accuracy**).
2. **Machine Learning Model (Calibrated Linear Classifier):** Sublinear TF-IDF bi-gram representation with calibrated probabilities for ultra-fast, sub-5ms inference (**99.71% Test Accuracy**).
3. **Interactive Streamlit Web Application:** A 5-tab production dashboard featuring real-time single article prediction, preset demos, batch CSV classification, side-by-side model comparison, and training analytics.

---

## 📊 Benchmark Results

| Model Architecture | Feature Representation | Test Accuracy | Macro F1-Score | Avg CPU Latency |
| :--- | :--- | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | TF-IDF (15k Features) | 97.42% | 0.97 | ~2 ms |
| **Logistic Regression** | TF-IDF (15k Features) | 99.08% | 0.99 | ~3 ms |
| **Calibrated Linear Classifier** | Sublinear TF-IDF (1, 2 n-grams) | **99.71%** | **1.00** | **< 3 ms** |
| **PyTorch TextCNN (Deep Learning)** | Learned Dense Embeddings (128-d) | **99.69%** | **1.00** | **~8 ms** |

---

## 🚀 Key Improvements & Innovations

### 1. Data Sanitization & Category Normalization
- **Fixed Stringified List Labels:** Inshorts stores categories as string representations of lists (e.g. `"['politics', 'national']"`). The pipeline uses `ast.literal_eval` to clean and extract distinct, atomic categories.
- **Top 10 High-Signal Classes:** Retained 10 distinct, non-overlapping categories: `Business`, `World`, `Entertainment`, `Politics`, `Automobile`, `National`, `Science`, `Sports`, `Education`, and `Technology`.

### 2. Feature Fusion (`Headline + " - " + Content`)
- Headlines contain the most information-dense, summary keywords. Merging the headline with article content resolved topic ambiguity and provided an immediate leap in classification accuracy.

### 3. Deep Learning Architecture (PyTorch TextCNN)
- **Embedding Layer:** 128-dimensional dense word vector space across a 25,000-token vocabulary.
- **Multi-Filter Convolutions:** Parallel 1D convolutional banks with kernel sizes `[2, 3, 4]` capturing bi-grams, tri-grams, and 4-gram phrases.
- **Global Max-Pooling & Dropout:** Extracts the most salient phrase features regardless of document length, regularized with 40% dropout.

### 4. Production Streamlit Dashboard (`app.py`)
- **Real-Time Classification:** Real-time prediction with confidence levels and latency metrics.
- **Quick-Fill Demo Presets:** 1-click test buttons for Sports, Business, Science, Tech, Politics, and Automotive.
- **Interactive Plotly Visualizations:** Dynamic horizontal bar charts showing the probability breakdown across all 10 categories.
- **Salient Keyword Attribution:** Highlights detected contextual keywords that influenced the prediction.
- **Batch CSV Processing:** Upload a CSV file, classify hundreds of articles with a live progress bar, and export the enriched CSV.
- **Live Model Comparison:** Run both models side-by-side on identical text to contrast confidence distributions and speeds.
- **Training Analytics Dashboard:** Interactive Confusion Matrices and per-class precision/recall/F1 metrics.

---

## 📁 Repository Structure

```
Week_3_project/
│
├── Code/
│   ├── app.py                         # Production Streamlit Web Application
│   ├── preprocessor.py                # Shared text cleaning & PyTorch TextCNN definitions
│   ├── train_models.py                # End-to-end model training & export script
│   ├── Document_Classification.ipynb  # Interactive Jupyter Notebook with complete EDA & DL
│   ├── lr_news_model.pkl              # Exported Calibrated Linear Classifier
│   ├── tfidf_vectorizer.pkl           # Exported TF-IDF Vectorizer
│   ├── label_encoder.pkl              # Exported LabelEncoder mapping
│   ├── textcnn_model.pt               # Exported PyTorch TextCNN weights
│   ├── vocab.json                     # Vocabulary dictionary & model configurations
│   ├── model_metrics.json             # Benchmarks, confusion matrices & training history
│   └── requirements.txt               # Application dependencies
│
├── Data/
│   └── inshort_news_dataset.csv       # Inshorts News Dataset (300k+ articles)
│
└── readme.md                          # Project documentation
```

---

## ⚙️ How to Run

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. From the `Code/` directory:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
cd Code
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. Run the Jupyter Notebook
Launch Jupyter Notebook or VS Code to step through `Document_Classification.ipynb` to view the exploratory data analysis, loss curves, and evaluation heatmaps.

### 4. Retrain Models (Optional)
To retrain both models from scratch and refresh all exported artifacts:
```bash
python train_models.py
```

---

*Developed by a Data Science Intern at Decodelabs — Week 3 Project*

*Developed by a Data Science Intern at Decodelabs — Week 3 Project*