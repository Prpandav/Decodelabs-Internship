# Project Report: AI News Document Classifier

**Internship Program:** Decodelabs Data Science Remote Internship  
**Project Title:** Week 3 Project — End-to-End NLP & Deep Learning Document Classification  
**Student Name:** Prpandav  
**Domain:** Natural Language Processing (NLP) & Deep Learning (DL)  
**Deployment:** Streamlit Web Application  

---

## 1. Executive Summary
The goal of this project is to build an automated AI system that can read any news article or headline and accurately classify it into its correct news topic (such as Business, Technology, Sports, or Politics). 

By combining traditional Natural Language Processing (NLP) techniques with modern Deep Learning (PyTorch TextCNN), the final system achieves a **99.69% accuracy** on unseen test data. The system is packaged into an interactive web application built with Streamlit, supporting both real-time individual predictions and bulk batch document uploads.

---

## 2. Problem Statement
Every day, thousands of news articles are published across the internet. Manually reading, tagging, and organizing these articles is slow, expensive, and prone to human error. 

An automated text classification system solves this by:
* Automatically routing news stories to the right category in milliseconds.
* Filtering information feeds for readers and businesses.
* Handling high volumes of articles without any manual intervention.

---

## 3. Dataset Overview
* **Source:** Inshorts News Dataset (over 300,000 articles).
* **Main Attributes:**
  * `Headline`: Short, punchy summary of the news story.
  * `Content`: The 60-word body of the news article.
  * `News Categories`: The ground-truth topic label.
  * `Date`: Date of publication.
* **Target Classes (10 Clean Categories):**  
  Automobile, Business, Education, Entertainment, National, Politics, Science, Sports, Technology, and World.

---

## 4. Key Engineering Challenges & Solutions

### Challenge 1: Dirty Category Labels
* **The Problem:** The raw dataset stored categories as text representations of Python lists (e.g., `"['politics', 'national']"`). A standard text read treated these as messy compound strings with brackets.
* **The Solution:** We used Python's `ast.literal_eval` to clean and extract distinct, atomic topic categories.

### Challenge 2: Discarding the Headline Column
* **The Problem:** Many basic NLP approaches only look at the article body and discard the headline. However, headlines contain the most keyword-dense information.
* **The Solution (Feature Fusion):** We merged the headline and article body together (`Headline + " - " + Content`). This single step provided the models with rich contextual signals and eliminated topic confusion.

### Challenge 3: Preprocessing Consistency
* **The Problem:** If text is cleaned one way during model training and another way during web app inference, accuracy drops significantly in real-world use.
* **The Solution:** We created a shared helper module (`preprocessor.py`) that applies the exact same text cleaning and tokenization rules in both the training notebook and the Streamlit app.

---

## 5. Model Architecture & Methodology

We trained and compared two different model architectures:

### Model A: Fast Machine Learning Baseline (Calibrated Linear Classifier)
1. **Text Representation:** Sublinear TF-IDF (Term Frequency-Inverse Document Frequency) using both single words (unigrams) and two-word phrases (bigrams) across 15,000 features.
2. **Classifier:** Linear Support Vector Classifier (`LinearSVC`) wrapped with Platt Scaling (`CalibratedClassifierCV`) so the model outputs reliable probability percentages.
3. **Strength:** Ultra-fast inference (< 3 milliseconds per article).

### Model B: Deep Learning Architecture (PyTorch TextCNN)
1. **Embedding Layer (128 Dimensions):** Converts incoming words into dense semantic vectors where similar words sit close to each other in mathematical space.
2. **Multi-Scale 1D Convolutional Filter Banks:** 
   * **Filter size 2:** Captures 2-word phrases (e.g., *"interest rate"*).
   * **Filter size 3:** Captures 3-word phrases (e.g., *"supreme court ruling"*).
   * **Filter size 4:** Captures 4-word clauses.
3. **Global Max-Pooling:** Extracts the strongest detected feature from each filter regardless of sentence length.
4. **Dropout Regularization (40%):** Prevents overfitting.
5. **Softmax Output Head:** Produces confidence probabilities across all 10 news classes.

---

## 6. Evaluation & Benchmark Results

The models were evaluated on an unseen 20% test split (13,215 articles) with balanced category representation:

| Model Architecture | Model Family | Test Accuracy | Precision | Recall | F1-Score | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | Baseline ML | 97.42% | 0.97 | 0.97 | 0.97 | ~2 ms |
| **Logistic Regression** | Linear ML | 99.08% | 0.99 | 0.99 | 0.99 | ~3 ms |
| **Calibrated Linear Classifier** | Advanced ML | **99.71%** | **1.00** | **1.00** | **1.00** | **< 3 ms** |
| **PyTorch TextCNN** | Deep Learning | **99.69%** | **1.00** | **1.00** | **1.00** | **~8 ms** |

---

## 7. Streamlit Web Application Features
The web application (`app.py`) provides a complete dashboard with 5 dedicated tabs:
1. **Real-Time Classifier:** Live classification with one-click test presets (e.g., ISRO space launch, stock market crash, sports victory). Shows confidence score, latency counter, and an interactive Plotly horizontal probability bar chart.
2. **Batch Document Upload:** Allows users to upload a CSV file with hundreds of news articles, view a live progress bar, see a summary pie chart, and download the categorized CSV.
3. **Side-by-Side Model Comparison:** Runs both the Deep Learning TextCNN and the Linear Classifier on the same input simultaneously to compare probability spreads and speed.
4. **Training Analytics & Confusion Matrix:** Displays the interactive Confusion Matrix heatmap and per-class precision/recall metrics.
5. **Architecture & Methodology:** Provides an educational visual guide explaining how 1D convolutions extract meaning from text.

---

## 8. Conclusion & Takeaways
* Cleaning the raw labels and combining `Headline + Content` solved the primary accuracy bottleneck of the initial project.
* The PyTorch Deep Learning model successfully learns semantic phrase patterns, achieving **99.69% accuracy**.
* The calibrated linear model provides a complementary, ultra-fast option (< 3 ms) for high-throughput production environments.
* The interactive Streamlit application bridges the gap between machine learning research and real-world user interaction.
