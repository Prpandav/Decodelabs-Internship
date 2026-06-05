# Decodelabs Data Science Internship: Week 3 Project
**NLP News Text Classification & Streamlit Web App**

This repository contains my Week 3 project for the Decodelabs Data Science Remote Internship. The objective of this project is to perform end-to-end Natural Language Processing (NLP)—from text cleaning and tokenization to vectorization, training classification models, and deploying an interactive web application.

The analysis is based on the **Inshorts News Dataset**, focusing on categorizing news headlines and snippets into distinct topics (e.g., Politics, Sports, Technology).

---

## 🚀 Executive Summary of Tasks

### Task 1 & 2: Data Collection, Cleaning & Preprocessing
To ensure high-quality model inputs, I performed the following text sanitization steps:
*   **Memory Optimization:** Isolated the `Content` and `News Categories` columns to manage large dataset sizes.
*   **Regex Text Cleaning:** Developed a custom `clean_text()` function utilizing regular expressions (`re`) to lowercase all text and remove numbers, punctuation, and special characters.
*   **Target Consolidation:** Filtered the dataset to retain only the **Top 10 Most Frequent Categories** (down from 1,300+ noisy categories) to provide sufficient class representation for the classification models.

### Task 3: NLP Exploratory Data Analysis (EDA)
*   **Class Distribution:** Visualized the article count for the top 10 categories using horizontal bar charts.
*   **Vocabulary Visualization:** Generated a rich **Word Cloud** from a sample of 5,000 cleaned articles to visualize the most frequent vocabulary.
*   **N-Gram Analysis:** Leveraged `CountVectorizer` to extract and visualize the Top 15 Most Frequent **Bigrams** (2-word phrases), removing English stop words to reveal deep contextual subjects (e.g., "supreme court", "prime minister").

### Task 4: Feature Engineering & Vectorization
*   **Label Encoding:** Converted categorical text labels into numerical targets using `LabelEncoder`.
*   **Data Splitting:** Divided the data into an 80% training and 20% testing set using `train_test_split`.
*   **TF-IDF Vectorization:** Initialized a `TfidfVectorizer(max_features=5000)` to transform the text corpus into a sparse numerical matrix. Strictly applied `.fit_transform()` on the training set and `.transform()` on the testing set to prevent data leakage.

### Task 5: Predictive Modeling & Evaluation
Trained and compared two distinct machine learning classification models:
*   **Multinomial Naive Bayes:** Served as a fast and reliable baseline model.
*   **Logistic Regression:** Achieved superior accuracy and F1-Scores by better handling complex relationships between vocabulary words in the TF-IDF matrix.
*   **Evaluation:** Generated detailed Classification Reports and built a **Confusion Matrix Heatmap** to visually diagnose category misclassifications.
*   **Pipeline Export:** Saved the winning Logistic Regression model, TF-IDF Vectorizer, and Label Encoder using `joblib` for external deployment.

### Task 6: Streamlit Web App Deployment
*   Built an interactive **AI News Classifier** web application using `Streamlit`.
*   Integrated the saved `joblib` components to allow users to paste custom news snippets and receive real-time category predictions from the trained model.

---

## ⚙️ How to Run

### Dependencies
To execute the Jupyter Notebook and Streamlit App, ensure your environment has the following Python libraries installed:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `wordcloud`
*   `streamlit`
*   `joblib`

### Execution (Web App)
1. Clone this repository to your local machine.
2. Open your terminal and navigate to the directory containing `app.py` and the exported `.pkl` files.
3. Run the following command:
   ```bash
   streamlit run app.py
   ```
4. Access the web interface via the `localhost` URL provided in the terminal to test live predictions.

### Execution (Notebook)
To view the entire EDA and training pipeline, launch Jupyter Notebook and run the main `.ipynb` file sequentially.

---
*Developed by a Data Science Intern at Decodelabs - Week 3 Project*