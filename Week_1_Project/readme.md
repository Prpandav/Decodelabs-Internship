# Decodelabs Data Science Internship: Week 1 Project
**Anime Data Analysis & Content-Based Recommendation Engine**

This repository contains my Week 1 project for the Decodelabs Data Science Remote Internship. The objective of this project is to perform end-to-end data processing—from initial data ingestion and cleaning to exploratory data analysis (EDA), data visualization, and finally building an unsupervised machine learning recommendation engine.

The analysis is based on the **CooperUnion Anime Recommendations Database** (sourced from Kaggle), containing approximately 12,000 anime records.

---

## 🚀 Executive Summary of Tasks

### Task 1: Data Collection & Dataset Understanding
*   **Data Ingestion:** Loaded the raw `anime.csv` file using `pandas` via relative file paths to ensure environment portability.
*   **Initial Inspection:** Utilized `df.shape` and `df.info()` to understand the dataset's dimensionality, feature data types, and identify early instances of missing or malformed data.

### Task 2: Data Cleaning & Preprocessing
To ensure high-quality model inputs and accurate analysis, I performed the following data sanitization steps:
*   **Feature Formatting:** Identified that the `episodes` feature was improperly typed as an `object` (string) due to the presence of `"Unknown"` values. Converted these string values to `NaN`, dropped the respective rows, and safely cast the column to a strict `integer` type.
*   **Handling Missing Values:** Dropped rows containing null values in the `genre` and `rating` columns, as imputing these categorical/target variables could skew the recommendation engine's accuracy.
*   **Deduplication:** Applied `drop_duplicates()` to ensure dataset integrity.

### Task 3: Exploratory Data Analysis (EDA)
*   **Statistical Summaries:** Executed `df.describe()` to capture the central tendencies and spread of the numerical features.
*   **Top Performers:** Extracted the **Top 10 Highest-Rated Anime**, applying a community filter (`members > 10000`) to eliminate extreme statistical outliers caused by low vote counts.
*   **Categorical Analysis:** Analyzed the distribution count of various anime formats (e.g., TV, Movie, OVA, Special) and calculated the average rating grouped by each format type.

### Task 4: Data Visualization
Leveraging `matplotlib` and `seaborn`, I generated presentation-ready visualizations to highlight key EDA insights:
1.  **Horizontal Bar Chart:** Showcased the Top 10 Highest-Rated Anime (filtered for >10k members), utilizing a horizontal layout for readable title text.
2.  **Ordered Count Plot:** Displayed the volume distribution of anime formats, ordered dynamically from most frequent to least frequent.
3.  **Bar Chart:** Visualized the average community rating by anime format type, standardizing the Y-axis to a 0-10 scale for proper contextual representation.

### Task 5: Predictive Model / Insight Project
Designed and built a **Content-Based Recommendation Engine** from scratch to recommend similar anime titles based on structural feature matching.
*   **Feature Engineering:** Concatenated the `genre` and `type` columns into a unified `tags` feature to capture contextual metadata.
*   **Vectorization:** Utilized `scikit-learn`'s `CountVectorizer(stop_words='english')` to transform the text tags into a mathematical vector matrix.
*   **Similarity Computation:** Applied `cosine_similarity` to calculate the multi-dimensional distance between all anime records. *(Note: As this is an unsupervised distance matching algorithm, a train-test split was neither required nor applicable).*
*   **Inference Function:** Developed a scalable Python function, `recommend_anime(anime_name)`, which locates a target anime in the matrix and returns the top 5 closest matching titles and their associated genres (successfully tested with popular entries like *'One Piece'*).

---

## ⚙️ How to Run

### Dependencies
To execute the Jupyter Notebook, ensure your environment has the following Python libraries installed:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`

### Execution
1. Clone this repository to your local machine.
2. Ensure the `anime.csv` dataset is properly placed in the `data/` directory.
3. Launch Jupyter Notebook or Jupyter Lab (e.g., via VS Code).
4. Open `Code/Week1_Anime_Analysis.ipynb` and Run All cells sequentially.

---
*Developed by a Data Science Intern at Decodelabs - Week 1 Project*