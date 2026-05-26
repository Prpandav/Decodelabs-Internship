# Decodelabs Data Science Internship: Week 2 Project
**Data Science Salary Analysis & Prediction Model**

This repository contains my Week 2 project for the Decodelabs Data Science Remote Internship. The objective of this project is to perform end-to-end data processing from initial data ingestion and cleaning to exploratory data analysis (EDA), data visualization, and finally building a supervised machine learning regression model.

The analysis is based on a **Data Science Salaries Database** (sourced via `kagglehub`), focusing on predicting salaries in USD based on various employment factors.

---

## 🚀 Executive Summary of Tasks

### Task 1: Data Collection & Dataset Understanding
*   **Data Ingestion:** Loaded the raw dataset using `kagglehub`.
*   **Initial Inspection:** Utilized `df.shape` and `df.info()` to understand the dataset's dimensionality and feature data types.

### Task 2: Data Cleaning & Preprocessing
To ensure high-quality model inputs and accurate analysis, I performed the following data sanitization steps:
*   **Index Cleaning:** Identified and dropped the hanging index column `Unnamed: 0`.
*   **Deduplication & Nulls:** Removed all duplicate rows using `drop_duplicates()` and dropped rows with missing values using `dropna()`.
*   **Target Isolation:** Dropped redundant `salary` and `salary_currency` columns to strictly isolate `salary_in_usd` as our primary target variable for the predictive model.

### Task 3: Exploratory Data Analysis (EDA)
*   **Statistical Summaries:** Executed `df.describe()` to capture the central tendencies of numerical features.
*   **Categorical Analysis:** Grouped the average `salary_in_usd` by `experience_level` and sorted the results from highest to lowest.
*   **Job Frequency:** Extracted the **Top 10 Most Frequent Job Titles** in the dataset.
*   **Work Arrangements:** Compared the average salary between fully remote roles (`remote_ratio == 100`) and fully in-person roles (`remote_ratio == 0`).

### Task 4: Data Visualization
Leveraging `matplotlib` and `seaborn`, I generated presentation-ready visualizations to highlight key EDA insights:
1.  **Distribution Plot:** Showcased the frequency distribution of `salary_in_usd` using a histogram with a KDE overlay.
2.  **Bar Chart:** Visualized the average salary by experience level.
3.  **Horizontal Count Plot:** Displayed the top 10 most frequent job titles dynamically.
4.  **Boxplot:** Compared the salary distribution and outliers between fully remote and fully in-person work arrangements.
5.  **Advanced Heatmap:** Created a dual-axis correlation heatmap plotting the average salary against `experience_level` and `company_size`.

### Task 5: Predictive Model
Designed and built a **Supervised Regression Model** to predict the `salary_in_usd` for new candidates based on employment features.
*   **Feature Engineering:** Selected key features (`experience_level`, `job_title`, `company_size`, `remote_ratio`) and applied one-hot encoding using `pd.get_dummies(drop_first=True)` to convert categorical text into numeric format.
*   **Data Splitting:** Divided the data into an 80% training and 20% testing set using `train_test_split`.
*   **Model Training:** Initialized and trained a `RandomForestRegressor(random_state=42)` on the training data.
*   **Evaluation:** Evaluated the model using Mean Absolute Error (MAE) and R-squared ($R^2$) Score metrics.
*   **Inference Function:** Developed a scalable Python function, `predict_user_salary()`, which interactively prompts users for candidate details, aligns the input data with the trained model's structure using `.reindex()`, and outputs a predicted salary.

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
2. Launch Jupyter Notebook or Jupyter Lab (e.g., via VS Code).
3. Open the main analysis notebook and Run All cells sequentially.

---
*Developed by a Data Science Intern at Decodelabs - Week 2 Project*
