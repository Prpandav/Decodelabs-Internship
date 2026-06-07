# Decodelabs Data Science Internship Portfolio

**Author:** Prasiddh Pandav

Welcome to my data science internship portfolio! This central repository tracks the weekly projects and machine learning pipelines I am building during my 4-week remote Data Science Internship at Decodelabs.

## Repository Structure

This repository is organized by week. Each folder is a self-contained project with its own detailed `README.md`, Jupyter Notebook, and datasets.

### [Week 1: Anime Data Analysis & Recommendation Engine](./Week_1_Project)
* **Objective:** Foundational data ingestion, preprocessing, Exploratory Data Analysis (EDA), and insight generation.
* **Dataset:** CooperUnion Anime Recommendations Database (~12,000 entries).
* **Key Achievement:** Built an Unsupervised Content-Based Recommendation Engine from scratch using `scikit-learn`, utilizing `CountVectorizer` and Cosine Similarity to find mathematically related shows based on format and genre tags.

### [Week 2: AI & Data Science Job Salary Predictor](./Week_2_Project)
* **Objective:** Feature engineering, advanced data visualization, and Supervised Machine Learning.
* **Dataset:** Data Science Job Salaries (Global hiring trends & metrics).
* **Key Achievement:** Developed an end-to-end Supervised Regression pipeline. Engineered categorical features and trained a `RandomForestRegressor` to predict accurate candidate salary estimates based on experience level, remote ratios, and company size.

### [Week 3: NLP News Text Classification & Web App](./Week_3_Project)
* **Objective:** Text preprocessing, NLP feature engineering, model comparison, and deployment.
* **Dataset:** Inshorts News Dataset (Text content & Categories).
* **Key Achievement:** Processed over 300k rows of text data using regex, built a TF-IDF vectorization pipeline, and trained classification models (Naive Bayes & Logistic Regression). Deployed the winning Logistic Regression model as an interactive `Streamlit` web application.
* **Live App:** [AI News Classifier](https://ai-news-classifier.streamlit.app/)

## Technology Stack
* **Language:** Python
* **Data Manipulation & Cleaning:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / VS Code

# [Week 4 Capstone: AI Document Analyst (RAG System)](./Week_4_Project)

## Project Overview
For the final capstone of the Decodelabs Data Science Internship, I architected a complete **Retrieval-Augmented Generation (RAG)** pipeline. This application allows users to upload complex PDF documents and query them using natural language. The system retrieves the most semantically relevant context from the document and synthesizes highly accurate, hallucination-free answers.

## Architectural Design
This project implements a decoupled, hybrid AI architecture to maximize speed, privacy, and accuracy:
1. **Local Vectorization:** Bypassed cloud API dependencies by utilizing a local open-source HuggingFace embedding model (`all-MiniLM-L6-v2`).
2. **High-Speed Retrieval:** Engineered a local vector database using **FAISS** (Facebook AI Similarity Search) to store and query multidimensional text embeddings.
3. **Generative Inference:** Orchestrated the **Gemini 3.5 Flash** LLM via LangChain to generate responses strictly grounded in the retrieved document chunks.
4. **Interactive UI:** Deployed a persistent conversational interface using **Streamlit**, featuring session-state memory and real-time document processing feedback.

## Tech Stack
* **Orchestration Framework:** LangChain (`langchain`, `langchain-google-genai`)
* **Vector Database:** FAISS (`faiss-cpu`)
* **Embedding Model:** Sentence-Transformers (HuggingFace)
* **Large Language Model (LLM):** Google Gemini 3.5 Flash
* **Document Processing:** PyPDF2, RecursiveCharacterTextSplitter
* **Frontend UI:** Streamlit
* **Live App:** [AI Document Analyst](https://ai-doc-analyst.streamlit.app/)

## Key Learnings & Engineering Wins
* Successfully diagnosed and resolved cloud API versioning/deprecation conflicts by pivoting to a decoupled, local-embedding architecture.
* Mastered semantic chunking strategies (handling character limits and overlap) to ensure the LLM never loses narrative context.
* Designed a secure environment management system (`python-dotenv`) to safely handle API authentication.

## How to Navigate
To explore a specific project, open the corresponding folder above. Inside, you will find a dedicated `README.md` that explains the exact methodology, the insights discovered, and instructions on how to run the code locally.
