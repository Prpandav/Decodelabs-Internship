# Decodelabs Data Science Internship: Week 4 Capstone Project
**AI Document Analyst (Retrieval-Augmented Generation System)**

This repository contains my final Week 4 capstone project for the Decodelabs Data Science Remote Internship. The objective of this project was to move beyond traditional machine learning and architect a modern Generative AI application using a **Retrieval-Augmented Generation (RAG)** pipeline.

The resulting application allows users to upload complex, multi-page PDF documents and query them using natural language. The AI acts as a smart assistant that reads the document and answers questions based *strictly* on the provided text, effectively eliminating AI hallucinations.

---

##  Executive Summary of Tasks

### Task 1: Document Ingestion & Text Extraction
To allow the AI to read local files, I engineered a robust document parsing backend:
*   **PDF Parsing:** Utilized `PyPDF2` to iterate through uploaded PDF files and extract raw text from every page.
*   **Memory Management:** Processed the extracted text in memory via Streamlit's file uploader, ensuring no temporary files cluttered the local hard drive.

### Task 2: Contextual Text Chunking
Large Language Models (LLMs) have strict context limits. To solve this, I implemented semantic chunking:
*   **Chunking Strategy:** Leveraged LangChain's `RecursiveCharacterTextSplitter` to divide the massive raw text string into smaller, overlapping chunks of 1000 characters.
*   **Context Preservation:** Set a `chunk_overlap` of 200 characters to ensure that sentences or concepts cut in half at the end of a chunk carried over to the next, preventing the AI from losing the narrative context.

### Task 3: Local Embeddings & Vector Storage
To make the text searchable by the AI, I built a local vectorization pipeline:
*   **Open-Source Embeddings:** Bypassed expensive cloud APIs by integrating HuggingFace's local `all-MiniLM-L6-v2` embedding model to mathematically map the semantic meaning of the text chunks into high-dimensional space.
*   **Vector Database:** Engineered a local high-speed vector database using **FAISS** (Facebook AI Similarity Search) to index the embedded chunks, allowing for millisecond-speed similarity lookups when a user asks a question.

### Task 4: The Generative AI Engine
I orchestrated the retrieval and generation logic using modern frameworks:
*   **LLM Integration:** Connected the **Google Gemini Flash** API via LangChain to act as the cognitive engine for the application.
*   **Zero-Hallucination Prompting:** Designed a strict `PromptTemplate` that forces the AI to answer only using the retrieved FAISS context, explicitly instructing it to state *"The answer is not available in the document"* if the information is missing.
*   **LCEL Pipeline:** Replaced deprecated LangChain chains with the modern **LangChain Expression Language (LCEL)** (`prompt | model | StrOutputParser()`) for cleaner, faster, and more professional code execution.

### Task 5: Streamlit UI Deployment
*   Built a highly interactive, ChatGPT-style user interface using `Streamlit`.
*   Utilized Streamlit's session state (`st.session_state`) and chat elements (`st.chat_message`, `st.chat_input`) to maintain conversation history and provide dynamic loading states (`st.spinner`) while the backend processes the vectors.

---

##  How to Run

### Dependencies
Ensure your environment has the following Python libraries installed:
*   `streamlit`
*   `langchain`
*   `langchain-google-genai`
*   `langchain-text-splitters`
*   `langchain-community`
*   `faiss-cpu`
*   `PyPDF2`
*   `sentence-transformers`
*   `python-dotenv`

*(You can install all dependencies quickly by running `pip install -r requirements.txt`)*

### Environment Setup
This application requires a Google Gemini API Key.
1. Create a file named `.env` in the root of the `Week_4_Project` directory.
2. Add your API key in the following format:
   ```dotenv
   GOOGLE_API_KEY="AIzaSyYourKeyHere..."
   ```
*(Note: Ensure `.env` is added to your `.gitignore` to prevent leaking your API key to GitHub).*

### Execution
1. Clone this repository to your local machine.
2. Open your terminal and navigate to the `Week_4_Project` directory.
3. Run the following command to boot the Streamlit server:
   ```bash
   streamlit run app.py
   ```
4. Access the web interface via the `localhost` URL provided in the terminal.
5. Upload a PDF using the sidebar, click **Process Documents**, and start asking questions!

---
*Developed by a Data Science Intern at Decodelabs - Week 4 Capstone Project*