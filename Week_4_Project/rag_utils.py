import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load the environment variables from the .env file securely
load_dotenv()

def extract_text_from_pdf(pdf_docs):
    """
    Iterates through the uploaded PDF documents and extracts all raw text.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Loop through every page in the PDF and extract text
        for page in pdf_reader.pages:
            # Some pages might return None if they are images, so we check first
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def get_text_chunks(text):
    """
    Splits a massive string of text into smaller, overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # The max number of characters in a chunk
        chunk_overlap=200,   # The overlap between chunks to maintain context
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Converts text chunks into embeddings and stores them in a local FAISS vector database.
    """
    if len(text_chunks) == 0:
        print("❌ ERROR: No text was extracted!")
        return
        
    print("--- DIAGNOSTIC: Initializing HuggingFace Embeddings... ---")
    # Use the local open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("--- DIAGNOSTIC: Building FAISS Index... ---")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    vector_store.save_local("faiss_index")
    print("✅ SUCCESS: faiss_index folder created successfully!\n")

def get_conversational_chain():
    """
    Defines the LLM model and the prompt template for answering questions.
    """
    # Define the strict instructions for the AI
    prompt_template = """
    You are a helpful AI document assistant. Answer the question as detailed as possible from the provided context. 
    If the answer is not contained in the provided context, just say, "The answer is not available in the document." 
    Do not guess or make up information.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    
    # A low temperature (0.3) ensures the AI gives factual, non-creative answers
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Use modern LangChain Expression Language (LCEL) instead of load_qa_chain
    chain = prompt | model | StrOutputParser()
    return chain

def get_answer(user_question):
    """
    Loads the vector database, searches for relevant chunks, and asks the LLM to generate an answer.
    """
    # Use the same HuggingFace model here!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = vector_store.similarity_search(user_question)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    chain = get_conversational_chain()
    response = chain.invoke({"context": context_text, "question": user_question})
    
    return response