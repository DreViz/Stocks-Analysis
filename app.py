import os
import streamlit as st
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Streamlit app
st.title("Stocks Research Tool")
st.sidebar.title("Enter Your URLs Here")

# Initialize session state
if "URLS_INPUT" not in st.session_state:
    st.session_state.URLS_INPUT = []
if "vectorindex_google" not in st.session_state:
    st.session_state.vectorindex_google = None

main_placeholder = st.empty()

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Sidebar input for URLs
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_input_{i + 1}")
    if len(url) > 0:
        st.session_state.URLS_INPUT.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=st.session_state.URLS_INPUT)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectorindex_google = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorindex_google.save_local("faiss_index")

    main_placeholder.success("Process Complete. Now you can ask questions...✅✅✅")

# Query section
if st.session_state.vectorindex_google:
    query = st.text_input('Type your questions')
    if query:
        vectorindex_google = FAISS.load_local(
            "faiss_index", 
            GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_google.as_retriever())
        response = chain.invoke({"question": query})  # Updated to use invoke
        st.write("Answer --- ", response['answer'])
        st.write("Source --- ", response['sources'])
