import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import time
import pytesseract
from PIL import Image
import speech_recognition as sr
import os
import hashlib
import base64
import json

# Helper Functions
def get_pdf_text(pdf_docs):
    """Extract text from the uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for better processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks using sentence-transformers embeddings."""
    # Load a SentenceTransformer model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight and effective
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Create the FAISS vectorstore
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Set up the conversational chain with a retry mechanism."""
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.3, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    attempts = 3

    for attempt in range(attempts):
        try:
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            return conversation_chain
        except Exception as e:
            st.error(f"Attempt {attempt + 1}: Failed to create conversation chain: {e}")
            if attempt < attempts - 1:
                time.sleep(5)  # Retry after a delay
    return None  # Return None if all attempts fail

def get_pdf_metadata(pdf_docs):
    """Extract metadata from the uploaded PDFs."""
    metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        meta = pdf_reader.metadata
        metadata.append({
            'Author': meta.get('/Author', 'Unknown'),
            'CreationDate': meta.get('/CreationDate', 'Unknown'),
            'Title': meta.get('/Title', 'Unknown')
        })
    return metadata

# New Features
def summarize_text(text, model="t5-small"):
    """Summarize text using a model."""
    summarizer = HuggingFaceHub(repo_id=model)
    summary = summarizer(text)
    return summary

def download_chat_history(chat_history):
    """Download chat history as a text file."""
    history_text = []
    for msg in chat_history:
        role = getattr(msg, 'role', 'Unknown')
        content = getattr(msg, 'content', 'No content available')
        history_text.append(f"{role}: {content}")
    
    history_text = "\n".join(history_text)
    b64 = base64.b64encode(history_text.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="chat_history.txt">Download Chat History</a>'
    st.markdown(href, unsafe_allow_html=True)

# Main Function
def main():
    load_dotenv()
    st.set_page_config(page_title="Enhanced PDF Chat", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")
    st.subheader("1. Upload PDFs in the sidebar")
    st.subheader("2. Process and chat with the content")

    # Sidebar
    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        
        # Display metadata
        if pdf_docs:
            metadata = get_pdf_metadata(pdf_docs)
            st.write("**PDF Metadata**")
            st.write(metadata)

        process_pages = st.text_input("Specify pages to process (e.g., 1,2,5-10)")
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                try:
                    if process_pages:
                        pages = [int(p) - 1 for p in process_pages.split(",")]
                        raw_text = process_selected_pages(pdf_docs[0], pages)  # Process only the first PDF for now
                    else:
                        raw_text = get_pdf_text(pdf_docs)
                    
                    # Summarize text
                    st.write("**Summarizing Document...**")
                    summary = summarize_text(raw_text)
                    st.write(summary)

                    # Prepare conversation
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                    if st.session_state.conversation:
                        st.success("Conversation chain successfully initialized!")
                    else:
                        st.error("Failed to initialize the conversation chain. Please try again.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

    # Chat Interface
    user_question = st.text_input("Ask a question:")
    if user_question:
        if st.session_state.conversation:
            try:
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history = response['chat_history']
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    else:
                        st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in conversation: {e}")
        else:
            st.error("Please upload and process a document to start the chat.")

    # Download chat history
    if st.session_state.chat_history:
        st.subheader("Download Chat History")
        download_chat_history(st.session_state.chat_history)

if __name__ == '__main__':
    main()
