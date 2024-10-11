import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
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
from transformers import T5Tokenizer, T5ForConditionalGeneration

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
        chunk_size=600,  # Adjusted chunk size for better processing
        chunk_overlap=100,  # Overlap to maintain context
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks using embeddings."""
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
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
            if attempt < attempts - 1:
                time.sleep(5)  # Retry after a delay
            else:
                st.error(f"Error creating conversation chain: {e}")
                return None

# Updated PDF metadata function using PyPDF2 >= 2.x
def get_pdf_metadata(pdf_docs):
    """Extract metadata from the uploaded PDFs."""
    metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        meta = pdf_reader.metadata  # Accessing metadata in PyPDF2 >= 2.x
        metadata.append({
            'Author': meta.get('/Author', 'Unknown'),
            'CreationDate': meta.get('/CreationDate', 'Unknown'),
            'Title': meta.get('/Title', 'Unknown')
        })
    return metadata

# Local summarizer using transformers
def load_local_summarizer(model_name="t5-small"):
    """Load T5 summarizer locally."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(text, model_name="t5-small"):
    """Summarize text using a local Hugging Face model."""
    tokenizer, model = load_local_summarizer(model_name)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Other Helper Functions
def extract_text_from_image(image_file):
    """Extract text from scanned PDF images using OCR."""
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def speech_to_text(audio_file):
    """Convert audio input to text using speech recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

def download_chat_history(chat_history):
    """Download chat history as a text file."""
    history_text = "\n".join([f"{msg.content}" for msg in chat_history])  # Adjusted to handle message content
    b64 = base64.b64encode(history_text.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="chat_history.txt">Download Chat History</a>'
    st.markdown(href, unsafe_allow_html=True)

def authenticate_user(username, password):
    """Simple user authentication with username and password."""
    stored_password = os.getenv(username)
    if stored_password:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return hashed_password == stored_password
    return False

def process_selected_pages(pdf, pages_to_process):
    """Process selected pages from the PDF document."""
    pdf_reader = PdfReader(pdf)
    selected_text = ""
    for page_num in pages_to_process:
        selected_text += pdf_reader.pages[page_num].extract_text()
    return selected_text

def main():
    load_dotenv()
    st.set_page_config(page_title="Enhanced PDF Chat", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # User Authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.subheader("Login to Access Features")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")
        return

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input("Ask a question (or use voice input):")

    # Speech-to-Text Input
    audio_input = st.file_uploader("Upload Audio for Speech-to-Text", type=["wav", "mp3"])
    if audio_input:
        user_question = speech_to_text(audio_input)
        st.success(f"Recognized Speech: {user_question}")

    if user_question:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        
        # Metadata Display
        if pdf_docs:
            metadata = get_pdf_metadata(pdf_docs)
            st.write("**PDF Metadata**")
            st.write(metadata)
        
        process_pages = st.text_input("Specify pages to process (e.g., 1,2,5-10)")
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                if process_pages:
                    pages = [int(p) for p in process_pages.split(",")]
                    raw_text = process_selected_pages(pdf_docs, pages)
                else:
                    raw_text = get_pdf_text(pdf_docs)
                
                st.write("**Summarizing Document...**")
                summary = summarize_text(raw_text)
                st.write(summary)

                # Process and create the conversation
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

    # Download chat history
    if st.session_state.chat_history:
        st.subheader("Download Chat History")
        download_chat_history(st.session_state.chat_history)

if __name__ == '__main__':
    main()
