import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import time
import base64
import speech_recognition as sr
import os

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
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Set up the conversational chain."""
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.3, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

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

def audio_to_text():
    """Capture audio input and convert to text."""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    st.info("Listening... Speak into your microphone.")
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand your speech.")
        return None
    except sr.RequestError as e:
        st.error(f"Error with the speech recognition service: {e}")
        return None

# User Authentication
def user_authentication():
    if "user_authenticated" not in st.session_state:
        st.session_state.user_authenticated = False

    if not st.session_state.user_authenticated:
        st.sidebar.subheader("Login / Signup")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        choice = st.sidebar.radio("Action", ["Login", "Signup"])

        if st.sidebar.button("Submit"):
            if choice == "Signup":
                # Add logic to save new user credentials
                st.session_state.user_authenticated = True
                st.success("Signup successful. You're now logged in!")
            elif choice == "Login":
                # Add logic to validate credentials
                st.session_state.user_authenticated = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials!")
    else:
        st.sidebar.success("You're logged in!")
        if st.sidebar.button("Logout"):
            st.session_state.user_authenticated = False

# Main Function
def main():
    load_dotenv()
    st.set_page_config(page_title="Enhanced PDF Chat", page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_authentication()
    if not st.session_state.user_authenticated:
        st.stop()

    st.header("Chat with Multiple PDFs :books:")
    st.subheader("1. Upload PDFs in the sidebar")
    st.subheader("2. Process and chat with the content")

    # Sidebar
    with st.sidebar:
        st.subheader("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process PDFs") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                if st.session_state.conversation:
                    st.success("Conversation chain successfully initialized!")

    # Chat Interface
    user_question = st.text_input("Ask a question:")
    if st.button("Use Audio Input"):
        audio_input = audio_to_text()
        if audio_input:
            user_question = audio_input

    if user_question:
        if st.session_state.conversation:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            for i, message in enumerate(st.session_state.chat_history):
                role = "User" if i % 2 == 0 else "Bot"
                st.markdown(f"**{role}:** {message.content}")
        else:
            st.error("Please upload and process a document to start the chat.")

    # Download chat history
    if st.session_state.chat_history:
        st.subheader("Download Chat History")
        download_chat_history(st.session_state.chat_history)

if __name__ == '__main__':
    main()
