Features

1.Upload multiple PDF files and extract text.

2.Ask questions and interact with the content through a chat interface.

3.Summarize the text using a local model.

4.Perform Optical Character Recognition (OCR) to extract text from scanned PDF images.

5.Convert speech to text via audio file uploads.

6.Download chat history as a text file for later reference.

7.Secure user authentication using environment variables.

8.Process specific pages from PDFs.


Tech Stack

Frontend: Streamlit

Backend: LangChain, Hugging Face Transformers, PyPDF2, FAISS, OpenAI API

Speech Recognition: speech_recognition (Google API)

OCR: pytesseract (Tesseract OCR)

Prerequisites

Before running the project, ensure that you have the following installed:

Python 3.7+

Git

Pip (Python package installer)

You also need accounts for:

Hugging Face API: Used for summarization and conversational models.

OpenAI API: Optional, if using OpenAI for chat models.

Environment variables set up for authentication.

Installation
Follow these steps to set up and run the project on your local machine.

1. Clone the repository
bash
Copy code
git clone https://github.com/TanishqMalik707/Python-Project.git
cd Python-Project
2. Set up a virtual environment
It's recommended to create a virtual environment to manage dependencies:

bash
Copy code
# For Linux/Mac
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
3. Install required packages
Install the dependencies listed in the requirements.txt:

bash
Copy code
pip install -r requirements.txt
4. Set up Environment Variables
Create a .env file in the project root directory and add the following environment variables. This will be used for secure authentication and API integrations:

php
Copy code
OPENAI_API_KEY=<Your-OpenAI-API-Key>
HUGGINGFACE_API_KEY=<Your-Huggingface-API-Key>
<Your-username>=<Your-password-hash>

