"# Python-Project" 
Features
Upload multiple PDF files and extract text.
Ask questions and interact with the content through a chat interface.
Summarize the text using a local model.
Perform Optical Character Recognition (OCR) to extract text from scanned PDF images.
Convert speech to text via audio file uploads.
Download chat history as a text file for later reference.
Secure user authentication using environment variables.
Process specific pages from PDFs.
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
Make sure to replace the placeholders with your actual API keys and hashed passwords.

5. Install Tesseract (for OCR)
To enable text extraction from images within PDFs, install Tesseract OCR:

Linux:
bash
Copy code
sudo apt install tesseract-ocr
Mac:
bash
Copy code
brew install tesseract
Windows: Download from the official Tesseract GitHub page.
6. (Optional) Set up Hugging Face API
If using Hugging Face models, you need a Hugging Face account and API key. Set this in your .env file as shown above.

7. Run the Application
Once everything is set up, run the Streamlit application:

bash
Copy code
streamlit run app.py
8. Interact with the App
Upload PDFs: In the sidebar, upload one or more PDF files. You can choose specific pages to process.
Ask Questions: Use the text box to ask questions about the uploaded PDFs.
Audio Input: Optionally upload an audio file to convert speech to text.
Download Chat History: You can download the conversation history from the sidebar.
Summarization: The application summarizes the uploaded PDFs for quick understanding.
Usage Instructions
Login: Use the username and password set in the .env file to log in.
Upload PDFs: You can upload multiple PDFs, and the application will extract and process the text.
Ask Questions: After uploading, ask any question related to the PDFs and receive answers based on the document's content.
Speech-to-Text: Upload an audio file (e.g., .wav or .mp3), and the application will convert the speech to text for querying the documents.
Download Chat: After your conversation, you can download the chat history as a text file.
Summarization: After uploading a document, the app will summarize it using the local T5 model.
File Structure
plaintext
Copy code
├── app.py                        # Main application file
├── htmlTemplates.py               # HTML templates for UI elements
├── requirements.txt               # Required Python libraries
├── .env                           # Environment variables
├── README.md                      # Project README
└── other supporting files
Troubleshooting
Model Errors: Ensure that API keys are correctly set in the .env file.
Authentication Failure: Check if the username and password hashes match the stored values.
OCR Issues: Ensure that Tesseract is correctly installed on your machine.
Future Improvements
Implement additional document formats (e.g., DOCX, TXT).
Improve error handling and add better support for non-English languages.
Add user authentication using OAuth.
License
This project is licensed under the MIT License.

