# SVR Engineering College AI Assistant

A **Retrieval-Augmented Generation (RAG) based AI chatbot** developed for SVR Engineering College.
The chatbot allows users to ask questions about the college and provides answers by retrieving relevant information from the college PDF using semantic search and Google Gemini.



```
## 🚀 Features

- PDF-based college knowledge base
- Retrieval-Augmented Generation (RAG)
- Semantic search using ChromaDB
- Text embeddings using Sentence Transformers
- Google Gemini integration
- Gemini-based query rewriting
- Spelling mistake understanding
- Conversation memory
- Flask web application
- Context-based AI responses
- College-specific answers
```
```
## 🛠️ Technologies Used

- Python
- Flask
- Google Gemini API
- ChromaDB
- Sentence Transformers
- LangChain Text Splitters
- PyPDF
- HTML
- CSS
- JavaScript
- python-dotenv
```
```
## 🔄 Project Workflow

College PDF
↓
PDF Text Extraction
↓
Text Chunking
↓
Sentence Transformer
↓
Generate Embeddings
↓
vector DB
↓
User Question
↓
Gemini Query Rewriting
↓
Generate Query Embedding
↓
ChromaDB Semantic Search
↓
Retrieve Relevant Context
↓
Conversation History + Context
↓
Google Gemini
↓
AI Generated Answer
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```
##  gitignore
```
.env
venv/
__pycache__/
*.pyc
```
## Clone the repository:
```
git clone https://github.com/ThulaseswaraReddy/SVREC-AI-ASSISTANT.git
cd SVREC-AI-ASSISTANT
```
## Create a virtual environment:
```
python -m venv venv
```
## Activate the virtual environment on Windows:
```
venv\Scripts\activate
```
## ▶️ Run the Project
```
Start the Flask application:
python app.py
```
## 👨‍💻 Author
```
Thulaseswara Reddy
AI Engineering Student
```
