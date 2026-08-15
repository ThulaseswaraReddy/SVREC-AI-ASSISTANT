import os
import chromadb
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import Flask, render_template, request, jsonify
load_dotenv()

app = Flask(__name__)
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="svr_college_documents"
)
pdf_file = "SVRDoc.pdf"
reader = PdfReader(pdf_file)
document = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        document += text + "\n"
if not document.strip():
    print("Could not extract text from the PDF.")
    exit()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(document)
print("PDF loaded successfully.")
print("Number of chunks:", len(chunks))
embeddings = embedding_model.encode(
    chunks
).tolist()
collection.add(
    ids=[
        str(i)
        for i in range(len(chunks))
    ],
    documents=chunks,
    embeddings=embeddings
)
print("College information stored in ChromaDB.")
chat_history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global chat_history
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({
            "answer": "Please enter a question."
        })
    rewrite_response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"""
Rewrite the user's question into a clear search query
for a college document.

Correct spelling mistakes and understand the user's intent.

If the user asks about:
- number of departments
- list of departments
- departments available
- HODs
- college management
- fees
- transportation
- hostel
- placements

make the search query explicitly mention the relevant topic.

Do not answer the question.
Only return the rewritten search query.

User question:
{question}
"""
    )
    search_query = rewrite_response.text.strip()
    print("Original question:", question)
    print("Search query:", search_query)
    query_embedding = embedding_model.encode(
        search_query
    ).tolist()
    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=8
    )
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(
        retrieved_chunks
    )
    history = "\n".join(
        chat_history
    )
    prompt = f"""
You are the AI assistant for
SVR Engineering College, Nandyal.

Answer the user's question using the
college context and conversation history.

IMPORTANT RULES:

1. Use ONLY information from the college context.

2. You may use conversation history to
understand follow-up questions.

3. Do NOT use outside knowledge.

4. Do NOT guess.

5. Do NOT invent information.

6. If the required information is not
available, say:

"I don't have that information."

CONVERSATION HISTORY:
{history}

COLLEGE CONTEXT:
{context}

CURRENT QUESTION:
{question}
"""
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    answer = response.text
    chat_history.append(
        f"User: {question}"
    )
    chat_history.append(
        f"AI: {answer}"
    )
    return jsonify({
        "answer": answer
    })

if __name__ == "__main__":
    app.run()
