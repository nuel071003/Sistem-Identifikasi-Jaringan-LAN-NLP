from flask import Flask, render_template, request, jsonify
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from gtts import gTTS
import os
import uuid  # Import uuid for unique file names

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Path to PDF file
pdf_path = "yolov9_paper.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

# Create vectorstore
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Load Language Model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=2500,
    timeout=None
)

# Define system prompt
system_prompt = (
    "Anda adalah asisten untuk tugas tanya jawab. "
    "Gunakan bagian konteks yang diambil berikut untuk menjawab "
    "pertanyaan. Jika Anda tidak tahu jawabannya, katakan bahwa Anda "
    "tidak tahu. Gunakan maksimal tiga kalimat dan pertahankan "
    "jawaban ringkas.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# Routes
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/konsultasi")
def konsultasi():
    return render_template('konsultasi.html')

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        query = request.json.get("query")
        
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Dapatkan dokumen relevan
        documents = retriever.get_relevant_documents(query)
        
        # Buat dan panggil rantai
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )
        
        # Dapatkan respons
        response = document_chain.invoke({
            "input": query,
            "context": documents
        })
        
        # Periksa apakah respons berisi jawaban
        if isinstance(response, dict) and "answer" in response:
            answer = response["answer"]
        else:
            answer = str(response)

        # Generate audio from answer with unique filename
        unique_id = str(uuid.uuid4())  # Generate a unique identifier
        audio_file = f"static/response_{unique_id}.mp3"  # Unique file name
        tts = gTTS(answer, lang='id')
        tts.save(audio_file)

        return jsonify({"response": answer, "audio": audio_file})

    except Exception as e:
        print(f"Error in ask_question: {str(e)}")  # Debugging
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)