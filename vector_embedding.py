# Di sini tertera pada hasil similiraty score pada vector embedding

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os

import pandas as pd
import numpy as np

# Muat file konfigurasi dari .env
load_dotenv()
pdf_file = "yolov9_paper.pdf"  # Nama file PDF yang akan diproses

# List untuk menyimpan konten PDF
pdf_content = []

try:
    # Baca file PDF
    pdf_loader = PyPDFLoader(pdf_file)
    loaded_data = pdf_loader.load()
    pdf_content.extend(loaded_data)
    print(f"Sukses memuat {len(loaded_data)} halaman dari {pdf_file}")
except Exception as error:
    print(f"Terjadi kesalahan saat memuat PDF {pdf_file}: {error}")

if pdf_content:
    # Potong dokumen menjadi bagian-bagian kecil
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    split_docs = splitter.split_documents(pdf_content)
    print(f"Total bagian dokumen yang diproses: {len(split_docs)}")
else:
    print("Tidak ada data yang dapat diproses.")

# Inisialisasi embeddings dan penyimpanan vektor
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-max-tokens")

    # Simpan vektor ke direktori 'data'
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory="data"
    )
    retriever_tool = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    print("Vektor berhasil dibuat dan disimpan.")
except Exception as error:
    print(f"Kesalahan saat inisialisasi embeddings atau penyimpanan vektor: {error}")

# Buat LLM dan chain untuk RAG
try:
    generative_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

    # Siapkan memori percakapan
    chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Buat template prompt dengan memori
    prompt_template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Anda adalah sebuah Pakar yang dapat Mengidentifikasi Jaringan LAN. "
                "Anda dapat memberikan Jawaban secara Fakta. "

            ),
            MessagesPlaceholder(variable_name="chat_history"),  # Memori otomatis dimasukkan di sini
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Buat chain percakapan
    conversation_chain = LLMChain(
        llm=generative_model,
        prompt=prompt_template,
        memory=chat_memory,
        verbose=True
    )

    # Pertanyaan pengguna
    user_query = "Apa itu jaringan lan?"

    # Proses pertanyaan menggunakan chain
    response = conversation_chain.run(question=user_query)
    print("Jawaban:", response)

    # Ambil dokumen relevan berdasarkan similarity
    relevant_docs = retriever_tool.invoke(user_query)

    # Hitung kesamaan antara jawaban dan dokumen yang diambil
    similarities = []
    query_vector = embedding_model.embed_query(response)

    for doc in relevant_docs:
        doc_vector = embedding_model.embed_query(doc.page_content)
        similarity_score = cosine_similarity([query_vector], [doc_vector])[0][0]
        similarities.append((doc.page_content, similarity_score))

    # Urutkan hasil berdasarkan nilai kemiripan
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    print("Hasil pencarian berdasarkan tingkat kemiripan:")
    for content, score in similarities:
        print(f"Kemiripan: {score:.4f}")

except Exception as error:
    print(f"Terjadi kesalahan saat menginisialisasi model atau memori: {error}")

# Buat DataFrame untuk menyimpan hasil perhitungan
similarity_data = []
query_vector = np.array(embedding_model.embed_query(response))

# Simpan semua vektor dataset dalam satu array
dataset_vectors = [np.array(embedding_model.embed_query(doc.page_content)) for doc in relevant_docs]

# Konversi ke array numpy untuk perhitungan
vector_dataset_matrix = np.array(dataset_vectors)
vector_query = np.tile(query_vector, (len(dataset_vectors), 1))

# Hitung dot product dan magnitudo
sumproduct_values = np.sum(vector_dataset_matrix * vector_query, axis=1)
magnitude_dataset = np.sqrt(np.sum(vector_dataset_matrix**2, axis=1))
magnitude_query = np.sqrt(np.sum(query_vector**2))

cosine_similarity_scores = sumproduct_values / (magnitude_dataset * magnitude_query)

# Simpan hasil dalam DataFrame dengan format yang lebih rapi
columns = [f"Embed{i+1}" for i in range(vector_dataset_matrix.shape[1])]
df_vectors = pd.DataFrame(vector_dataset_matrix, columns=columns)
df_vectors.insert(0, "Subjudul", [f"Dimensi {i+1}" for i in range(len(relevant_docs))])
df_vectors.insert(1, "Vector Dataset", magnitude_dataset)
df_vectors.insert(2, "Vector Embedding 1", vector_query[:, 0])
df_vectors["Similarity 1"] = cosine_similarity_scores


import numpy as np
import pandas as pd


# Mengurutkan hasil berdasarkan similarity
try:
    similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)
except NameError:
    print("Variable 'similarities' not found. Check if the previous cell executed successfully.")
    similarities_sorted = []  # Initialize as empty list to avoid further errors

# Menyusun data untuk ditulis ke Excel
# Assign query_vector to query_embedding
# Use 'embedding_model' instead of 'embeddings'
query_embedding = embedding_model.embed_query(response)  # Assuming 'response' holds the query text
vector_embeddings = np.array([query_embedding])  # Misal embedding hasil model LLM
# Use 'embedding_model' instead of 'embeddings' for consistency
dataset_vectors = np.array([embedding_model.embed_query(doc[0]) for doc in similarities_sorted])

# Menyusun data dalam bentuk dataframe
columns = ['Subjudul', 'Vector Dataset'] + [f'Vector Embeddings {i+1}' for i in range(len(similarities_sorted))]
rows = []

# Menyusun dimensi dan vektor ke dalam bentuk yang sesuai
for dim in range(len(vector_embeddings[0])):  # Misalkan dimensi 768
    row = ['Dimensi ' + str(dim+1), vector_embeddings[0][dim]]  # Vector Hasil Embedding
    for doc_vector in dataset_vectors:
        row.append(doc_vector[dim])  # Vector Dataset
    rows.append(row)

# Membuat DataFrame
df = pd.DataFrame(rows, columns=columns)

# Menulis DataFrame ke Excel
output_file = "embedding_manual_laporan.xlsx"
df.to_excel(output_file, index=False)

print(f"File Excel disimpan di {output_file}")