from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

import numpy as np
import faiss

# Load embedding model only once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class RAGEngine:

    def __init__(self, groq_api_key):

        self.client = Groq(api_key=groq_api_key)

        self.chunks = []

        self.index = None

    # ----------------------------
    # Read PDF
    # ----------------------------
    def load_pdf(self, pdf_file):

        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text

    # ----------------------------
    # Chunk PDF
    # ----------------------------
    def chunk_text(self, text, chunk_size=500):

        chunks = []

        for i in range(0, len(text), chunk_size):

            chunks.append(text[i:i + chunk_size])

        return chunks

    # ----------------------------
    # Build Vector Database
    # ----------------------------
    def create_vector_database(self, text):

        self.chunks = self.chunk_text(text)

        vectors = embedding_model.encode(self.chunks)

        vectors = np.array(vectors).astype("float32")

        self.index = faiss.IndexFlatL2(vectors.shape[1])

        self.index.add(vectors)

    # ----------------------------
    # Search
    # ----------------------------
    def search(self, question):

        query_vector = embedding_model.encode([question])

        query_vector = np.array(query_vector).astype("float32")

        distance, indexes = self.index.search(query_vector, 3)

        context = ""

        for i in indexes[0]:

            context += self.chunks[i] + "\n"

        return context

    # ----------------------------
    # Ask GROQ
    # ----------------------------
    def ask(self, question):

        context = self.search(question)

        prompt = f"""
You are a helpful AI Assistant.

Answer only using the supplied context.

Context:

{context}

Question:

{question}
"""

        response = self.client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2

        )

        return response.choices[0].message.content
