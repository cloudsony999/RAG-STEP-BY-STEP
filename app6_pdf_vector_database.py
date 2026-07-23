'''
Objective

Convert PDF chunks into embeddings and store them inside FAISS.

'''


# ======================================================
# Program : app6_pdf_vector_database.py
# Purpose : Store PDF Chunks in FAISS
# ======================================================

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import numpy as np
import faiss

print("="*60)
print("PDF VECTOR DATABASE")
print("="*60)

# -------------------------------
# Read PDF
# -------------------------------

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:

    extracted = page.extract_text()

    if extracted:
        text += extracted + "\n"

# -------------------------------
# Create Chunks
# -------------------------------

def chunk_text(text, chunk_size=400):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(text[i:i+chunk_size])

    return chunks

chunks = chunk_text(text)

print("\nChunks Created :", len(chunks))

# -------------------------------
# Embedding Model
# -------------------------------

print("\nLoading Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully")

# -------------------------------
# Convert Chunks to Embeddings
# -------------------------------

vectors = model.encode(chunks)

vectors = np.array(vectors).astype("float32")

print("\nEmbedding Shape")

print(vectors.shape)

# -------------------------------
# Create Vector Database
# -------------------------------

dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(vectors)

print("\nVectors Stored :", index.ntotal)

# -------------------------------
# User Query
# -------------------------------

query = input("\nAsk something about the PDF : ")

query_vector = model.encode([query])

query_vector = np.array(query_vector).astype("float32")

distance, indexes = index.search(query_vector, 3)

print("\nTop Matching Chunks")

print("="*60)

for rank, idx in enumerate(indexes[0], start=1):

    print("\nRank :", rank)

    print("Distance :", distance[0][rank-1])

    print()

    print(chunks[idx])

    print("="*60)

print("\nSearch Completed!")

'''
Suppose your PDF contains

Python

Machine Learning

Deep Learning

Neural Networks

Generative AI

Large Language Models

RAG

Groq

Ask

Ask something about the PDF :

What is Deep Learning?

Output

Rank 1

Deep Learning is a subset of Machine Learning...

--------------------------------------------

Rank 2

Neural Networks consist of multiple layers...

--------------------------------------------

Rank 3

Artificial Intelligence...

Notice something important.

We have NOT used GROQ yet.

The program itself finds the most relevant PDF chunks using embeddings and the FAISS vector database.

This is called Retrieval.

The final step (Program 7) is Retrieval-Augmented Generation (RAG):

PDF
   ↓
Read PDF
   ↓
Chunk PDF
   ↓
Embeddings
   ↓
FAISS Search
   ↓
Top 3 Chunks
   ↓
Send Chunks + User Question
   ↓
GROQ Llama Model
   ↓
Final Answer

That last program completes the end-to-end RAG pipeline by combining 
retrieval with a large language model.

'''

