# Objective

# Store embeddings inside a FAISS Vector Database and search similar documents.

# ==============================================
# Program : app3_vector_database.py
# Purpose : Create and Search FAISS Vector DB
# ==============================================

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("=" * 60)
print("VECTOR DATABASE USING FAISS")
print("=" * 60)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents
documents = [

    "Python is a programming language.",

    "Machine Learning is a branch of AI.",

    "The Taj Mahal is in India.",

    "Groq provides very fast AI inference.",

    "Cats are domestic animals."
]

print("\nDocuments\n")

for i, doc in enumerate(documents):
    print(i, "->", doc)

# Convert documents into vectors
print("\nGenerating Embeddings...")

vectors = model.encode(documents)

vectors = np.array(vectors).astype("float32")

print("Embedding Shape :", vectors.shape)

# Create FAISS Index
dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

# Store vectors
index.add(vectors)

print("\nTotal Vectors Stored :", index.ntotal)

# Ask user
query = input("\nEnter your search query : ")

# Convert query into embedding
query_vector = model.encode([query])

query_vector = np.array(query_vector).astype("float32")

# Search Top 3 Results
distance, indexes = index.search(query_vector, 3)

print("\nSearch Results")

print("-" * 60)

for rank, idx in enumerate(indexes[0], start=1):

    print("Rank :", rank)

    print("Document :", documents[idx])

    print("Distance :", distance[0][rank - 1])

    print("-" * 60)

print("\nProgram Finished Successfully!")

'''
Sample Run
Enter your search query :

Tell me about Python

Output

Search Results

Rank : 1

Python is a programming language.

Distance : 0.42

--------------------------------------

Rank : 2

Machine Learning is a branch of AI.

Distance : 0.96

--------------------------------------

Rank : 3

Groq provides very fast AI inference.

Distance : 1.31
What You Have Learned
Program	Concept	Output
app1_embedding.py	Convert text into embeddings	384-dimensional vectors
app2_similarity.py	Compare embeddings	Cosine similarity scores
app3_vector_database.py	Store and search embeddings	Retrieve the most 
similar documents

These three programs provide the foundation for understanding RAG. 
In the next programs, you'll read a PDF, 
split it into chunks, store those chunks in a FAISS vector database, 
and finally build a complete PDF Question Answering application 
using the GROQ API.

'''