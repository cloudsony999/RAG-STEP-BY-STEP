'''
It performs the complete RAG (Retrieval-Augmented Generation) pipeline:

Read the PDF
Split it into chunks
Create embeddings
Store embeddings in FAISS
Search for relevant chunks
Send the retrieved context + user question to the GROQ LLM
Display the final answer

'''

# ============================================================
# Program : app7_complete_rag_groq.py
# Purpose : Complete RAG using FAISS + GROQ
# ============================================================

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

import numpy as np
import faiss

# ============================================================
# STEP 1 : Enter Your GROQ API Key
# ============================================================

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

# ============================================================
# STEP 2 : Read PDF
# ============================================================

print("=" * 70)
print("STEP 1 : READING PDF")
print("=" * 70)

reader = PdfReader("sample.pdf")

pdf_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:
        pdf_text += text + "\n"

print("PDF Loaded Successfully")

print("Total Characters :", len(pdf_text))

# ============================================================
# STEP 3 : Split into Chunks
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 : CHUNKING PDF")
print("=" * 70)


def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(text[i:i + chunk_size])

    return chunks


chunks = create_chunks(pdf_text)

print("Total Chunks :", len(chunks))

# ============================================================
# STEP 4 : Load Embedding Model
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 : LOADING EMBEDDING MODEL")
print("=" * 70)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding Model Loaded")

# ============================================================
# STEP 5 : Convert Chunks into Embeddings
# ============================================================

print("\nGenerating Embeddings...")

vectors = model.encode(chunks)

vectors = np.array(vectors).astype("float32")

print("Embedding Shape :", vectors.shape)

# ============================================================
# STEP 6 : Create FAISS Vector Database
# ============================================================

print("\nCreating Vector Database...")

dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(vectors)

print("Vectors Stored :", index.ntotal)

# ============================================================
# STEP 7 : Connect to GROQ
# ============================================================

client = Groq(api_key=GROQ_API_KEY)

print("\nConnected to GROQ")

# ============================================================
# STEP 8 : Question Loop
# ============================================================

while True:

    print("\n" + "=" * 70)

    question = input("Ask a Question (type exit to quit): ")

    if question.lower() == "exit":
        print("\nGood Bye!")
        break

    # --------------------------------------
    # Convert Question into Embedding
    # --------------------------------------

    question_vector = model.encode([question])

    question_vector = np.array(question_vector).astype("float32")

    # --------------------------------------
    # Retrieve Top 3 Chunks
    # --------------------------------------

    distances, indexes = index.search(question_vector, 3)

    context = ""

    print("\nRetrieved Chunks")
    print("-" * 70)

    for i in indexes[0]:

        print(chunks[i])
        print("-" * 70)

        context += chunks[i] + "\n"

    # --------------------------------------
    # Prompt
    # --------------------------------------

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information given below.

If the answer is not present in the context,
reply exactly:

"I could not find the answer in the supplied PDF."

==========================
PDF Context
==========================

{context}

==========================
User Question
==========================

{question}

==========================
Answer
==========================
"""

    print("\nGenerating Answer from GROQ...\n")

    # --------------------------------------
    # Call GROQ
    # --------------------------------------

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_tokens=500

    )

    answer = response.choices[0].message.content

    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(answer)

    print("=" * 70)



'''

Folder Structure
RAG_Project/
│
├── sample.pdf
├── requirements.txt
│
├── app1_embedding.py
├── app2_similarity.py
├── app3_vector_database.py
├── app4_read_pdf.py
├── app5_chunk_pdf.py
├── app6_pdf_vector_database.py
└── app7_complete_rag_groq.py
Running the Final Program
python app7_complete_rag_groq.py
Sample Console Output
======================================================================
STEP 1 : READING PDF
======================================================================

PDF Loaded Successfully
Total Characters : 5820

======================================================================
STEP 2 : CHUNKING PDF
======================================================================

Total Chunks : 12

======================================================================
STEP 3 : LOADING EMBEDDING MODEL
======================================================================

Embedding Model Loaded

Generating Embeddings...

Embedding Shape : (12, 384)

Creating Vector Database...

Vectors Stored : 12

Connected to GROQ

User enters:

Ask a Question:

What is Machine Learning?

The program retrieves relevant chunks:

Retrieved Chunks

------------------------------------------------------------

Machine Learning is a subset of Artificial Intelligence...

------------------------------------------------------------

Supervised Learning uses labelled data...

------------------------------------------------------------

Deep Learning is based on Neural Networks...

Then GROQ generates:

FINAL ANSWER

Machine Learning is a subset of Artificial Intelligence that
enables computers to learn patterns from data and make
predictions or decisions without being explicitly programmed
for every task.


'''