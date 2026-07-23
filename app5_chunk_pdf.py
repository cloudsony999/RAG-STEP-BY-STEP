'''
Objective

Split a large PDF into smaller chunks.

Instead of storing one huge paragraph, RAG stores many small pieces.

'''

# ======================================================
# Program : app5_chunk_pdf.py
# Purpose : Split PDF into Chunks
# ======================================================

from pypdf import PdfReader

print("="*60)
print("PDF CHUNKING")
print("="*60)

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:

    extracted = page.extract_text()

    if extracted:
        text += extracted + "\n"

print("\nTotal Characters :", len(text))

# -------------------------------
# Chunk Function
# -------------------------------

def create_chunks(text, chunk_size=400):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(text[i:i+chunk_size])

    return chunks

chunks = create_chunks(text)

print("\nTotal Chunks Created :", len(chunks))

print("\nDisplaying Chunks\n")

for i, chunk in enumerate(chunks):

    print("="*60)

    print("Chunk", i + 1)

    print("="*60)

    print(chunk)

print("\nFinished!")

'''
Output
Total Characters : 3200

Total Chunks : 8

Chunk 1

Artificial Intelligence...

-----------------------------------

Chunk 2

Machine Learning...

-----------------------------------

Chunk 3

Deep Learning...

'''