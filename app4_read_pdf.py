'''
These programs take us from plain text to PDF-based Retrieval, which is the heart of RAG.

Embedding ✔
        ↓
Similarity ✔
        ↓
Vector Database ✔
        ↓
Read PDF  ← Program 4
        ↓
Split PDF into Chunks ← Program 5
        ↓
Store PDF Chunks in FAISS ← Program 6
        ↓
RAG (Program 7)

Objective

Read text from a PDF file.

Place a file named sample.pdf in the project folder.

RAG_Project/

sample.pdf
app4_read_pdf.py

'''


# ======================================================
# Program : app4_read_pdf.py
# Purpose : Read a PDF file
# ======================================================

from pypdf import PdfReader

print("="*60)
print("READING PDF FILE")
print("="*60)

pdf_file = "sample.pdf"

reader = PdfReader(pdf_file)

print("\nTotal Pages :", len(reader.pages))

print("\nReading PDF...\n")

full_text = ""

for page_number, page in enumerate(reader.pages):

    text = page.extract_text()

    print("-"*60)
    print("Page :", page_number + 1)
    print("-"*60)

    print(text)

    full_text += text + "\n"

print("="*60)

print("Complete PDF Text")

print("="*60)

print(full_text)

print("\nTotal Characters :", len(full_text))

'''
Sample Output
READING PDF FILE

Total Pages : 3

Page : 1

Artificial Intelligence is the simulation...

Page : 2

Machine Learning is a subset...

Page : 3

Deep Learning uses Neural Networks...

Total Characters : 3560

'''
