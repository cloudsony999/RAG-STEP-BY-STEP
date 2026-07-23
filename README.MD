# RAG-STEP-BY-STEP 📚

**A hands-on, incremental walkthrough of Retrieval-Augmented Generation (RAG) — from raw sentence embeddings to a full PDF Question-Answering app powered by FAISS and GROQ.**

Instead of dropping you into one large, opaque RAG codebase, this repository builds the concept **one Python file at a time**. Each script is a self-contained lesson that introduces exactly one new idea, so by the time you reach the final program, every moving part is already familiar.

---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [Core Concepts Covered](#core-concepts-covered)
- [Architecture](#architecture)
- [Learning Path (File-by-File)](#learning-path-file-by-file)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running Each Program](#running-each-program)
- [Running the Streamlit App](#running-the-streamlit-app)
- [How the Streamlit App Works](#how-the-streamlit-app-works)
- [Suggested Improvements](#suggested-improvements)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

---

## Why This Project Exists

Most RAG tutorials show a finished pipeline and expect you to reverse-engineer how embeddings, vector search, chunking, and LLM prompting fit together. This repo does the opposite: it isolates each concept into its own runnable script, in increasing order of complexity, so learners can:

1. See exactly what an embedding vector looks like.
2. Understand how similarity search actually works under the hood.
3. Watch a vector database get built and queried, first with plain text, then with real PDF content.
4. Assemble all of it into a single, working Retrieval-Augmented Generation pipeline.
5. Wrap that pipeline in a simple Streamlit web UI.

---

## Core Concepts Covered

| Concept | What You'll Learn |
|---|---|
| **Sentence Embeddings** | Converting text into dense numerical vectors using `sentence-transformers` |
| **Cosine Similarity** | Measuring semantic closeness between two pieces of text |
| **Vector Databases** | Storing and searching embeddings efficiently with FAISS |
| **PDF Ingestion** | Extracting raw text from PDF documents with `pypdf` |
| **Chunking** | Splitting long documents into retrievable, bite-sized pieces |
| **Retrieval** | Finding the most relevant chunks for a given query |
| **Augmented Generation** | Feeding retrieved context + question into an LLM (GROQ's Llama 3.3 70B) to get a grounded answer |
| **Productization** | Wrapping the pipeline into a reusable class (`RAGEngine`) and a Streamlit UI |

---

## Architecture

The final pipeline (`app7_complete_rag_groq.py` / `rag_engine.py`) follows the classic RAG flow:

```
                 User Uploads / Points to PDF
                            │
                            ▼
                  Read PDF Text (pypdf)
                            │
                            ▼
              Split Text into Fixed-Size Chunks
                            │
                            ▼
     Generate Embeddings for Each Chunk (SentenceTransformer:
                 all-MiniLM-L6-v2, 384-dim vectors)
                            │
                            ▼
        Store Embeddings in a FAISS Vector Index
              (IndexFlatL2 — exact L2 search)
                            │
                            ▼
                  User Asks a Question
                            │
                            ▼
           Embed the Question with the Same Model
                            │
                            ▼
       Search FAISS for Top-K (default 3) Nearest Chunks
                            │
                            ▼
     Build a Prompt: Retrieved Context + User Question
                            │
                            ▼
        Send Prompt to GROQ (llama-3.3-70b-versatile)
                            │
                            ▼
             GROQ Returns a Context-Grounded Answer
                            │
                            ▼
          Displayed in Console or Streamlit UI
```

---

## Learning Path (File-by-File)

The scripts are numbered so you can work through them in order. Each one builds directly on the previous concept.

### 1️⃣ `app1_embedding.py` — Understanding Sentence Embeddings
Loads the `all-MiniLM-L6-v2` model from `sentence-transformers` and encodes three sample sentences, printing the resulting 384-dimensional vectors. This demystifies what "an embedding" actually is before any retrieval logic is introduced.

### 2️⃣ `app2_similarity.py` — Comparing Meaning with Cosine Similarity
Takes embeddings for multiple sentences and computes a pairwise cosine similarity matrix using `scikit-learn`. Demonstrates that semantically related sentences ("Python is a programming language" vs. "Python is used for coding") score close to `1.0`, while unrelated ones score close to `0`.

### 3️⃣ `app3_vector_database.py` — Building Your First Vector Database
Introduces **FAISS** (`IndexFlatL2`). Five sample documents are embedded and stored in a flat L2 index. The script then accepts a live query from the terminal, embeds it, and retrieves the top 3 nearest documents by distance — the essence of "retrieval."

### 4️⃣ `app4_read_pdf.py` — Reading Real Documents
Swaps toy sentences for a real file: `sample.pdf`. Uses `pypdf.PdfReader` to extract text page-by-page and concatenates it into one string, printing page count and total character count.

### 5️⃣ `app5_chunk_pdf.py` — Chunking Strategy
A PDF's full text is almost always too long (and too unfocused) to embed as a single vector. This script introduces a simple fixed-size chunking function (`chunk_size=400`) that splits the extracted text into smaller, independently searchable pieces.

### 6️⃣ `app6_pdf_vector_database.py` — Combining Chunking + FAISS
Merges steps 4 and 5: reads the PDF, chunks it, embeds every chunk, and stores the vectors in FAISS. A user query then retrieves the top 3 most relevant chunks — pure retrieval, still with **no LLM involved yet**, to make clear that retrieval and generation are separate concerns.

### 7️⃣ `app7_complete_rag_groq.py` — The Complete RAG Pipeline
The capstone script. It performs the entire pipeline end-to-end in a command-line loop:
- Reads and chunks `sample.pdf`
- Embeds the chunks and builds a FAISS index
- Accepts questions in a `while True` loop
- Retrieves the top 3 relevant chunks per question
- Constructs a strict, context-only prompt (the model is instructed to reply `"I could not find the answer in the supplied PDF."` if the answer isn't present)
- Sends the prompt to **GROQ's `llama-3.3-70b-versatile`** model
- Prints the grounded answer

### 🧩 `rag_engine.py` — Refactored into a Reusable Class
Everything from `app7` is reorganized into a `RAGEngine` class with clean methods:
- `load_pdf(pdf_file)` — extract text
- `chunk_text(text, chunk_size=500)` — split into chunks
- `create_vector_database(text)` — embed + index
- `search(question)` — retrieve top-3 chunks
- `ask(question)` — retrieve + prompt GROQ + return the answer

This class is the reusable "engine" that powers the web UI.

### 🖥️ `app.py` — Streamlit Web Application
The final deliverable: a browser-based UI built with Streamlit that wraps `RAGEngine`. Users upload a PDF, the app indexes it in-session (`st.session_state.engine`), and they can then ask free-form questions and get answers rendered directly in the browser — no terminal required.

---

## Project Structure

```
RAG-STEP-BY-STEP/
│
├── app.py                        # Streamlit web UI (final application)
├── rag_engine.py                 # RAGEngine class used by app.py
│
├── app1_embedding.py              # Step 1: Sentence embeddings
├── app2_similarity.py             # Step 2: Cosine similarity
├── app3_vector_database.py        # Step 3: FAISS with plain text
├── app4_read_pdf.py                # Step 4: Reading PDFs
├── app5_chunk_pdf.py               # Step 5: Chunking PDF text
├── app6_pdf_vector_database.py     # Step 6: FAISS with PDF chunks
├── app7_complete_rag_groq.py        # Step 7: Full RAG pipeline (console)
│
├── requirements.txt                # Python dependencies
├── .env                             # GROQ_API_KEY (not committed with a real key)
└── sample.pdf                       # ⚠️ You must add this yourself (see below)
```

> **Note:** `sample.pdf` is referenced by `app4`–`app7` but is not included in the repository. Add any text-based PDF named `sample.pdf` to the project root before running those scripts.

---

## Requirements

Everything needed is listed in `requirements.txt`:

```
groq>=0.31.0
sentence-transformers>=5.0.0
faiss-cpu>=1.11.0
pypdf>=6.0.0
numpy>=2.3.0
scikit-learn>=1.7.0
torch>=2.7.0
transformers>=4.53.0
huggingface-hub>=0.33.0
tokenizers>=0.21.0
safetensors>=0.5.0
streamlit>=1.47.0
python-dotenv>=1.1.0
```

**Python version:** 3.9+ recommended (required by recent `sentence-transformers` / `torch` releases).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/cloudsony999/RAG-STEP-BY-STEP.git
cd RAG-STEP-BY-STEP

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The first run of any script that uses `SentenceTransformer("all-MiniLM-L6-v2")` will download the model (~80MB) from Hugging Face and cache it locally.

---

## Environment Variables

The console scripts (`app4`–`app6`) don't need an API key at all — they only perform retrieval. Scripts that call GROQ need a valid key.

1. Get a free API key from [GROQ Console](https://console.groq.com).
2. Create or edit the `.env` file in the project root:

```env
GROQ_API_KEY='your_actual_groq_api_key_here'
```

3. `app.py` loads this automatically via `python-dotenv`.

> **Important:** `app7_complete_rag_groq.py` currently hardcodes `GROQ_API_KEY = "YOUR_GROQ_API_KEY"` at the top of the file rather than reading from `.env`. Replace that placeholder directly with your key (or refactor it to use `os.getenv("GROQ_API_KEY")` like `app.py` does) before running it.

---

## Running Each Program

Work through the scripts in order to build understanding incrementally:

```bash
python app1_embedding.py
python app2_similarity.py
python app3_vector_database.py        # prompts for a search query in the terminal

# Add sample.pdf to the project root before running the next scripts
python app4_read_pdf.py
python app5_chunk_pdf.py
python app6_pdf_vector_database.py     # prompts for a question about the PDF

# Add your GROQ API key inside the file first
python app7_complete_rag_groq.py       # interactive Q&A loop, type 'exit' to quit
```

---

## Running the Streamlit App

Once you're comfortable with the underlying pipeline, launch the polished web version:

```bash
streamlit run app.py
```

This opens a local browser tab where you can:
1. Upload any PDF via the file uploader.
2. Watch it get read and indexed (`PDF Indexed Successfully!`, with a character count shown).
3. Type a question in the text box and click **Get Answer**.
4. See the GROQ-generated, context-grounded answer displayed on the page.

---

## How the Streamlit App Works

```
        User Uploads PDF
                │
                ▼
       Read PDF using PyPDF
                │
                ▼
      Split into Text Chunks
                │
                ▼
Create Embeddings (SentenceTransformer)
                │
                ▼
    Store Vectors in FAISS Index
                │
                ▼
        User Types Question
                │
                ▼
Convert Question to an Embedding
                │
                ▼
 Search Top Matching Chunks in FAISS
                │
                ▼
Send Retrieved Context + Question to GROQ
                │
                ▼
      GROQ Generates Final Answer
                │
                ▼
    Streamlit Displays the Answer
```

The `RAGEngine` instance is stored in `st.session_state` so the FAISS index persists across interactions within the same browser session, avoiding re-indexing the PDF on every question.

---

## Suggested Improvements

This project is intentionally kept simple as a teaching tool. Natural next steps for anyone extending it:

- **Chat history** — Use `st.session_state.messages` to support multi-turn conversation instead of single-shot Q&A.
- **Retrieval transparency** — Show the top retrieved chunks in an expandable UI section so users can see *why* the model answered the way it did.
- **Better chunking** — Switch to overlapping chunks (e.g., 500 characters with a 100-character overlap) so important information isn't split across chunk boundaries.
- **Multi-document support** — Combine multiple PDFs into a single FAISS index instead of one file at a time.
- **Persistent vector storage** — Save the FAISS index to disk (`faiss.write_index`) so a PDF doesn't need to be re-embedded every time the app restarts.
- **Configurable API key handling** — Have `app7_complete_rag_groq.py` read from `.env` via `python-dotenv`, matching `app.py`, instead of hardcoding the key in source.
- **Top-K as a parameter** — Expose the number of retrieved chunks (currently hardcoded to 3) as a user-adjustable setting.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`, 384-dim) |
| Vector Search | [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatL2`) |
| PDF Parsing | [pypdf](https://pypdf.readthedocs.io/) |
| LLM Inference | [GROQ API](https://groq.com/) — `llama-3.3-70b-versatile` |
| Web UI | [Streamlit](https://streamlit.io/) |
| Similarity Math | [scikit-learn](https://scikit-learn.org/) (cosine similarity, teaching scripts only) |
| Config | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## Troubleshooting

- **`FileNotFoundError: sample.pdf`** — Scripts 4 through 7 expect a PDF named exactly `sample.pdf` in the project root. Add one before running them.
- **Slow first run** — The embedding model downloads on first use; subsequent runs use the local Hugging Face cache.
- **GROQ authentication errors** — Double-check the key in `.env` (for `app.py`) or directly in `app7_complete_rag_groq.py` (for the console script), and confirm it hasn't expired or hit rate limits.
- **Empty or garbled PDF text** — `pypdf` extracts text layer only; scanned/image-only PDFs will return little or no text. Use an OCR step first if working with scanned documents.

---

## Author

Built by **Amitava Chattopadhyay** — Corporate Trainer specializing in Data Analytics and Full Stack Technology.

- 🌐 Website: [techsutra.co.in](https://techsutra.co.in)
- 💻 GitHub: [@cloudsony999](https://github.com/cloudsony999)

---

*This repository is designed as a teaching resource. Fork it, break it, and rebuild the pipeline your own way — that's the fastest way to actually understand RAG.*