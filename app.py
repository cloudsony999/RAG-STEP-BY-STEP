import streamlit as st

from rag_engine import RAGEngine

from dotenv import load_dotenv

import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="PDF RAG using GROQ",
    layout="wide"
)

st.title("📚 PDF Question Answering using RAG")
st.write("Upload a PDF and ask questions.")

# Keep RAG engine alive
if "engine" not in st.session_state:

    st.session_state.engine = RAGEngine(api_key)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    with st.spinner("Reading PDF..."):

        text = st.session_state.engine.load_pdf(uploaded_file)

        st.session_state.engine.create_vector_database(text)

    st.success("PDF Indexed Successfully!")

    st.info(f"Characters Loaded : {len(text)}")

    question = st.text_input("Ask your Question")

    if st.button("Get Answer"):

        if question.strip() == "":

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching..."):

                answer = st.session_state.engine.ask(question)

            st.subheader("Answer")

            st.write(answer)


'''
Step 5: Run
streamlit run app.py
________________________________________
UI
---------------------------------------------------

📚 PDF Question Answering using RAG

Upload PDF

[ Choose File ]

--------------------------------------------

✓ PDF Indexed Successfully

Characters Loaded : 8250

--------------------------------------------

Ask your Question

[____________________________]

        [ Get Answer ]

--------------------------------------------

Answer

Machine Learning is a subset of Artificial
Intelligence...

---------------------------------------------------
________________________________________
How the Application Works
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
Suggested Improvements
This is an excellent beginner version. As your next steps, you can enhance it by adding:
•	Chat history using st.session_state.messages, so users can have a conversation instead of asking one question at a time. 
•	Retrieval transparency, displaying the top retrieved PDF chunks in an expandable section so users understand why the model answered the way it did. 
•	Better chunking, using overlapping chunks (for example, 500 characters with a 100-character overlap) to avoid splitting important information across chunk boundaries. 
•	Support for multiple PDFs, combining documents into a single vector index. 
•	Persistent vector storage, saving the FAISS index to disk so the PDF doesn't need to be re-indexed every time the app starts. 
This version is intentionally kept simple so that it closely matches the console-based RAG application you built earlier while introducing only the Streamlit concepts needed to create a web interface.



'''