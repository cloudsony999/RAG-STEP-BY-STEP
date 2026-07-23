# ==============================================
# Program : app1_embedding.py
# Purpose : Understanding Sentence Embeddings
# ==============================================

from sentence_transformers import SentenceTransformer

print("=" * 50)
print("LOADING EMBEDDING MODEL...")
print("=" * 50)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nModel Loaded Successfully!")

# Sample sentences
sentences = [
    "Python is easy to learn.",
    "Artificial Intelligence is changing the world.",
    "Cricket is a popular sport."
]

print("\nGenerating Embeddings...\n")

# Generate embeddings
embeddings = model.encode(sentences)

# Display results
for i in range(len(sentences)):
    print("-" * 60)
    print("Sentence :", sentences[i])

    print("\nVector Length :", len(embeddings[i]))

    print("\nFirst 15 Values of the Vector:")
    print(embeddings[i][:15])

print("\nDone!")

'''
Output
==================================================
LOADING EMBEDDING MODEL...
==================================================

Model Loaded Successfully!

Generating Embeddings...

------------------------------------------------------------
Sentence : Python is easy to learn.

Vector Length : 384

First 15 Values:
[0.1245 -0.3456 0.8978 ...]

------------------------------------------------------------
Sentence : Artificial Intelligence is changing the world.

Vector Length : 384

...

Done!

'''