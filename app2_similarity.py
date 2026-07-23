# ==============================================
# Program : app2_similarity.py
# Purpose : Compare sentence similarity
# ==============================================

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 60)
print("SENTENCE SIMILARITY USING EMBEDDINGS")
print("=" * 60)

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python is a programming language.",
    "Python is used for coding.",
    "Football is a popular game."
]

embeddings = model.encode(sentences)

print("\nCalculating Similarity...\n")

similarity_matrix = cosine_similarity(embeddings)

print("Similarity Matrix\n")

print(similarity_matrix)

print("\n" + "=" * 60)

print("\nSentence Comparison\n")

for i in range(len(sentences)):
    print(i, ":", sentences[i])

print("\n" + "=" * 60)

print("\nMeaning")

print("Similarity close to 1  --> Highly Similar")

print("Similarity close to 0  --> Not Similar")

'''
Output
Similarity Matrix

[[1.00 0.83 0.17]
 [0.83 1.00 0.20]
 [0.17 0.20 1.00]]

Explanation

Python ↔ Python

0.83

Highly Similar

Football

0.17

Different Topic

'''