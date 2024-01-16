sentences_embeddings

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# Sample data
query = "What is the meaning of life?"
sentences = [
    "The meaning of life is to be happy.",
    "Life is a journey that must be traveled no matter how bad the roads and accommodations.",
    "In three words, I can sum up everything I've learned about life: it goes on.",
    "Life is really simple, but we insist on making it complicated.",
]

# Load a pre-trained Sentence Transformers model (e.g., 'paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Obtain embeddings for the query and sentences
query_embedding = model.encode(query)
sentences_embeddings = model.encode(sentences)

# Plot the embeddings
plt.figure(figsize=(8, 6))

# Plot query point
plt.scatter(query_embedding[0], query_embedding[1], color='red', marker='x', label='Query')

# Plot other sentences
for i, sentence_embedding in enumerate(sentences_embeddings):
    plt.scatter(sentence_embedding[0], sentence_embedding[1], label=f"Sentence {i + 1}")

# Add labels and legend
plt.title('Embeddings of Query and Sentences')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()

# Show the plot
plt.show()
