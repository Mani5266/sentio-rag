import chromadb
from sentio_backend import sentio_single
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent client (this creates folder)
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(name="sentio")

emails = [
    "Priya from SecureStack requested approval for ₹68,200.",
    "John from Alpha Corp submitted the invoice.",
    "Daniel from Apex Compliance signed off."
]

for i, mail in enumerate(emails):

    masked = sentio_single(mail)["masked"]
    emb = embedder.encode(masked).tolist()

    collection.add(
        documents=[masked],
        embeddings=[emb],
        ids=[str(i)]
    )

print("Indexed masked emails.")