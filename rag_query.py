import chromadb
from sentence_transformers import SentenceTransformer
from sentio_backend import generate_answer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(name="sentio")

query = "Who approved the invoice?"

q_emb = embedder.encode(query).tolist()

res = collection.query(
    query_embeddings=[q_emb],
    n_results=2
)

docs = res["documents"][0]

if not docs:
    print("No documents retrieved.")
    exit()

context = "\n".join(docs)

answer = generate_answer(context, query)

print("\nRetrieved Context:\n", context)
print("\nAnswer:\n", answer)