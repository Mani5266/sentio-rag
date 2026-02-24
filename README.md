# Sentio-RAG — Privacy-Preserving Retrieval Augmented Generation

Sentio-RAG is a privacy-preserving RAG system that enables LLM-based question answering over emails while preventing leakage of sensitive information.

Instead of naive redaction, Sentio applies semantic anonymization: identities are replaced with role-aware placeholders, financial values are generalized, dates are abstracted, and epsilon-controlled randomness is introduced to model privacy–utility tradeoffs.

Only masked content is embedded and stored in the vector database.

---

## ✨ Features

- PII detection (Person, Org, Email, Phone, Money, Date)
- Canonical entity resolution (same person → same placeholder)
- Role-aware masking (Vendor / Requester / Approver / Compliance)
- Money bucketing (₹68,200 → tens of thousands of dollars)
- Date generalization (April 12 → mid April)
- ε-controlled randomized anonymization
- Privacy attack simulation (re-identification attempts)
- QA benchmarking (raw vs masked)
- Privacy–utility curve (ε vs accuracy)
- Masked Retrieval-Augmented Generation using ChromaDB
- Streamlit demo UI

---

## 🏗 Architecture

Email  
→ PII Detection  
→ Semantic + Role Anonymization (ε-noise)  
→ Masked Embeddings  
→ ChromaDB  
→ Retriever  
→ LLM QA  

Raw data is never stored in the vector database.

---

## 📂 Project Structure

sentio_backend.py # Core anonymization + privacy engine
app.py # Streamlit demo
rag_index.py # Index masked emails into ChromaDB
rag_query.py # Query masked DB + LLM
benchmark.py # QA utility benchmarking
epsilon_eval.py # Privacy–utility evaluation
chroma_db/ # Persistent masked embeddings



---

## 🚀 How to Run

### Install dependencies

```bash
pip install spacy presidio-analyzer sentence-transformers chromadb streamlit transformers
python -m spacy download en_core_web_lg


Run demo
streamlit run app.py

Build RAG index
python rag_index.py

Query RAG
python rag_query.py

Benchmark utility
python benchmark.py

Evaluate epsilon tradeoff
python epsilon_eval.py

📊 Evaluation

QA accuracy before vs after anonymization
Semantic similarity
ε vs accuracy curve
Adversarial re-identification attempts