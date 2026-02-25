# 🔐 SentioShield — Universal Privacy-Preserving Document QA

SentioShield is a privacy-first document intelligence system that anonymizes sensitive information before sending content to Large Language Models (LLMs).

Unlike traditional DP-RAG systems that perturb entire documents, SentioShield performs **span-level anonymization**, masking only high-risk regions while preserving useful context.

Supports: PDFs, DOCX, TXT, Images (OCR), Web URLs, Raw Text.

---

## 🚀 Features

- Layout-aware PDF ingestion (Unstructured)
- Transformer NER (BERT)
- Indian name support with fallback regex
- Person / Organization masking
- Student ID masking
- Email & Phone masking
- Money bucketing (₹42,750 → tens of thousands)
- Date generalization (April 12 → mid April)
- Entity mapping transparency
- LLM Question Answering on anonymized text
- Summarization

---

## 🧠 Architecture

Document → Layout Parser → Transformer NER → Span Detection → Anonymizer → Masked Text → LLM QA

---

## 🧩 Tech Stack

Python, Streamlit  
Transformers (dslim/bert-base-NER, FLAN-T5)  
Unstructured (PDF parsing)  
SentenceTransformers  
Presidio  
scikit-learn  
pytesseract  

---

## 📂 Structure


ActualProject/
├── app.py
├── ingest.py
├── sentio_universal_backend.py
├── sentio_backend.py
├── rag_index.py
├── rag_query.py
├── benchmark.py
├── epsilon_eval.py
└── README.md


---

## ⚙ Installation

```bash
pip install streamlit transformers sentence-transformers presidio-analyzer scikit-learn
pip install "unstructured[pdf]" pdfminer.six pillow pytesseract beautifulsoup4 requests

Windows: install Tesseract OCR separately.

▶ Run
streamlit run app.py
🧪 Sample Input
1. J.Mani – 23071A6724
2. T.Venkat Vishnu – 23071A6761

Reviewed by Apex Compliance Pvt Ltd on April 12.
Contact: venkat.vishnu@gmail.com or +91 9988776655
Cost: ₹42,750
Deployment: March 25
🔒 Masked Output
1. [Person_1] – [StudentID_1]
2. [Person_2] – [StudentID_2]

Reviewed by [Org_1] on mid April.
Contact: [Email_1] or [Phone_1]
Cost: tens of thousands
Deployment: late March
🧩 Mapping
{
  "J Mani": "Person_1",
  "23071A6724": "StudentID_1",
  "Apex Compliance Pvt Ltd": "Org_1",
  "venkat.vishnu@gmail.com": "Email_1",
  "+91 9988776655": "Phone_1"
}
❓ Sample Questions

Who reviewed the project?

What is deployment timeline?

How much is the cost?