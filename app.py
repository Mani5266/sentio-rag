import streamlit as st
from ingest import extract_text
from sentio_universal_backend import sentio_universal
from transformers import pipeline

st.set_page_config(layout="wide")

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_new_tokens=200
)

st.title("🔐 SentioShield — Universal Privacy-Preserving QA")

uploaded = st.file_uploader("Upload Document (PDF / DOCX / TXT / Image)")
url = st.text_input("Or paste webpage URL")

raw = ""

if uploaded:
    with open(uploaded.name, "wb") as f:
        f.write(uploaded.getbuffer())
    raw = extract_text(file_path=uploaded.name)

elif url.strip():
    raw = extract_text(url=url)

else:
    raw = st.text_area("Or paste raw text", height=250)

question = st.text_input("Ask Question (optional — leave empty for summary)")

if st.button("Run"):

    if raw.strip() == "":
        st.warning("Please upload file, URL, or text.")
    else:

        masked, mapping = sentio_universal(raw)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Raw Content")
            st.text_area("", raw, height=400)

        with col2:
            st.subheader("🔒 Masked Content")
            st.text_area("", masked, height=400)

        st.subheader("🧩 Entity Mapping (Original → Placeholder)")
        st.json(mapping)

        if question.strip():
            prompt = f"question: {question} context: {masked}"
        else:
            prompt = f"summarize: {masked}"

        answer = llm(prompt)[0]["generated_text"]

        st.subheader("🤖 LLM Answer")
        st.success(answer)