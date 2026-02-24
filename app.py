import streamlit as st
from sentio_backend import sentio_single

st.set_page_config(layout="wide")

st.title("🔐 Sentio — Privacy Preserving Email QA")

email = st.text_area("Paste Email", height=250)
question = st.text_input("Ask Question (optional — leave empty for summary)")

if st.button("Run"):

    if email.strip() == "":
        st.warning("Please provide email text.")
    else:

        result = sentio_single(email, question if question.strip() else None)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Raw Email")
            st.text_area("", result["raw"], height=300)

        with col2:
            st.subheader("Masked Email")
            st.text_area("", result["masked"], height=300)

        st.subheader("LLM Output")
        st.success(result["output"])

        m1, m2 = st.columns(2)

        m1.metric("PII Survival Rate", round(result["pii_survival_rate"], 3))
        m2.metric("Semantic Similarity", round(result["semantic_similarity"], 3))

        with st.expander("Placeholder Mapping"):
            st.json(result["placeholders"])