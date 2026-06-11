import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Fitness RAG Agent",
    page_icon="💪",
    layout="centered",
)

VECTORSTORE_DB = Path("./vectorstore/chroma.sqlite3")


def check_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@st.cache_resource(show_spinner="Building knowledge base from fitness documents… this takes ~1 minute on first run.")
def load_agent():
    if not VECTORSTORE_DB.exists():
        from app.ingest import DocumentIngestion
        ingestion = DocumentIngestion(strategy="recursive")
        ingestion.run()
    from app.agent import RAGAgent
    return RAGAgent()


def format_source(source_meta: dict) -> str:
    raw = source_meta.get("source", "Unknown")
    name = Path(raw).stem.replace("-", " ").replace("_", " ").title()
    page = source_meta.get("page", "N/A")
    return f"**{name}** — page {page}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💪 Fitness RAG Agent")
    st.markdown(
        "Ask anything about **fitness**, **nutrition**, and **sports performance**. "
        "Answers are grounded in the loaded documents — no hallucination."
    )
    st.divider()
    st.subheader("Knowledge Base")
    st.markdown(
        "- High-Energy High-Protein Diet Guide\n"
        "- Exercise & Nutrition Science\n"
        "- Sports Nutrition Fundamentals\n"
        "- Calorie & Protein Education Sheet"
    )
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("💪 Fitness RAG Agent")
st.caption("Grounded answers from your fitness knowledge base · Powered by GPT-4.1-mini")

if not check_api_key():
    st.error(
        "**OPENAI_API_KEY is not set.**  \n"
        "Add it to a `.env` file locally, or set it as an environment variable in Railway."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"- {format_source(s)}")

# Load agent (triggers ingestion on very first run)
agent = load_agent()

# Chat input
if prompt := st.chat_input("Ask about fitness, nutrition, or sports performance…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            result = agent.query(prompt)

        answer = result["answer"]
        sources = result.get("sources", [])

        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=False):
                for s in sources:
                    st.markdown(f"- {format_source(s)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
