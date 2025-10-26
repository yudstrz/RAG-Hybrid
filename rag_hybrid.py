# ==========================
# 🚀 RAG SYSTEM STREAMLIT FIXED VERSION
# ==========================
import os
import time
import requests
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
from sqlalchemy import create_engine, text
import chromadb
from chromadb.config import Settings

# ==========================
# 🧩 Embedding Model
# ==========================
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# 📚 Base Retriever Class
# ==========================
class BaseRetriever:
    def retrieve(self, query: str) -> List[str]:
        raise NotImplementedError

# ==========================
# 📄 Document Retriever
# ==========================
class DocumentRetriever(BaseRetriever):
    def __init__(self, paths: List[str]):
        self.docs = []
        for p in paths:
            ext = p.lower().split(".")[-1]
            try:
                if ext == "pdf":
                    from PyPDF2 import PdfReader
                    reader = PdfReader(p)
                    self.docs.extend([p.extract_text() for p in reader.pages])
                elif ext in ["doc", "docx"]:
                    import docx2txt
                    self.docs.append(docx2txt.process(p))
                elif ext in ["ppt", "pptx"]:
                    from pptx import Presentation
                    prs = Presentation(p)
                    text_runs = []
                    for slide in prs.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, "text"):
                                text_runs.append(shape.text)
                    self.docs.append(" ".join(text_runs))
                elif ext in ["txt", "md"]:
                    with open(p, "r", encoding="utf-8") as f:
                        self.docs.append(f.read())
            except Exception as e:
                st.warning(f"Gagal memuat dokumen {p}: {e}")

    def retrieve(self, query: str) -> List[str]:
        if not self.docs:
            return []
        query_emb = EMBED_MODEL.encode(query, convert_to_tensor=True)
        docs_emb = EMBED_MODEL.encode(self.docs, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, docs_emb)[0]
        topk = min(3, len(self.docs))
        best_idx = np.argsort(scores)[-topk:][::-1]
        return [self.docs[i] for i in best_idx]

# ==========================
# 🌐 Web Search Retriever
# ==========================
class WebSearchRetriever(BaseRetriever):
    def __init__(self, max_results=3, min_delay=2):
        self.max_results = max_results
        self.min_delay = min_delay
        self.ddgs = DDGS()

    def retrieve(self, query: str) -> List[str]:
        time.sleep(self.min_delay)
        try:
            results = list(self.ddgs.text(query, max_results=self.max_results))
            return [r["body"] for r in results if "body" in r]
        except Exception as e:
            st.warning(f"Gagal melakukan pencarian web: {e}")
            return []

# ==========================
# 🗄️ SQL Retriever
# ==========================
class SQLRetriever(BaseRetriever):
    def __init__(self, connection_str: str, table: str):
        self.engine = create_engine(connection_str)
        self.table = table

    def retrieve(self, query: str) -> List[str]:
        try:
            with self.engine.connect() as conn:
                q = text(f"SELECT * FROM {self.table} WHERE content LIKE :kw LIMIT 3")
                rows = conn.execute(q, {"kw": f"%{query}%"}).fetchall()
                return [str(r) for r in rows]
        except Exception as e:
            st.warning(f"SQL retrieval error: {e}")
            return []

# ==========================
# 🧠 Vector Store Retriever
# ==========================
class VectorStoreRetriever(BaseRetriever):
    def __init__(self, collection_name="rag_collection"):
        try:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            self.collection = self.client.get_or_create_collection(collection_name)
        except Exception as e:
            st.warning(f"Gagal inisialisasi ChromaDB: {e}")
            self.collection = None

    def add(self, docs: List[str]):
        if not self.collection:
            return
        embeddings = EMBED_MODEL.encode(docs).tolist()
        ids = [str(i) for i in range(len(docs))]
        try:
            self.collection.add(ids=ids, embeddings=embeddings, documents=docs)
        except Exception as e:
            st.warning(f"Vector add error: {e}")

    def retrieve(self, query: str) -> List[str]:
        if not self.collection:
            return []
        query_emb = EMBED_MODEL.encode([query]).tolist()
        try:
            results = self.collection.query(query_embeddings=query_emb, n_results=3)
            return results.get("documents", [[]])[0]
        except Exception as e:
            st.warning(f"Vector retrieval error: {e}")
            return []

# ==========================
# ⚙️ RAG System (Multi-Source)
# ==========================
class RAGSystem:
    def __init__(self, retrievers: Dict[str, BaseRetriever], provider="ollama", model="llama3.1"):
        self.retrievers = retrievers
        self.provider = provider
        self.model = model

    def _generate(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._ollama_generate(prompt)
        elif self.provider == "openai":
            return self._openai_generate(prompt)
        elif self.provider == "groq":
            return self._groq_generate(prompt)
        else:
            return f"Model (Simple): {prompt[:200]}..."

    def _ollama_generate(self, prompt: str) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=120)
            return resp.json().get("response", "")
        except Exception as e:
            return f"Ollama error: {e}"

    def _openai_generate(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI error: {e}"

    def _groq_generate(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY", "")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": "llama-3.1-70b-versatile", "messages": [{"role": "user", "content": prompt}]}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Groq error: {e}"

    def ask(self, query: str) -> str:
        contexts = []
        for name, retr in self.retrievers.items():
            try:
                results = retr.retrieve(query)
                contexts.extend(results)
            except Exception as e:
                st.warning(f"Retriever {name} error: {e}")
        context_text = "\n\n".join(contexts)
        prompt = f"Jawab pertanyaan berikut berdasarkan konteks:\n\n{context_text}\n\nPertanyaan: {query}\nJawaban:"
        return self._generate(prompt)

# ==========================
# 🎨 Streamlit UI
# ==========================
st.set_page_config(page_title="🧠 Multi-Source RAG Assistant", layout="wide")
st.title("🧠 Multi-Source RAG Assistant")

st.sidebar.header("⚙️ Settings")
provider = st.sidebar.selectbox("Provider", ["ollama", "openai", "groq", "simple"])
model = st.sidebar.text_input("Model", "llama3.1")
doc_files = st.sidebar.file_uploader("Upload Documents", type=["pdf", "docx", "pptx", "txt"], accept_multiple_files=True)

temp_paths = []
if doc_files:
    for uf in doc_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uf.name) as tmp:
            tmp.write(uf.read())
            temp_paths.append(tmp.name)

source_types = st.sidebar.multiselect("Sources", ["Documents", "Web", "SQL", "Vector"], default=["Documents", "Web"])
retrievers = {}

if "Documents" in source_types and temp_paths:
    retrievers["docs"] = DocumentRetriever(temp_paths)
if "Web" in source_types:
    retrievers["web"] = WebSearchRetriever()
if "SQL" in source_types:
    retrievers["sql"] = SQLRetriever("sqlite:///sample.db", "data")
if "Vector" in source_types:
    vec = VectorStoreRetriever()
    retrievers["vector"] = vec
    if "docs" in retrievers:
        vec.add(retrievers["docs"].docs)

rag = RAGSystem(retrievers, provider=provider, model=model)

query = st.text_area("💬 Masukkan pertanyaan:")
if st.button("🔍 Cari Jawaban"):
    if not query.strip():
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Sedang mencari jawaban..."):
            answer = rag.ask(query)
            st.success("✅ Jawaban ditemukan!")
            st.write(answer)
            st.divider()
            st.caption("🔎 Powered by Multi-Source RAG")

# ==========================
# 🧾 History (optional)
# ==========================
if "history" not in st.session_state:
    st.session_state.history = []
if st.button("💾 Simpan ke Riwayat") and query:
    st.session_state.history.append({"query": query, "provider": provider, "model": model})
if st.session_state.history:
    st.subheader("📜 Riwayat Pertanyaan")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
