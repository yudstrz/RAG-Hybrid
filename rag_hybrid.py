# ==========================
# 🌟 RAG SYSTEM - GEMINI ONLY
# =========================
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

# ==========================
# 🔑 Google Gemini API Key
# ==========================
API_KEY = "AIzaSyCR8xgDIv5oYBaDmMyuGGWjqpFi7U8SGA4"  # Ganti dengan key kamu
os.environ["GEMINI_API_KEY"] = API_KEY

# ==========================
# 🧩 Embedding Model
# ==========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

EMBED_MODEL = load_embedding_model()

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
        self.docs, self.metadata = [], []
        for p in paths:
            ext = p.lower().split(".")[-1]
            try:
                if ext == "pdf":
                    from PyPDF2 import PdfReader
                    reader = PdfReader(p)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            self.docs.append(text)
                            self.metadata.append({"source": os.path.basename(p), "page": i+1})
                elif ext in ["doc", "docx"]:
                    import docx2txt
                    text = docx2txt.process(p)
                    if text.strip():
                        self.docs.append(text)
                        self.metadata.append({"source": os.path.basename(p)})
                elif ext in ["ppt", "pptx"]:
                    from pptx import Presentation
                    prs = Presentation(p)
                    for i, slide in enumerate(prs.slides):
                        text_runs = [shape.text for shape in slide.shapes if hasattr(shape, "text")]
                        text = " ".join(text_runs)
                        if text.strip():
                            self.docs.append(text)
                            self.metadata.append({"source": os.path.basename(p), "slide": i+1})
                elif ext in ["txt", "md"]:
                    with open(p, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text.strip():
                            self.docs.append(text)
                            self.metadata.append({"source": os.path.basename(p)})
            except Exception as e:
                st.warning(f"⚠️ Gagal memuat {os.path.basename(p)}: {str(e)}")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
        query_emb = EMBED_MODEL.encode(query, convert_to_tensor=True)
        docs_emb = EMBED_MODEL.encode(self.docs, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, docs_emb)[0].cpu().numpy()
        topk = min(3, len(self.docs))
        best_idx = np.argsort(scores)[-topk:][::-1]
        return [{
            "content": self.docs[idx],
            "score": float(scores[idx]),
            "metadata": self.metadata[idx]
        } for idx in best_idx]

# ==========================
# 🌐 Web Search Retriever
# ==========================
class WebSearchRetriever(BaseRetriever):
    def __init__(self, max_results=5):
        self.max_results = max_results

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            return [{
                "content": r.get("body", ""),
                "metadata": {"title": r.get("title", ""), "url": r.get("href", ""), "source": "web_search"}
            } for r in results]
        except Exception as e:
            st.warning(f"⚠️ Web search error: {str(e)}")
            return []

# ==========================
# 🧠 Vector Store
# ==========================
class HuggingFaceVectorStore(BaseRetriever):
    def __init__(self):
        self.docs, self.embeddings, self.metadata = [], [], []

    def add(self, docs: List[str], metadata: List[Dict] = None):
        if not docs: return
        new_emb = EMBED_MODEL.encode(docs, convert_to_tensor=True)
        self.embeddings.extend(new_emb.cpu().numpy())
        self.docs.extend(docs)
        self.metadata.extend(metadata or [{"source": "unknown"}] * len(docs))

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
        query_emb = EMBED_MODEL.encode(query, convert_to_tensor=True).cpu().numpy()
        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        topk = min(3, len(self.docs))
        best_idx = np.argsort(scores)[-topk:][::-1]
        return [{
            "content": self.docs[idx],
            "score": float(scores[idx]),
            "metadata": self.metadata[idx]
        } for idx in best_idx]

# ==========================
# 🤖 Gemini RAG System
# ==========================
class GeminiRAG:
    def __init__(self, retrievers: Dict[str, BaseRetriever], model="gemini-1.5-flash"):
        self.retrievers = retrievers
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def _generate(self, prompt: str) -> str:
        if not self.api_key:
            return "❌ GEMINI_API_KEY tidak ditemukan."
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"❌ Gemini error: {str(e)}"

    def ask(self, query: str) -> Dict[str, Any]:
        all_results = []
        for name, retr in self.retrievers.items():
            try:
                for r in retr.retrieve(query):
                    r["retriever"] = name
                    all_results.append(r)
            except Exception as e:
                st.warning(f"⚠️ Retriever {name} error: {str(e)}")

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_results = all_results[:5]

        context_text = "\n\n".join([
            f"[Sumber: {r['metadata'].get('source', 'unknown')}]\n{r['content']}" for r in top_results
        ])

        prompt = f"""Berdasarkan konteks berikut, jawab pertanyaan dengan detail dan akurat.
KONTEKS:
{context_text}

PERTANYAAN: {query}

INSTRUKSI:
- Jawab dalam bahasa Indonesia
- Gunakan informasi dari konteks yang relevan
- Jika konteks tidak cukup, sebutkan secara jujur

JAWABAN:"""

        answer = self._generate(prompt)
        return {"answer": answer, "contexts": top_results, "num_sources": len(top_results)}

# ==========================
# 🎨 Streamlit UI
# ==========================
st.set_page_config(page_title="🤖 Gemini RAG Assistant", layout="wide")
st.title("🤖 Google Gemini RAG Assistant")
st.caption("✨ Menggabungkan dokumen, web, dan vektor dengan Gemini API")

st.sidebar.header("⚙️ Konfigurasi")
st.sidebar.success("✅ Gemini API Key aktif")

# Upload dokumen
uploaded_files = st.sidebar.file_uploader("📁 Upload dokumen", type=["pdf", "docx", "pptx", "txt", "md"], accept_multiple_files=True)
source_types = st.sidebar.multiselect("🔍 Data Sources", ["Documents", "Web Search", "Vector Store"], default=["Documents", "Web Search"])

# Proses retriever
retrievers, temp_paths = {}, []
if uploaded_files:
    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
            tmp.write(uf.read())
            temp_paths.append(tmp.name)
if "Documents" in source_types and temp_paths:
    retrievers["docs"] = DocumentRetriever(temp_paths)
if "Web Search" in source_types:
    retrievers["web"] = WebSearchRetriever()
if "Vector Store" in source_types:
    vec = HuggingFaceVectorStore()
    if "docs" in retrievers:
        vec.add(retrievers["docs"].docs, retrievers["docs"].metadata)
    retrievers["vector"] = vec

# Input pertanyaan
query = st.text_area("💬 Pertanyaan Anda:", placeholder="Contoh: Apa isi dokumen saya?")
if st.button("🔍 Cari Jawaban"):
    if not query.strip():
        st.warning("⚠️ Masukkan pertanyaan terlebih dahulu.")
    elif not retrievers:
        st.warning("⚠️ Pilih minimal satu sumber data.")
    else:
        with st.spinner("🤔 Sedang memproses dengan Gemini..."):
            rag = GeminiRAG(retrievers)
            result = rag.ask(query)
            st.markdown("### 💡 Jawaban:")
            st.markdown(result["answer"])
            with st.expander(f"📚 Lihat {result['num_sources']} sumber"):
                for i, ctx in enumerate(result["contexts"], 1):
                    st.markdown(f"**Sumber {i}:** {ctx['metadata'].get('source', 'Unknown')}")
                    st.caption(f"Relevansi: {ctx['score']:.2%}")
                    st.text(ctx['content'][:300] + "...")
                    st.divider()

st.caption("💡 Powered by Google Gemini + Semantic Retrieval")
