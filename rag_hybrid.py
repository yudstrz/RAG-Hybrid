# ==========================
# 🚀 RAG SYSTEM - FULL CLOUD API VERSION
# No localhost dependency - Pure cloud services
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

# ==========================
# 🧩 Cloud Embedding Model
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
        self.docs = []
        self.metadata = []
        
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
                        text_runs = []
                        for shape in slide.shapes:
                            if hasattr(shape, "text"):
                                text_runs.append(shape.text)
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
        
        results = []
        for idx in best_idx:
            results.append({
                "content": self.docs[idx],
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            })
        return results

# ==========================
# 🌐 Web Search Retriever
# ==========================
class WebSearchRetriever(BaseRetriever):
    def __init__(self, max_results=5, min_delay=1):
        self.max_results = max_results
        self.min_delay = min_delay

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        time.sleep(self.min_delay)
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "content": r.get("body", ""),
                    "metadata": {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "source": "web_search"
                    }
                })
            return formatted_results
        except Exception as e:
            st.warning(f"⚠️ Web search error: {str(e)}")
            return []

# ==========================
# 🤗 Hugging Face Vector Store
# ==========================
class HuggingFaceVectorStore(BaseRetriever):
    """Vector store using in-memory embeddings (no external DB needed)"""
    def __init__(self):
        self.docs = []
        self.embeddings = []
        self.metadata = []

    def add(self, docs: List[str], metadata: List[Dict] = None):
        if not docs:
            return
        
        new_embeddings = EMBED_MODEL.encode(docs, convert_to_tensor=True)
        self.embeddings.extend(new_embeddings.cpu().numpy())
        self.docs.extend(docs)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"source": "unknown"}] * len(docs))

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if not self.docs:
            return []
        
        query_emb = EMBED_MODEL.encode(query, convert_to_tensor=True).cpu().numpy()
        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        topk = min(3, len(self.docs))
        best_idx = np.argsort(scores)[-topk:][::-1]
        
        results = []
        for idx in best_idx:
            results.append({
                "content": self.docs[idx],
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            })
        return results

# ==========================
# ⚙️ RAG System (Cloud APIs Only)
# ==========================
class RAGSystem:
    def __init__(self, retrievers: Dict[str, BaseRetriever], provider="groq", model="llama-3.1-70b-versatile"):
        self.retrievers = retrievers
        self.provider = provider
        self.model = model

    def _generate(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._openai_generate(prompt)
        elif self.provider == "groq":
            return self._groq_generate(prompt)
        elif self.provider == "anthropic":
            return self._anthropic_generate(prompt)
        elif self.provider == "cohere":
            return self._cohere_generate(prompt)
        else:
            return f"⚠️ Provider '{self.provider}' tidak didukung. Gunakan: openai, groq, anthropic, atau cohere"

    def _openai_generate(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return "❌ OPENAI_API_KEY tidak ditemukan di environment variables"
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ OpenAI error: {str(e)}"

    def _groq_generate(self, prompt: str) -> str:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return "❌ GROQ_API_KEY tidak ditemukan di environment variables"
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model or "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Groq error: {str(e)}"

    def _anthropic_generate(self, prompt: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "❌ ANTHROPIC_API_KEY tidak ditemukan di environment variables"
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model or "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except Exception as e:
            return f"❌ Anthropic error: {str(e)}"

    def _cohere_generate(self, prompt: str) -> str:
        api_key = os.getenv("COHERE_API_KEY", "")
        if not api_key:
            return "❌ COHERE_API_KEY tidak ditemukan di environment variables"
        
        url = "https://api.cohere.ai/v1/chat"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model or "command",
            "message": prompt,
            "temperature": 0.7
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["text"]
        except Exception as e:
            return f"❌ Cohere error: {str(e)}"

    def ask(self, query: str) -> Dict[str, Any]:
        all_results = []
        
        for name, retr in self.retrievers.items():
            try:
                results = retr.retrieve(query)
                for r in results:
                    r["retriever"] = name
                    all_results.append(r)
            except Exception as e:
                st.warning(f"⚠️ Retriever {name} error: {str(e)}")
        
        # Sort by score if available
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Take top 5 contexts
        top_results = all_results[:5]
        
        context_text = "\n\n".join([
            f"[Sumber: {r['metadata'].get('source', 'unknown')}]\n{r['content']}"
            for r in top_results
        ])
        
        prompt = f"""Berdasarkan konteks berikut, jawab pertanyaan dengan detail dan akurat.

KONTEKS:
{context_text}

PERTANYAAN: {query}

INSTRUKSI:
- Jawab dalam bahasa Indonesia
- Gunakan informasi dari konteks yang relevan
- Jika konteks tidak cukup, sebutkan secara jujur
- Berikan jawaban yang terstruktur dan mudah dipahami

JAWABAN:"""
        
        answer = self._generate(prompt)
        
        return {
            "answer": answer,
            "contexts": top_results,
            "num_sources": len(top_results)
        }

# ==========================
# 🎨 Streamlit UI
# ==========================
st.set_page_config(page_title="🧠 Cloud RAG Assistant", layout="wide")

st.title("🧠 Multi-Source RAG Assistant")
st.caption("🌐 100% Cloud API - No Localhost Required")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# API Provider Selection
provider = st.sidebar.selectbox(
    "🤖 AI Provider",
    ["groq", "openai", "anthropic", "cohere"],
    help="Pilih provider AI (pastikan API key sudah diset di environment)"
)

# Model mapping
model_map = {
    "groq": ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
    "cohere": ["command", "command-light"]
}

model = st.sidebar.selectbox("📦 Model", model_map[provider])

# Document Upload
st.sidebar.subheader("📁 Upload Documents")
doc_files = st.sidebar.file_uploader(
    "Upload files",
    type=["pdf", "docx", "pptx", "txt", "md"],
    accept_multiple_files=True,
    help="Mendukung: PDF, Word, PowerPoint, Text"
)

# Source Selection
source_types = st.sidebar.multiselect(
    "🔍 Data Sources",
    ["Documents", "Web Search", "Vector Store"],
    default=["Documents", "Web Search"]
)

# Initialize retrievers
retrievers = {}
temp_paths = []

# Process uploaded documents
if doc_files:
    for uf in doc_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
            tmp.write(uf.read())
            temp_paths.append(tmp.name)

# Setup retrievers
if "Documents" in source_types and temp_paths:
    with st.spinner("📄 Memproses dokumen..."):
        retrievers["docs"] = DocumentRetriever(temp_paths)
        st.sidebar.success(f"✅ {len(retrievers['docs'].docs)} dokumen chunks berhasil dimuat")

if "Web Search" in source_types:
    retrievers["web"] = WebSearchRetriever(max_results=5)
    st.sidebar.info("🌐 Web search ready")

if "Vector Store" in source_types:
    vec_store = HuggingFaceVectorStore()
    if "docs" in retrievers:
        doc_retriever = retrievers["docs"]
        vec_store.add(doc_retriever.docs, doc_retriever.metadata)
        st.sidebar.success(f"🧠 Vector store: {len(vec_store.docs)} chunks")
    retrievers["vector"] = vec_store

# Main interface
st.divider()

# Initialize RAG
rag = RAGSystem(retrievers, provider=provider, model=model)

# Query input
query = st.text_area(
    "💬 Masukkan pertanyaan Anda:",
    height=100,
    placeholder="Contoh: Apa isi dokumen yang saya upload? Atau: Cari informasi terbaru tentang AI..."
)

col1, col2 = st.columns([1, 5])
with col1:
    search_btn = st.button("🔍 Cari Jawaban", type="primary")
with col2:
    clear_btn = st.button("🗑️ Clear")

if clear_btn:
    st.rerun()

# Process query
if search_btn:
    if not query.strip():
        st.warning("⚠️ Masukkan pertanyaan terlebih dahulu")
    elif not retrievers:
        st.warning("⚠️ Pilih minimal satu data source")
    else:
        with st.spinner("🔎 Mencari jawaban..."):
            result = rag.ask(query)
            
            st.success("✅ Jawaban ditemukan!")
            
            # Display answer
            st.markdown("### 💡 Jawaban:")
            st.markdown(result["answer"])
            
            # Display sources
            with st.expander(f"📚 Lihat Sumber ({result['num_sources']} dokumen)"):
                for i, ctx in enumerate(result["contexts"], 1):
                    st.markdown(f"**Sumber {i}:** {ctx['metadata'].get('source', 'Unknown')}")
                    if 'score' in ctx:
                        st.caption(f"Relevance: {ctx['score']:.2%}")
                    st.text(ctx["content"][:300] + "...")
                    st.divider()

# History tracking
if "history" not in st.session_state:
    st.session_state.history = []

if search_btn and query:
    st.session_state.history.append({
        "query": query,
        "provider": provider,
        "model": model,
        "timestamp": pd.Timestamp.now()
    })

# Display history
if st.session_state.history:
    with st.expander("📜 Riwayat Pertanyaan"):
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

# Footer
st.divider()
st.caption("🔐 Pastikan API keys sudah diset di environment variables (GROQ_API_KEY, OPENAI_API_KEY, dll)")
st.caption("💡 Powered by Multi-Source RAG with Cloud APIs")
