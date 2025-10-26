# ==========================
# 🌟 RAG SYSTEM - GEMINI ONLY (Updated)
# ==========================
import os
import time
import requests
import tempfile
import uuid
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional

# Core dependencies
import requests

# Web search & scraping
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# Document loaders
import pypdf
import docx2txt
from pptx import Presentation

# Embeddings and vector store
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LangChain for text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================
# 🎨 Streamlit Page Configuration (MUST BE FIRST STREAMLIT COMMAND)
# ==========================
st.set_page_config(page_title="🤖 Gemini RAG Assistant", layout="wide")

# ==========================
# 🔑 Google Gemini API Key
# ==========================
# For production, use st.secrets for better security
# API_KEY = st.secrets["GEMINI_API_KEY"]
API_KEY = "AIzaSyCR8xgDIv5oYBaDmMyuGGWjqpFi7U8SGA4"  # Ganti dengan key kamu
os.environ["GEMINI_API_KEY"] = API_KEY

# ==========================
# 🧩 Cached Resources
# ==========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_chroma_client():
    # Using in-memory database for this demo. For persistence, use a path.
    return chromadb.Client(Settings(allow_reset=True))

EMBED_MODEL = load_embedding_model()
CHROMA_CLIENT = load_chroma_client()
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# ==========================
# 📄 Document Loader and Splitter
# ==========================
def load_and_split_documents(paths: List[str]) -> List[Dict[str, Any]]:
    docs = []
    for p in paths:
        ext = p.lower().split(".")[-1]
        text_content = ""
        source_name = os.path.basename(p)
        try:
            if ext == "pdf":
                reader = pypdf.PdfReader(p)
                text_content = "\n".join([page.extract_text() or "" for page in reader.pages])
            elif ext in ["doc", "docx"]:
                text_content = docx2txt.process(p)
            elif ext in ["ppt", "pptx"]:
                prs = Presentation(p)
                text_content = "\n".join([shape.text for shape in prs.slides for shape in shape.shapes if hasattr(shape, "text")])
            elif ext in ["txt", "md"]:
                with open(p, "r", encoding="utf-8") as f:
                    text_content = f.read()
            
            if text_content.strip():
                chunks = TEXT_SPLITTER.split_text(text_content)
                for i, chunk in enumerate(chunks):
                    docs.append({
                        "content": chunk,
                        "metadata": {"source": source_name, "chunk": i+1}
                    })
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat {source_name}: {str(e)}")
    return docs

# ==========================
# 🌐 Web Search Retriever (with optional scraping)
# ==========================
def search_and_scrape_web(query: str, max_results: int, scrape_full_page: bool) -> List[Dict[str, Any]]:
    results = []
    try:
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(query, max_results=max_results))
        
        for r in ddgs_results:
            content = r.get("body", "")
            title = r.get("title", "")
            url = r.get("href", "")
            
            if scrape_full_page and url:
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, "lxml")
                    # Extract text from paragraph tags for cleaner content
                    page_content = "\n".join([p.get_text() for p in soup.find_all('p')])
                    if page_content.strip():
                        content = page_content
                except Exception as e:
                    st.warning(f"⚠️ Could not scrape {url}: {e}")

            if content.strip():
                chunks = TEXT_SPLITTER.split_text(content)
                for i, chunk in enumerate(chunks):
                    results.append({
                        "content": chunk,
                        "metadata": {"title": title, "url": url, "source": "web_search", "chunk": i+1}
                    })
    except Exception as e:
        st.warning(f"⚠️ Web search error: {str(e)}")
    return results

# ==========================
# 🧠 Chroma Vector Store
# ==========================
class ChromaVectorStore:
    def __init__(self, collection_name: str, embedding_model):
        self.client = CHROMA_CLIENT
        self.embedding_model = embedding_model
        # Reset collection for a fresh start on each run
        try:
            self.client.delete_collection(name=collection_name)
        except:
            pass
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None # We will provide embeddings manually
        )

    def add_documents(self, docs: List[Dict[str, Any]]):
        if not docs:
            return
        
        texts = [d["content"] for d in docs]
        metadatas = [d["metadata"] for d in docs]
        ids = [str(uuid.uuid4()) for _ in docs]
        
        embeddings = self.embedding_model.encode(texts).tolist()
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query).tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_docs = []
        for i in range(len(result['ids'][0])):
            retrieved_docs.append({
                "content": result['documents'][0][i],
                "metadata": result['metadatas'][0][i],
                "score": 1 - result['distances'][0][i] # Convert distance to similarity score
            })
        return retrieved_docs

# ==========================
# 🤖 Gemini RAG System
# ==========================
class GeminiRAG:
    def __init__(self, vector_store: ChromaVectorStore, model="gemini-1.5-flash"):
        self.vector_store = vector_store
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
        with st.spinner("🔍 Mengambil informasi relevan..."):
            top_results = self.vector_store.retrieve(query, n_results=5)

        if not top_results:
            return {"answer": "Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan Anda.", "contexts": []}

        context_text = "\n\n".join([
            f"[Sumber: {r['metadata'].get('source', 'unknown')} - {r['metadata'].get('title', r['metadata'].get('url', ''))}]\n{r['content']}" 
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
- Jangan menyebutkan "berdasarkan konteks" atau kalimat serupa dalam jawaban

JAWABAN:"""

        with st.spinner("🤖 Sedang generate jawaban dengan Gemini..."):
            answer = self._generate(prompt)
        
        return {"answer": answer, "contexts": top_results}

# ==========================
# 🎨 Streamlit UI
# ==========================
st.title("🤖 Google Gemini RAG Assistant")
st.caption("✨ Menggabungkan dokumen, web, dan vektor dengan Gemini API (ChromaDB + LangChain)")

st.sidebar.header("⚙️ Konfigurasi & Sumber Data")
st.sidebar.success("✅ Gemini API Key aktif")

# --- Data Source Configuration ---
uploaded_files = st.sidebar.file_uploader("📁 Upload dokumen", type=["pdf", "docx", "pptx", "txt", "md"], accept_multiple_files=True)
use_web_search = st.sidebar.checkbox("🌐 Gunakan Pencarian Web", value=True)
scrape_full_page = st.sidebar.checkbox("🕷️ Scraping konten penuh dari halaman web (lebih lambat)", value=True)

# --- Main Logic ---
if st.button("🚀 Inisialisasi Data & Mulai"):
    if not uploaded_files and not use_web_search:
        st.warning("⚠️ Silakan upload dokumen atau aktifkan pencarian web.")
    else:
        with st.spinner("📥 Memproses dan memuat data..."):
            all_docs = []
            
            # 1. Process uploaded documents
            if uploaded_files:
                temp_paths = []
                for uf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                        tmp.write(uf.read())
                        temp_paths.append(tmp.name)
                
                document_chunks = load_and_split_documents(temp_paths)
                all_docs.extend(document_chunks)
                st.success(f"✅ Berhasil memuat {len(document_chunks)} potongan dari {len(uploaded_files)} dokumen.")
                
                # Clean up temp files
                for p in temp_paths:
                    os.remove(p)

            # 2. Process web search
            if use_web_search:
                # Using a default query for initialization, or you can add a text_input here
                web_query = "artificial intelligence trends 2024" 
                web_chunks = search_and_scrape_web(web_query, max_results=3, scrape_full_page=scrape_full_page)
                all_docs.extend(web_chunks)
                st.success(f"✅ Berhasil mengambil {len(web_chunks)} potongan dari web untuk query: '{web_query}'")

            # 3. Initialize Vector Store and add all data
            if all_docs:
                vector_store = ChromaVectorStore(collection_name="rag_collection", embedding_model=EMBED_MODEL)
                vector_store.add_documents(all_docs)
                st.session_state.vector_store = vector_store
                st.success("🎉 Semua data telah dimuat ke dalam Vector Store. Anda bisa mulai bertanya!")
            else:
                st.error("❌ Tidak ada data yang berhasil dimuat.")

# --- Q&A Section ---
if "vector_store" in st.session_state:
    st.divider()
    query = st.text_area("💬 Pertanyaan Anda:", placeholder="Contoh: Apa saja tren AI terkini?")
    if st.button("🔍 Cari Jawaban"):
        if not query.strip():
            st.warning("⚠️ Masukkan pertanyaan terlebih dahulu.")
        else:
            rag = GeminiRAG(st.session_state.vector_store)
            result = rag.ask(query)
            
            st.markdown("### 💡 Jawaban:")
            st.markdown(result["answer"])
            
            with st.expander(f"📚 Lihat {len(result['contexts'])} sumber yang digunakan"):
                for i, ctx in enumerate(result["contexts"], 1):
                    source_info = ctx['metadata'].get('title', ctx['metadata'].get('source', 'Unknown'))
                    st.markdown(f"**Sumber {i}:** {source_info}")
                    st.caption(f"Relevansi: {ctx['score']:.2%}")
                    st.text_area("Konten:", ctx['content'][:500] + "...", height=150, key=f"source_{i}")
                    st.divider()
else:
    st.info("Silakan inisialisasi data terlebih dahulu dengan menekan tombol '🚀 Inisialisasi Data & Mulai' di sidebar.")

st.caption("💡 Powered by Google Gemini + ChromaDB + LangChain")
