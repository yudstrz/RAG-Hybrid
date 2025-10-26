"""
rag_hybrid_streamlit.py
RAG Hybrid dengan Streamlit UI - Pilih sistem RAG yang ingin digunakan
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st
import requests
from duckduckgo_search import DDGS
from sqlalchemy import create_engine, text
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# ---------------------------
# Helper: LLM client (Ollama HTTP)
# ---------------------------
class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.base = host.rstrip("/")

    def generate(self, model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        try:
            url = f"{self.base}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"

# ---------------------------
# Retriever: Documents (PDF, DOCX, PPTX)
# ---------------------------
class DocumentRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.store = []

    def load_files(self, file_paths: List[str]):
        texts = []
        for f in file_paths:
            try:
                if f.lower().endswith(".pdf"):
                    loader = PyPDFLoader(f)
                    docs = loader.load()
                    for d in docs:
                        texts.append(d.page_content)
                elif f.lower().endswith(".docx"):
                    loader = Docx2txtLoader(f)
                    docs = loader.load()
                    for d in docs:
                        texts.append(d.page_content)
                elif f.lower().endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(f)
                    docs = loader.load()
                    for d in docs:
                        texts.append(d.page_content)
                else:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        texts.append(fh.read())
            except Exception as e:
                st.warning(f"Error loading {f}: {str(e)}")

        for t in texts:
            chunks = [p.strip() for p in t.split("\n\n") if p.strip()]
            for c in chunks:
                emb = self.embedder.encode(c)
                self.store.append({"text": c, "emb": emb})

    def retrieve(self, query: str, k: int = 3):
        if not self.store:
            return []
        q_emb = self.embedder.encode(query)
        sims = []
        for item in self.store:
            v = item["emb"]
            sim = np.dot(q_emb, v) / (np.linalg.norm(q_emb) * np.linalg.norm(v) + 1e-10)
            sims.append(sim)
        idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [self.store[i]["text"] for i in idx]

# ---------------------------
# Retriever: Web (DuckDuckGo)
# ---------------------------
class WebRetriever:
    def search(self, query: str, n: int = 3) -> List[str]:
        texts = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=n):
                    body = r.get("body", "")
                    title = r.get("title", "")
                    snippet = f"{title}\n{body}".strip()
                    if snippet:
                        texts.append(snippet)
        except Exception as e:
            st.warning(f"Web search error: {str(e)}")
        return texts

# ---------------------------
# Retriever: SQL (SQLAlchemy)
# ---------------------------
class SQLRetriever:
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def query(self, sql: str, limit: int = 5) -> List[str]:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                return [str(dict(row._mapping)) for row in rows[:limit]]
        except Exception as e:
            return [f"SQL Error: {str(e)}"]

# ---------------------------
# Retriever: Generic API Client
# ---------------------------
class APIClientRetriever:
    def fetch_json(self, url: str, params: dict = None) -> str:
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return json.dumps(r.json())[:5000]
        except Exception as e:
            return f"API Error: {str(e)}"

# ---------------------------
# Retriever: Transcript
# ---------------------------
class TranscriptRetriever:
    def __init__(self):
        self.transcripts = []

    def add_transcript(self, text: str, meta: Optional[dict] = None):
        self.transcripts.append({"text": text, "meta": meta or {}})

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if not self.transcripts:
            return []
        scored = []
        q = query.lower()
        for t in self.transcripts:
            score = sum(1 for w in q.split() if w in t["text"].lower())
            scored.append((score, t["text"]))
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        return [s for _, s in top]

# ---------------------------
# Retriever: VectorStore (Chroma)
# ---------------------------
class VectorStoreRetriever:
    def __init__(self):
        self.client = chromadb.Client()
        try:
            self.col = self.client.get_collection("rag_collection")
        except Exception:
            self.col = self.client.create_collection("rag_collection")

    def add_texts(self, texts: List[str], ids: Optional[List[str]] = None):
        if not texts:
            return
        ids = ids or [str(time.time()) + "_" + str(i) for i in range(len(texts))]
        self.col.add(ids=ids, documents=texts)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        try:
            res = self.col.query(query_texts=[query], n_results=k)
            docs = []
            for rr in res["documents"]:
                docs.extend(rr)
            return docs[:k]
        except Exception:
            return []

# ---------------------------
# Retriever: Knowledge Graph (stub)
# ---------------------------
class KnowledgeGraphRetriever:
    def __init__(self):
        self.triples = []

    def add_triple(self, s, p, o):
        self.triples.append((s, p, o))

    def query(self, query_text: str) -> List[str]:
        return [str(t) for t in self.triples if query_text.lower() in " ".join(map(str, t)).lower()]

# ---------------------------
# Retriever: IoT (stub)
# ---------------------------
class IoTRetriever:
    def get_latest(self, device_id: str) -> str:
        return f"Latest data for {device_id}: temperature=28.5C, status=ok"

# ---------------------------
# User History
# ---------------------------
class UserHistory:
    def __init__(self):
        self.history = []

    def add(self, user: str, query: str, answer: str):
        self.history.append({"user": user, "query": query, "answer": answer, "ts": time.time()})

    def get_recent(self, user: str, k: int = 5) -> List[str]:
        return [h["query"] + " -> " + h["answer"] for h in self.history if h["user"] == user][-k:]

# ---------------------------
# Orchestrator
# ---------------------------
class RAGOrchestrator:
    def __init__(self, ollama_host="http://localhost:11434", model="llama3.1:8b"):
        self.ollama = OllamaClient(ollama_host)
        self.model = model
        self.doc_retriever = DocumentRetriever()
        self.web = WebRetriever()
        self.sql = None
        self.api = APIClientRetriever()
        self.transcripts = TranscriptRetriever()
        self.vector = VectorStoreRetriever()
        self.kg = KnowledgeGraphRetriever()
        self.iot = IoTRetriever()
        self.history = UserHistory()

    def set_sql(self, connection_string: str):
        self.sql = SQLRetriever(connection_string)

    def gather_context(self, query: str, sources: List[str], k_each: int = 3) -> str:
        contexts = []
        
        if "doc" in sources:
            contexts += self.doc_retriever.retrieve(query, k=k_each)
        if "web" in sources:
            contexts += self.web.search(query, n=k_each)
        if "sql" in sources and self.sql:
            contexts += self.sql.query(query)
        if "vector" in sources:
            contexts += self.vector.retrieve(query, k=k_each)
        if "transcript" in sources:
            contexts += self.transcripts.retrieve(query, k=k_each)
        if "kg" in sources:
            contexts += self.kg.query(query)
        if "iot" in sources:
            contexts.append(self.iot.get_latest(device_id="device123"))
        if "api" in sources:
            # Example API call - customize as needed
            pass
        
        seen = set()
        final_ctx = []
        for c in contexts:
            if not c:
                continue
            s = c.strip()
            if s in seen:
                continue
            seen.add(s)
            final_ctx.append(s)
            if len(final_ctx) >= 10:
                break
        
        return "\n\n---\n\n".join(final_ctx)

    def answer(self, user: str, query: str, sources: List[str]) -> str:
        ctx = self.gather_context(query, sources)
        prompt = f"""Kamu adalah asisten yang menjawab berdasarkan konteks. Gunakan konteks di bawah ini (jika ada) lalu jawab pertanyaan dengan singkat dan jelas.

KONTEKS:
{ctx}

PERTANYAAN:
{query}

JAWABAN:"""
        answer = self.ollama.generate(self.model, prompt, max_tokens=512, temperature=0.0)
        self.history.add(user, query, answer)
        return answer

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(page_title="RAG Hybrid System", page_icon="🤖", layout="wide")
    
    st.title("🤖 RAG Hybrid System")
    st.markdown("Pilih sistem RAG yang ingin digunakan untuk menjawab pertanyaan Anda")
    
    # Initialize session state
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = RAGOrchestrator()
    
    orch = st.session_state.orchestrator
    
    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # Ollama settings
        st.subheader("LLM Settings")
        ollama_host = st.text_input("Ollama Host", value="http://localhost:11434")
        model_name = st.text_input("Model Name", value="llama3.1:8b")
        
        if st.button("Update LLM Config"):
            st.session_state.orchestrator = RAGOrchestrator(ollama_host, model_name)
            st.success("Configuration updated!")
        
        st.divider()
        
        # Document upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, atau PPTX",
            type=["pdf", "docx", "pptx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    temp_paths = []
                    for uf in uploaded_files:
                        temp_path = f"temp_{uf.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uf.read())
                        temp_paths.append(temp_path)
                    
                    orch.doc_retriever.load_files(temp_paths)
                    
                    # Cleanup
                    for tp in temp_paths:
                        os.remove(tp)
                    
                    st.success(f"Processed {len(uploaded_files)} documents!")
        
        st.divider()
        
        # Add vector data
        st.subheader("📊 Add Vector Data")
        vector_text = st.text_area("Enter text to add to vector store")
        if st.button("Add to Vector Store"):
            if vector_text:
                orch.vector.add_texts([vector_text])
                st.success("Added to vector store!")
        
        st.divider()
        
        # Add transcript
        st.subheader("🎤 Add Transcript")
        transcript_text = st.text_area("Enter transcript text")
        if st.button("Add Transcript"):
            if transcript_text:
                orch.transcripts.add_transcript(transcript_text)
                st.success("Transcript added!")
        
        st.divider()
        
        # SQL Configuration
        st.subheader("🗄️ SQL Database")
        sql_conn = st.text_input("SQL Connection String", value="sqlite:///mydb.db")
        if st.button("Connect to Database"):
            try:
                orch.set_sql(sql_conn)
                st.success("Connected to database!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Knowledge Graph
        st.subheader("🕸️ Knowledge Graph")
        col1, col2, col3 = st.columns(3)
        with col1:
            kg_subject = st.text_input("Subject", key="kg_s")
        with col2:
            kg_predicate = st.text_input("Predicate", key="kg_p")
        with col3:
            kg_object = st.text_input("Object", key="kg_o")
        
        if st.button("Add Triple"):
            if kg_subject and kg_predicate and kg_object:
                orch.kg.add_triple(kg_subject, kg_predicate, kg_object)
                st.success("Triple added!")
    
    # Main area: RAG System Selection
    st.header("🔍 Pilih Sistem RAG")
    
    # Create columns for checkboxes
    col1, col2 = st.columns(2)
    
    rag_systems = {
        "doc": "📄 Document Retriever (PDF/DOCX/PPTX)",
        "web": "🌐 Web Search (DuckDuckGo)",
        "sql": "🗄️ SQL Database",
        "api": "🔌 API Client",
        "transcript": "🎤 Transcript Retriever",
        "vector": "📊 Vector Store (Chroma)",
        "kg": "🕸️ Knowledge Graph",
        "iot": "📡 IoT Retriever",
        "history": "📜 User History",
        "all": "✨ All Systems"
    }
    
    selected_systems = []
    
    with col1:
        for key in list(rag_systems.keys())[:5]:
            if st.checkbox(rag_systems[key], key=f"check_{key}"):
                selected_systems.append(key)
    
    with col2:
        for key in list(rag_systems.keys())[5:]:
            if st.checkbox(rag_systems[key], key=f"check_{key}"):
                selected_systems.append(key)
    
    # Handle "all" selection
    if "all" in selected_systems:
        selected_systems = ["doc", "web", "sql", "api", "transcript", "vector", "kg", "iot"]
    
    st.divider()
    
    # Query section
    st.header("💬 Tanya Sistem RAG")
    
    user_name = st.text_input("Nama User", value="wahyu")
    query = st.text_area("Masukkan pertanyaan Anda:", height=100)
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        k_results = st.number_input("Results per source", min_value=1, max_value=10, value=3)
    with col2:
        ask_button = st.button("🚀 Tanya", type="primary", use_container_width=True)
    
    if ask_button and query:
        if not selected_systems:
            st.warning("⚠️ Pilih minimal satu sistem RAG!")
        else:
            with st.spinner("Mengumpulkan konteks dan menjawab..."):
                try:
                    # Show selected systems
                    st.info(f"Menggunakan: {', '.join([rag_systems[s].split(' ')[1] for s in selected_systems])}")
                    
                    # Get answer
                    answer = orch.answer(user_name, query, selected_systems)
                    
                    # Display answer
                    st.subheader("📝 Jawaban:")
                    st.markdown(answer)
                    
                    # Show context sources
                    with st.expander("🔍 Lihat Konteks yang Digunakan"):
                        ctx = orch.gather_context(query, selected_systems, k_each=k_results)
                        st.text(ctx)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # History section
    st.divider()
    st.header("📜 Riwayat Percakapan")
    
    if orch.history.history:
        history_df = []
        for h in orch.history.history[-10:]:  # Show last 10
            history_df.append({
                "User": h["user"],
                "Query": h["query"][:50] + "..." if len(h["query"]) > 50 else h["query"],
                "Answer": h["answer"][:100] + "..." if len(h["answer"]) > 100 else h["answer"],
                "Time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(h["ts"]))
            })
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("Belum ada riwayat percakapan")

if __name__ == "__main__":
    main()
