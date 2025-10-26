"""
rag_hybrid.py
RAG Hybrid modular skeleton:
 - DocumentRetriever (PDF/DOCX/PPTX)
 - WebRetriever (duckduckgo-search)
 - SQLRetriever (SQLAlchemy)
 - APIClientRetriever (generic HTTP APIs)
 - TranscriptRetriever (Whisper or pre-transcribed text)
 - VectorStoreRetriever (Chroma local example)
 - KnowledgeGraphRetriever (stub)
 - IoTRetriever (stub)
 - UserHistory (simple in-memory)
 - Orchestrator: combine results, call LLM via Ollama HTTP API
"""

import os
import json
import sqlite3
import time
from typing import List, Dict, Any, Optional

# External libs
import requests
from duckduckgo_search import DDGS
from sqlalchemy import create_engine, text
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------
# Helper: LLM client (Ollama HTTP)
# ---------------------------
class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.base = host.rstrip("/")

    def generate(self, model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Calls Ollama local HTTP API (service started by `ollama serve`).
        If you don't run Ollama server, alternative: call CLI via subprocess.
        """
        url = f"{self.base}/api/generate"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Ollama's response structure may vary; common field: 'choices'
        # safe extraction:
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("message", {}).get("content", "").strip()
        # fallback
        return data.get("output", "")

# ---------------------------
# Retriever: Documents (PDF, DOCX, PPTX)
# ---------------------------
class DocumentRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # for local embeddings
        self.embedder = SentenceTransformer(model_name)
        # in-memory store: list of (text, embedding)
        self.store = []

    def load_files(self, file_paths: List[str]):
        texts = []
        for f in file_paths:
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
                # fallback text read
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    texts.append(fh.read())

        # chunking naive: keep as paragraphs
        for t in texts:
            chunks = [p.strip() for p in t.split("\n\n") if p.strip()]
            for c in chunks:
                emb = self.embedder.encode(c)
                self.store.append({"text": c, "emb": emb})

    def retrieve(self, query: str, k: int = 3):
        q_emb = self.embedder.encode(query)
        # simple cosine search
        import numpy as np
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
    def __init__(self):
        pass

    def search(self, query: str, n: int = 3) -> List[str]:
        texts = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                # duckduckgo_search returns dict with "title" and "body"
                body = r.get("body", "")
                title = r.get("title", "")
                snippet = f"{title}\n{body}".strip()
                if snippet:
                    texts.append(snippet)
        return texts

# ---------------------------
# Retriever: SQL (SQLAlchemy)
# ---------------------------
class SQLRetriever:
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def query(self, sql: str, limit: int = 5) -> List[str]:
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            return [str(dict(row)) for row in rows[:limit]]

# ---------------------------
# Retriever: Generic API Client
# ---------------------------
class APIClientRetriever:
    def __init__(self):
        pass

    def fetch_json(self, url: str, params: dict = None) -> str:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        try:
            return json.dumps(r.json())[:5000]  # cap size
        except Exception:
            return r.text[:5000]

# ---------------------------
# Retriever: Transcript (pre-transcribed text)
# ---------------------------
class TranscriptRetriever:
    def __init__(self):
        self.transcripts = []

    def add_transcript(self, text: str, meta: Optional[dict] = None):
        self.transcripts.append({"text": text, "meta": meta or {}})

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        # naive keyword match
        scored = []
        q = query.lower()
        for t in self.transcripts:
            score = q.count(" ".join(q.split()[:3]))  # naive
            # fallback: length overlap
            score += sum(1 for w in q.split() if w in t["text"].lower())
            scored.append((score, t["text"]))
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:k]
        return [s for _, s in top]

# ---------------------------
# Retriever: VectorStore (Chroma example)
# ---------------------------
class VectorStoreRetriever:
    def __init__(self):
        # simple Chroma client
        self.client = chromadb.Client()
        # create a collection if not exists
        try:
            self.col = self.client.get_collection("rag_collection")
        except Exception:
            self.col = self.client.create_collection("rag_collection")

    def add_texts(self, texts: List[str], ids: Optional[List[str]] = None):
        # uses default embedding from chroma if configured; for demo store raw texts as metadata
        ids = ids or [str(time.time()) + "_" + str(i) for i in range(len(texts))]
        self.col.add(ids=ids, documents=texts)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        res = self.col.query(query_texts=[query], n_results=k)
        docs = []
        for rr in res["documents"]:
            docs.extend(rr)
        return docs[:k]

# ---------------------------
# Retriever: Knowledge Graph (stub)
# ---------------------------
class KnowledgeGraphRetriever:
    def __init__(self):
        # real implementation: neo4j / rdflib etc.
        self.triples = []

    def add_triple(self, s, p, o):
        self.triples.append((s, p, o))

    def query(self, sparql_like: str) -> List[str]:
        # naive substring match
        return [str(t) for t in self.triples if sparql_like.lower() in " ".join(map(str, t)).lower()]

# ---------------------------
# Retriever: IoT (stub)
# ---------------------------
class IoTRetriever:
    def __init__(self):
        pass

    def get_latest(self, device_id: str) -> str:
        # stub: integrate with MQTT / REST endpoint
        return f"Latest data for {device_id}: temperature=28.5C, status=ok"

# ---------------------------
# Simple User History (in-memory)
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
        self.sql = None  # initialize if needed
        self.api = APIClientRetriever()
        self.transcripts = TranscriptRetriever()
        self.vector = VectorStoreRetriever()
        self.kg = KnowledgeGraphRetriever()
        self.iot = IoTRetriever()
        self.history = UserHistory()

    def set_sql(self, connection_string: str):
        self.sql = SQLRetriever(connection_string)

    def gather_context(self, query: str, sources: List[str] = ["doc", "web", "sql", "vector", "transcript", "kg", "iot"], k_each: int = 3) -> str:
        contexts = []
        if "doc" in sources:
            contexts += self.doc_retriever.retrieve(query, k=k_each)
        if "web" in sources:
            contexts += self.web.search(query, n=k_each)
        if "sql" in sources and self.sql:
            contexts += self.sql.query(query)  # caution: ensure query is safe or prepare a mapping
        if "vector" in sources:
            contexts += self.vector.retrieve(query, k=k_each)
        if "transcript" in sources:
            contexts += self.transcripts.retrieve(query, k=k_each)
        if "kg" in sources:
            contexts += self.kg.query(query)
        if "iot" in sources:
            contexts.append(self.iot.get_latest(device_id=query if len(query) < 20 else "device123"))
        # dedupe and truncate
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

    def answer(self, user: str, query: str, sources: List[str] = None) -> str:
        sources = sources or ["doc", "web", "vector"]
        ctx = self.gather_context(query, sources)
        prompt = f"""Kamu adalah asisten yang menjawab berdasarkan konteks. Gunakan konteks di bawah ini (jika ada) lalu jawab pertanyaan dengan singkat dan jelas.\n\nKONTEKS:\n{ctx}\n\nPERTANYAAN:\n{query}\n\nJAWABAN:"""
        answer = self.ollama.generate(self.model, prompt, max_tokens=512, temperature=0.0)
        self.history.add(user, query, answer)
        return answer

# ---------------------------
# Example usage (main)
# ---------------------------
def main_example():
    orch = RAGOrchestrator()
    # 1) load docs (if any)
    docs = ["slide_sample.pptx", "report.pdf"]  # replace with real files
    # Only call load_files if files exist
    existing = [f for f in docs if os.path.exists(f)]
    if existing:
        orch.doc_retriever.load_files(existing)

    # 2) add some vector docs (optional)
    orch.vector.add_texts(["This is a company policy about data privacy.",
                           "Quarterly report: revenue up 10%."])

    # 3) set SQL connection (optional)
    # orch.set_sql("sqlite:///mydb.db")

    # 4) add transcript (optional)
    orch.transcripts.add_transcript("Pada rapat hari Senin, dibahas target Q4 dan alokasi anggaran.")

    # 5) ask a question
    q = "Apa kebijakan privasi perusahaan terkait pembagian data?"
    print("Query:", q)
    ans = orch.answer(user="wahyu", query=q, sources=["doc","vector","transcript","web"])
    print("Answer:\n", ans)

if __name__ == "__main__":
    main_example()
