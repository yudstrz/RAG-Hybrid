# ==========================
# 🌟 RAG SYSTEM - GEMINI (Improved Version)
# ==========================
import os
import requests
import tempfile
import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st

# Core dependencies
import numpy as np

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
# 📝 Logging Configuration
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================
# 🎨 Streamlit Page Configuration
# ==========================
st.set_page_config(
    page_title="🤖 Gemini RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# 🔑 API Key Management (SECURE)
# ==========================
def get_api_key() -> Optional[str]:
    """Ambil API key dengan prioritas: secrets > env > user input"""
    # Priority 1: Streamlit secrets (untuk production)
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    
    # Priority 2: Environment variable
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    
    # Priority 3: User input (fallback)
    if "api_key_input" not in st.session_state:
        st.session_state.api_key_input = ""
    
    api_key = st.sidebar.text_input(
        "🔑 Masukkan Gemini API Key:",
        type="password",
        value=st.session_state.api_key_input,
        help="Dapatkan API key gratis di: https://ai.google.dev/"
    )
    
    if api_key:
        st.session_state.api_key_input = api_key
        return api_key
    
    return None

API_KEY = get_api_key()

# ==========================
# 🧩 Cached Resources
# ==========================
@st.cache_resource
def load_embedding_model():
    """Load embedding model dengan error handling"""
    try:
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        st.error(f"❌ Gagal load embedding model: {e}")
        return None

@st.cache_resource
def load_chroma_client():
    """Initialize ChromaDB client"""
    try:
        logger.info("Initializing ChromaDB...")
        client = chromadb.Client(Settings(allow_reset=True))
        logger.info("ChromaDB initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        st.error(f"❌ Gagal initialize ChromaDB: {e}")
        return None

EMBED_MODEL = load_embedding_model()
CHROMA_CLIENT = load_chroma_client()
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ==========================
# 📊 Session State Initialization
# ==========================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "initialized": False,
        "vector_store": None,
        "chat_history": [],
        "processing_stats": {},
        "last_query_time": None,
        "query_count": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==========================
# 🛡️ Rate Limiting
# ==========================
def check_rate_limit(max_queries_per_minute: int = 10) -> bool:
    """Simple rate limiting based on query count"""
    current_time = datetime.now()
    
    if st.session_state.last_query_time:
        time_diff = (current_time - st.session_state.last_query_time).total_seconds()
        
        # Reset counter setelah 1 menit
        if time_diff > 60:
            st.session_state.query_count = 0
            st.session_state.last_query_time = current_time
        
        # Check limit
        if st.session_state.query_count >= max_queries_per_minute:
            return False
    
    st.session_state.query_count += 1
    st.session_state.last_query_time = current_time
    return True

# ==========================
# 📄 Document Loading Functions
# ==========================
def load_and_split_documents(paths: List[str]) -> List[Dict[str, Any]]:
    """Load dan split documents dengan improved error handling"""
    docs = []
    stats = {"success": 0, "failed": 0, "total_chunks": 0}
    
    for p in paths:
        ext = p.lower().split(".")[-1]
        text_content = ""
        source_name = os.path.basename(p)
        
        try:
            # Load berdasarkan tipe file
            if ext == "pdf":
                reader = pypdf.PdfReader(p)
                text_content = "\n".join([
                    page.extract_text() or "" for page in reader.pages
                ])
            elif ext in ["doc", "docx"]:
                text_content = docx2txt.process(p)
            elif ext in ["ppt", "pptx"]:
                prs = Presentation(p)
                text_content = "\n".join([
                    shape.text for slide in prs.slides 
                    for shape in slide.shapes 
                    if hasattr(shape, "text")
                ])
            elif ext in ["txt", "md"]:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
            else:
                logger.warning(f"Unsupported file type: {ext}")
                continue
            
            # Split text ke chunks
            if text_content.strip():
                chunks = TEXT_SPLITTER.split_text(text_content)
                for i, chunk in enumerate(chunks):
                    docs.append({
                        "content": chunk,
                        "metadata": {
                            "source": source_name,
                            "chunk": i + 1,
                            "total_chunks": len(chunks),
                            "type": "document"
                        }
                    })
                
                stats["success"] += 1
                stats["total_chunks"] += len(chunks)
                logger.info(f"Loaded {source_name}: {len(chunks)} chunks")
            else:
                logger.warning(f"Empty content in {source_name}")
                stats["failed"] += 1
                
        except Exception as e:
            logger.error(f"Failed to load {source_name}: {e}")
            st.warning(f"⚠️ Gagal memuat {source_name}: {str(e)}")
            stats["failed"] += 1
    
    st.session_state.processing_stats["documents"] = stats
    return docs

# ==========================
# 🌐 Web Search Functions
# ==========================
def search_and_scrape_web(
    query: str,
    max_results: int = 3,
    scrape_full_page: bool = True
) -> List[Dict[str, Any]]:
    """Search dan scrape web dengan improved error handling"""
    results = []
    stats = {"success": 0, "failed": 0, "total_chunks": 0}
    
    try:
        # Search dengan DuckDuckGo
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(query, max_results=max_results))
        
        logger.info(f"Found {len(ddgs_results)} search results")
        
        for idx, r in enumerate(ddgs_results, 1):
            content = r.get("body", "")
            title = r.get("title", "Untitled")
            url = r.get("href", "")
            
            # Scrape full page jika diminta
            if scrape_full_page and url:
                try:
                    with st.spinner(f"🕷️ Scraping {idx}/{len(ddgs_results)}: {title[:50]}..."):
                        response = requests.get(
                            url,
                            timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.content, "lxml")
                        
                        # Hapus script dan style tags
                        for tag in soup(["script", "style", "nav", "footer"]):
                            tag.decompose()
                        
                        # Extract paragraphs
                        paragraphs = soup.find_all(['p', 'article', 'section'])
                        page_content = "\n".join([
                            p.get_text(strip=True) for p in paragraphs
                        ])
                        
                        if page_content.strip() and len(page_content) > len(content):
                            content = page_content
                            logger.info(f"Scraped {len(page_content)} chars from {url}")
                        
                        stats["success"] += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    stats["failed"] += 1
            
            # Split content ke chunks
            if content.strip():
                chunks = TEXT_SPLITTER.split_text(content)
                for i, chunk in enumerate(chunks):
                    results.append({
                        "content": chunk,
                        "metadata": {
                            "title": title,
                            "url": url,
                            "source": "web_search",
                            "chunk": i + 1,
                            "total_chunks": len(chunks),
                            "type": "web"
                        }
                    })
                
                stats["total_chunks"] += len(chunks)
            
    except Exception as e:
        logger.error(f"Web search error: {e}")
        st.error(f"❌ Error pencarian web: {str(e)}")
    
    st.session_state.processing_stats["web"] = stats
    return results

# ==========================
# 🧠 ChromaDB Vector Store Class
# ==========================
class ChromaVectorStore:
    """Enhanced ChromaDB vector store dengan caching dan persistence"""
    
    def __init__(self, collection_name: str, embedding_model):
        self.client = CHROMA_CLIENT
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Reset collection untuk data baru
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted old collection: {collection_name}")
        except Exception:
            pass
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )
        logger.info(f"Created collection: {collection_name}")
    
    def add_documents(self, docs: List[Dict[str, Any]]) -> int:
        """Add documents dengan batch processing"""
        if not docs:
            return 0
        
        batch_size = 100
        total_added = 0
        
        try:
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                
                texts = [d["content"] for d in batch]
                metadatas = [d["metadata"] for d in batch]
                ids = [str(uuid.uuid4()) for _ in batch]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=False
                ).tolist()
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                
                total_added += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} docs")
            
            return total_added
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents dengan optional filtering"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                where=filter_metadata
            )
            
            retrieved_docs = []
            for i in range(len(result['ids'][0])):
                distance = result['distances'][0][i]
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                
                retrieved_docs.append({
                    "content": result['documents'][0][i],
                    "metadata": result['metadatas'][0][i],
                    "score": similarity,
                    "distance": distance
                })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception:
            return {"total_documents": 0, "collection_name": self.collection_name}

# ==========================
# 🤖 Gemini RAG System Class
# ==========================
class GeminiRAG:
    """Enhanced RAG system dengan conversation history"""
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        api_key: str,
        model: str = "gemini-1.5-flash"
    ):
        self.vector_store = vector_store
        self.model = model
        self.api_key = api_key
    
    def _generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response dari Gemini API"""
        if not self.api_key:
            return "❌ API Key tidak tersedia. Silakan masukkan API key di sidebar."
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            logger.info(f"Generated response: {len(answer)} chars")
            return answer
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"API Error {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f": {error_detail.get('error', {}).get('message', 'Unknown error')}"
            except:
                pass
            logger.error(error_msg)
            return f"❌ {error_msg}"
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"❌ Error: {str(e)}"
    
    def ask(
        self,
        query: str,
        n_results: int = 5,
        use_history: bool = True
    ) -> Dict[str, Any]:
        """Ask question dengan RAG"""
        
        # Retrieve relevant documents
        with st.spinner("🔍 Mencari informasi relevan..."):
            top_results = self.vector_store.retrieve(query, n_results=n_results)
        
        if not top_results:
            return {
                "answer": "Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan Anda. Coba tambahkan lebih banyak sumber data.",
                "contexts": [],
                "metadata": {"sources_used": 0}
            }
        
        # Build context
        context_parts = []
        for idx, r in enumerate(top_results, 1):
            source = r['metadata'].get('source', 'Unknown')
            title = r['metadata'].get('title', '')
            url = r['metadata'].get('url', '')
            
            source_info = f"{source}"
            if title:
                source_info = f"{title}"
            if url:
                source_info += f" ({url})"
            
            context_parts.append(
                f"[Sumber {idx}: {source_info}]\n{r['content']}"
            )
        
        context_text = "\n\n".join(context_parts)
        
        # Build conversation history context
        history_context = ""
        if use_history and st.session_state.chat_history:
            recent_history = st.session_state.chat_history[-3:]  # Last 3 exchanges
            history_context = "\n\nRiwayat Percakapan:\n" + "\n".join([
                f"Q: {h['query']}\nA: {h['answer'][:200]}..."
                for h in recent_history
            ])
        
        # Generate prompt
        prompt = f"""Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan.

KONTEKS INFORMASI:
{context_text}
{history_context}

PERTANYAAN: {query}

INSTRUKSI:
- Jawab dalam bahasa Indonesia dengan jelas dan terstruktur
- Gunakan informasi dari konteks yang relevan
- Jika konteks tidak cukup untuk menjawab, jelaskan keterbatasannya
- Berikan jawaban yang informatif dan membantu
- Jangan menyebutkan "berdasarkan konteks" atau kalimat meta serupa

JAWABAN:"""
        
        # Generate response
        with st.spinner("🤖 Menghasilkan jawaban..."):
            answer = self._generate(prompt)
        
        # Save to history
        chat_entry = {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "sources_used": len(top_results)
        }
        st.session_state.chat_history.append(chat_entry)
        
        return {
            "answer": answer,
            "contexts": top_results,
            "metadata": {
                "sources_used": len(top_results),
                "timestamp": chat_entry["timestamp"]
            }
        }

# ==========================
# 🎨 Streamlit UI
# ==========================
def main():
    st.title("🤖 Google Gemini RAG Assistant")
    st.caption("✨ Sistem RAG yang ditingkatkan dengan multiple data sources dan conversation memory")
    
    # Check API Key
    if not API_KEY:
        st.warning("⚠️ Silakan masukkan Gemini API Key di sidebar untuk melanjutkan.")
        st.info("💡 Dapatkan API key gratis di: https://ai.google.dev/")
        return
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Konfigurasi Sumber Data")
    
    # --- Document Upload ---
    use_documents = st.sidebar.checkbox("📄 Gunakan Dokumen", value=True)
    uploaded_files = None
    if use_documents:
        uploaded_files = st.sidebar.file_uploader(
            "📁 Upload dokumen",
            type=["pdf", "docx", "pptx", "txt", "md"],
            accept_multiple_files=True,
            key="doc_uploader",
            help="Support: PDF, DOCX, PPTX, TXT, MD"
        )
    
    # --- Web Search ---
    use_web_search = st.sidebar.checkbox("🌐 Gunakan Pencarian Web", value=False)
    web_query = ""
    scrape_full_page = False
    if use_web_search:
        web_query = st.sidebar.text_input(
            "🔎 Query pencarian web:",
            placeholder="Contoh: artificial intelligence trends 2024"
        )
        scrape_full_page = st.sidebar.checkbox(
            "🕷️ Scrape konten lengkap",
            value=True,
            help="Lebih lambat tapi lebih detail"
        )
    
    st.sidebar.markdown("---")
    
    # --- Advanced Settings ---
    with st.sidebar.expander("⚙️ Pengaturan Lanjutan"):
        n_results = st.slider(
            "Jumlah dokumen yang diambil",
            min_value=3,
            max_value=10,
            value=5,
            help="Lebih banyak = lebih komprehensif tapi lebih lambat"
        )
        
        use_history = st.checkbox(
            "Gunakan riwayat percakapan",
            value=True,
            help="AI akan mengingat 3 pertanyaan terakhir"
        )
    
    # --- Action Buttons ---
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        process_btn = st.button("🚀 Proses Data", type="primary", use_container_width=True)
    
    with col2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)
    
    # --- Process Data ---
    if process_btn:
        all_docs = []
        st.session_state.processing_stats = {}
        
        # Process Documents
        if use_documents and uploaded_files:
            with st.spinner("📥 Memproses dokumen..."):
                temp_paths = []
                for uf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uf.name}") as tmp:
                        tmp.write(uf.read())
                        temp_paths.append(tmp.name)
                
                document_chunks = load_and_split_documents(temp_paths)
                all_docs.extend(document_chunks)
                
                # Cleanup temp files
                for p in temp_paths:
                    try:
                        os.remove(p)
                    except:
                        pass
                
                if document_chunks:
                    st.success(
                        f"✅ Berhasil memproses {len(uploaded_files)} dokumen → "
                        f"{len(document_chunks)} chunks"
                    )
        
        # Process Web Search
        if use_web_search and web_query.strip():
            with st.spinner("🌐 Mencari di web..."):
                web_chunks = search_and_scrape_web(
                    web_query,
                    max_results=3,
                    scrape_full_page=scrape_full_page
                )
                all_docs.extend(web_chunks)
                
                if web_chunks:
                    st.success(f"✅ Berhasil mengambil {len(web_chunks)} chunks dari web")
        
        # Initialize Vector Store
        if all_docs:
            with st.spinner("🧠 Menyimpan ke vector database..."):
                try:
                    vector_store = ChromaVectorStore(
                        collection_name="rag_collection",
                        embedding_model=EMBED_MODEL
                    )
                    added = vector_store.add_documents(all_docs)
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.initialized = True
                    
                    # Show stats
                    stats = vector_store.get_stats()
                    st.success(
                        f"🎉 Sistem RAG siap! Total dokumen: {stats['total_documents']}"
                    )
                    
                    # Show processing stats
                    if st.session_state.processing_stats:
                        with st.expander("📊 Detail Pemrosesan"):
                            st.json(st.session_state.processing_stats)
                    
                except Exception as e:
                    logger.error(f"Initialization error: {e}")
                    st.error(f"❌ Gagal inisialisasi: {str(e)}")
        else:
            st.error("❌ Tidak ada data yang berhasil diproses. Periksa konfigurasi Anda.")
    
    # --- Reset Session ---
    if reset_btn:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # --- Display Stats ---
    if st.session_state.initialized:
        stats = st.session_state.vector_store.get_stats()
        st.sidebar.success(f"📚 {stats['total_documents']} dokumen tersimpan")
        st.sidebar.info(f"💬 {len(st.session_state.chat_history)} percakapan")
    
    st.divider()
    
    # ==========================
    # 💬 Q&A Section
    # ==========================
    if st.session_state.initialized:
        st.header("💬 Tanya Jawab")
        
        # Display chat history
        if st.session_state.chat_history:
            with st.expander(f"📜 Riwayat Chat ({len(st.session_state.chat_history)} pertanyaan)", expanded=False):
                for idx, entry in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
                    st.markdown(f"**Q{idx}:** {entry['query']}")
                    st.markdown(f"**A{idx}:** {entry['answer'][:200]}...")
                    st.caption(f"⏰ {entry['timestamp']} | 📚 {entry['sources_used']} sumber")
                    st.divider()
        
        # Query input
        query = st.text_area(
            "Masukkan pertanyaan Anda:",
            placeholder="Contoh: Jelaskan tren AI terkini berdasarkan sumber yang tersedia",
            height=100
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            ask_btn = st.button("🔍 Cari Jawaban", type="primary", use_container_width=True)
        
        with col2:
            if st.button("📋 Copy Hasil", use_container_width=True):
                if st.session_state.chat_history:
                    last_qa = st.session_state.chat_history[-1]
                    st.code(f"Q: {last_qa['query']}\n\nA: {last_qa['answer']}")
        
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Process query
        if ask_btn:
            if not query.strip():
                st.warning("⚠️ Masukkan pertanyaan terlebih dahulu.")
            elif not check_rate_limit():
                st.error("❌ Terlalu banyak request. Tunggu 1 menit.")
            else:
                try:
                    rag = GeminiRAG(
                        vector_store=st.session_state.vector_store,
                        api_key=API_KEY
                    )
                    
                    result = rag.ask(
                        query,
                        n_results=n_results,
                        use_history=use_history
                    )
                    
                    # Display answer
                    st.markdown("### 💡 Jawaban:")
                    st.markdown(result["answer"])
                    
                    # Display metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📚 Sumber Digunakan", result["metadata"]["sources_used"])
                    with col2:
                        st.metric("⏰ Timestamp", result["metadata"]["timestamp"].split("T")[1][:8])
                    
                    # Display sources
                    if result["contexts"]:
                        with st.expander(f"📚 Lihat {len(result['contexts'])} Sumber yang Digunakan"):
                            for i, ctx in enumerate(result["contexts"], 1):
                                source = ctx['metadata'].get('source', 'Unknown')
                                title = ctx['metadata'].get('title', '')
                                url = ctx['metadata'].get('url', '')
                                
                                st.markdown(f"### Sumber {i}")
                                if title:
                                    st.markdown(f"**📰 {title}**")
                                if url:
                                    st.markdown(f"🔗 [{url}]({url})")
                                else:
                                    st.markdown(f"**📄 {source}**")
                                
                                st.caption(
                                    f"✨ Relevansi: {ctx['score']:.1%} | "
                                    f"📝 Chunk {ctx['metadata'].get('chunk', '?')}/"
                                    f"{ctx['metadata'].get('total_chunks', '?')}"
                                )
                                
                                st.text_area(
                                    "Konten:",
                                    ctx['content'][:500] + ("..." if len(ctx['content']) > 500 else ""),
                                    height=150,
                                    key=f"ctx_{i}",
                                    disabled=True
                                )
                                st.divider()
                
                except Exception as e:
                    logger.error(f"Query processing error: {e}")
                    st.error(f"❌ Terjadi kesalahan: {str(e)}")
    
    else:
        # Initial state - no data processed yet
        st.info("👈 Silakan konfigurasikan sumber data di sidebar dan tekan **'Proses Data'** untuk memulai.")
        
        # Show feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📄 Multi-Format Documents
            - PDF, DOCX, PPTX
            - TXT, Markdown
            - Batch upload support
            """)
        
        with col2:
            st.markdown("""
            ### 🌐 Web Integration
            - Real-time web search
            - Full page scraping
            - Smart content extraction
            """)
        
        with col3:
            st.markdown("""
            ### 🧠 Smart Features
            - Conversation memory
            - Context-aware answers
            - Source attribution
            """)
    
    # ==========================
    # 📊 Footer & Info
    # ==========================
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("💡 **Powered by:**")
        st.caption("Google Gemini 1.5 Flash")
    
    with col2:
        st.caption("🗃️ **Vector Store:**")
        st.caption("ChromaDB + Sentence Transformers")
    
    with col3:
        st.caption("🔧 **Framework:**")
        st.caption("Streamlit + LangChain")
    
    # Debug info (only in development)
    if st.sidebar.checkbox("🐛 Debug Mode", value=False):
        with st.expander("🔍 Debug Information"):
            st.json({
                "session_state": {
                    "initialized": st.session_state.initialized,
                    "chat_history_count": len(st.session_state.chat_history),
                    "query_count": st.session_state.query_count,
                    "has_vector_store": st.session_state.vector_store is not None
                },
                "models": {
                    "embedding_model_loaded": EMBED_MODEL is not None,
                    "chroma_client_loaded": CHROMA_CLIENT is not None
                },
                "api_key_available": bool(API_KEY)
            })

# ==========================
# 🚀 Run Application
# ==========================
if __name__ == "__main__":
    main()
