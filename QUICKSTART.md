# 🚀 Quick Start Guide

Panduan cepat untuk menjalankan Gemini RAG Assistant dalam 5 menit!

## ⚡ Instalasi Express

### Linux/Mac

```bash
# 1. Clone repository
git clone <your-repo-url>
cd gemini-rag-assistant

# 2. Jalankan setup script
chmod +x setup.sh
./setup.sh

# 3. Edit API key
nano .streamlit/secrets.toml
# Ganti "your-gemini-api-key-here" dengan API key Anda

# 4. Run aplikasi
streamlit run app.py
```

### Windows

```bash
# 1. Clone repository
git clone <your-repo-url>
cd gemini-rag-assistant

# 2. Jalankan setup script
setup.bat

# 3. Edit API key
notepad .streamlit\secrets.toml
# Ganti "your-gemini-api-key-here" dengan API key Anda

# 4. Run aplikasi
streamlit run app.py
```

## 🔑 Mendapatkan API Key

1. Kunjungi: https://ai.google.dev/
2. Klik **"Get API Key"**
3. Sign in dengan akun Google
4. Klik **"Create API Key"**
5. Copy API key yang generated
6. Paste ke `.streamlit/secrets.toml`

## 🎯 Penggunaan Pertama

### Skenario 1: Upload Dokumen

```
1. ✅ Centang "📄 Gunakan Dokumen"
2. 📁 Upload file PDF/DOCX/PPTX
3. 🚀 Klik "Proses Data"
4. 💬 Tanyakan: "Apa isi dari dokumen yang saya upload?"
```

### Skenario 2: Web Search

```
1. ✅ Centang "🌐 Gunakan Pencarian Web"
2. 🔎 Masukkan: "artificial intelligence trends 2024"
3. 🚀 Klik "Proses Data"
4. 💬 Tanyakan: "Apa tren AI terbaru menurut hasil pencarian?"
```

### Skenario 3: Kombinasi

```
1. ✅ Centang keduanya
2. 📁 Upload dokumen tentang AI
3. 🔎 Cari: "AI developments 2024"
4. 🚀 Klik "Proses Data"
5. 💬 Tanyakan: "Bandingkan informasi dari dokumen dengan hasil pencarian web"
```

## 🔧 Troubleshooting Cepat

### Error: ModuleNotFoundError

```bash
# Pastikan virtual environment aktif
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install ulang dependencies
pip install -r requirements.txt
```

### Error: API Key Invalid

```bash
# Cek file secrets.toml
cat .streamlit/secrets.toml

# Pastikan formatnya benar:
GEMINI_API_KEY = "AIza..."  # Dengan quotes
```

### Error: Port Already in Use

```bash
# Gunakan port lain
streamlit run app.py --server.port 8502
```

### Aplikasi Lambat

```python
# Edit app.py, kurangi chunk size:
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Dari 1000
    chunk_overlap=50   # Dari 150
)
```

## 📊 Tips Performa

### Untuk Response Cepat
- Gunakan `gemini-1.5-flash` (default)
- Matikan "Scrape konten lengkap"
- Set "Jumlah dokumen" ke 3-5

### Untuk Akurasi Maksimal
- Ganti ke `gemini-1.5-pro`
- Aktifkan "Scrape konten lengkap"
- Set "Jumlah dokumen" ke 7-10
- Aktifkan "Gunakan riwayat percakapan"

## 🎓 Contoh Pertanyaan

### Untuk Dokumen
```
- "Rangkum poin-poin utama dari dokumen"
- "Apa kesimpulan yang dapat diambil?"
- "Jelaskan konsep X yang disebutkan dalam dokumen"
- "Bandingkan dokumen A dan B"
```

### Untuk Web Search
```
- "Apa berita terbaru tentang [topik]?"
- "Jelaskan tren terkini dalam [industri]"
- "Siapa expert di bidang [topik]?"
- "Apa perkembangan terbaru tentang [teknologi]?"
```

### Kombinasi
```
- "Apakah informasi di dokumen masih relevan dengan kondisi terkini?"
- "Validasi fakta dalam dokumen dengan informasi dari web"
- "Update saya tentang perkembangan topik ini sejak dokumen dibuat"
```

## 📱 Access dari Device Lain

### Di Network yang Sama

```bash
# Cek IP address
# Linux/Mac:
ifconfig | grep "inet "

# Windows:
ipconfig

# Run dengan server address
streamlit run app.py --server.address 0.0.0.0

# Access dari device lain:
http://<your-ip>:8501
```

### Deploy Online (Gratis)

**Streamlit Cloud (Recommended)**

```bash
1. Push ke GitHub
2. Visit: share.streamlit.io
3. Connect repository
4. Add GEMINI_API_KEY di Secrets
5. Deploy!
```

## 🆘 Butuh Bantuan?

- 📖 Baca: [README.md](README.md)
- 🐛 Report issue di GitHub
- 💬 Diskusi di Issues section

## 🎉 Selamat Mencoba!

Jika berhasil, jangan lupa:
- ⭐ Star repository ini
- 📢 Share ke teman-teman
- 🤝 Contribute jika ada improvement

---

**Pro Tip**: Bookmark aplikasi lokal Anda dan gunakan sebagai research assistant pribadi! 🚀
