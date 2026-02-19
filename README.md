# 🚀 YouTube RAG System (FastAPI + Gemini)

A production-ready **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about any YouTube video using AI.

This system:
- Extracts YouTube transcripts
- Splits text into chunks
- Generates embeddings using HuggingFace
- Stores vectors in FAISS
- Uses Google Gemini (LLM) to answer questions
- Stores transcripts in SQLite database
- Provides FastAPI backend + Jinja2 UI

---

## 📌 Features

✅ YouTube Transcript Extraction  
✅ Semantic Search with FAISS  
✅ Gemini 2.5 Flash LLM Integration  
✅ FastAPI Backend   
✅ Jinja2 Web Interface  
✅ Clean Project Structure  
✅ Production-Ready Architecture  

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/youtube-rag-fastapi.git
cd youtube-rag-fastapi

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Add Environment Variables

GOOGLE_API_KEY=your_gemini_api_key_here

▶️ Run the Application

uvicorn app:app --reload




