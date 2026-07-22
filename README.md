# 📚 Data Analysis RAG Mentor — v2.0 (OpenRouter Edition)

> A production-grade RAG system that transforms your Data Analysis PDF books into an
> interactive AI mentor powered by OpenRouter LLM + HuggingFace local embeddings.

---

## 🏗️ Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| LLM | OpenRouter → Llama 3.1 8B | Free tier available |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | 100% Free / Local |
| Vector Store | ChromaDB (persistent, local) | Free |
| Framework | LangChain | Free |
| Web UI | Streamlit | Free |
| API Server | FastAPI + Uvicorn | Free |
| Deployment | Render.com | Free tier available |

---

## 🚀 Quick Start

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First install downloads the HuggingFace model (~90MB). This is cached locally.

### 3. Configure Your API Key

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-your_actual_key_here
```

Get your free key at: https://openrouter.ai/keys

### 4. Add PDF Books

```
my_rag_project/
└── books/
    ├── data_analysis_fundamentals.pdf
    ├── sql_for_analysts.pdf
    └── python_pandas_guide.pdf
```

### 5. Build Vector Database

```bash
# ⚠️ If migrating from Google embeddings, delete old DB first:
rm -rf ./chroma_db       # macOS/Linux
rmdir /s /q chroma_db    # Windows

python ingest.py
```

---

## 🧪 Local Testing

### CLI Interface

```bash
python query_test.py
```

### Streamlit Web UI

```bash
streamlit run app.py
# Opens at: http://localhost:8501
```

### FastAPI Server

```bash
uvicorn api:app --reload
# API:          http://localhost:8000
# Swagger Docs: http://localhost:8000/docs
# ReDoc:        http://localhost:8000/redoc
```

#### Example API Call

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain pivot tables with a real business example"}'
```

---

## ☁️ Streamlit Cloud Deployment

1. Push project to GitHub (include `chroma_db/` folder)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo and set **Main file:** `app.py`
4. Go to **Settings → Secrets** and add:

```toml
OPENROUTER_API_KEY = "sk-or-v1-your_actual_key_here"
```

5. Click **Deploy**

---

## ☁️ Render.com Deployment

### Step-by-Step

1. Push your project to GitHub (include `chroma_db/`)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your repository

| Setting | Value |
|---------|-------|
| **Runtime** | Python |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn api:app --host 0.0.0.0 --port $PORT` |

4. Under **Environment Variables**, add:

| Key | Value |
|-----|-------|
| `OPENROUTER_API_KEY` | `sk-or-v1-your_actual_key_here` |
| `PYTHON_VERSION` | `3.11.0` |

5. Click **Create Web Service**

Your API will be live at: `https://your-service-name.onrender.com`

---

## 🔄 Changing the LLM Model

Edit `rag_core.py`, line:

```python
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
```

Popular OpenRouter free models:

| Model ID | Notes |
|----------|-------|
| `meta-llama/llama-3.1-8b-instruct:free` | Default — fast, capable |
| `mistralai/mistral-7b-instruct:free` | Strong reasoning |
| `google/gemma-2-9b-it:free` | Google Gemma 2 |
| `meta-llama/llama-3.1-70b-instruct` | Paid — highest quality |

Browse all: https://openrouter.ai/models

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENROUTER_API_KEY not set` | Check `.env` file or Streamlit Secrets |
| `ChromaDB not found` | Run `python ingest.py` |
| `No PDFs found` | Add files to `./books/` |
| Embedding mismatch error | Delete `./chroma_db/` and re-run `ingest.py` |
| OpenRouter 429 rate limit | Free tier limit hit — wait or upgrade |
| Render build fails | Ensure `torch` installs cleanly; check build logs |

---

## 🇸🇦 ملخص بالعربي

### نظرة عامة (الإصدار 2.0)
تم تحديث النظام لاستخدام **OpenRouter** بدلاً من Google Gemini،
مع نموذج **HuggingFace محلي** للـ Embeddings بدون أي تكلفة.

### خطوات التشغيل:
1. ضع ملفات PDF في مجلد `books/`
2. احصل على مفتاح API مجاني من openrouter.ai/keys
3. أضف المفتاح في ملف `.env`
4. احذف المجلد القديم: `rm -rf ./chroma_db`
5. شغّل: `python ingest.py`
6. اختبر: `python query_test.py`
7. الواجهة: `streamlit run app.py`
8. الـ API: `uvicorn api:app --reload`

### للنشر على Streamlit Cloud:
- أضف `OPENROUTER_API_KEY` في Settings → Secrets

### للنشر على Render.com:
- أضف `OPENROUTER_API_KEY` في Environment Variables
- Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
