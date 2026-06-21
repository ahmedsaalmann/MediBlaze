# 🏥 MediBlaze

AI Medical Assistant — RAG (Pinecone) + Gemini + DuckDuckGo web search, served via FastAPI.

## 📁 Project Structure

```
MediBlaze/
├── agent/
│   ├── __init__.py
│   ├── agent.py          # LangGraph agent (LLM + tool routing)
│   └── utils/
│       ├── __init__.py
│       ├── prompt.py     # System prompt
│       └── tools.py      # rag_tool + medical_web_search
├── main.py                # FastAPI app (chat, streaming chat, health check)
├── ingest.py               # One-time / repeatable script to upload PDFs to Pinecone
├── Data/                   # Put your source PDF(s) here (not committed to git)
├── static/                 # Frontend static assets (css/js/images)
├── templates/
│   └── index.html          # Optional custom frontend (falls back to a basic page if missing)
├── requirements.txt
├── .env.example
└── DEPLOYMENT.md           # Docker / cloud deployment guide
```

## 🚀 Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# then edit .env and add your GOOGLE_API_KEY and PINECONE_API_KEY

# 4. Add your medical PDF(s) into the Data/ folder
#    e.g. Data/Book.pdf

# 5. Upload the PDFs to Pinecone (run once, or again whenever Data/ changes)
python ingest.py

# 6. Run the API
python main.py
```

Then open:
- Web interface: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## ⚠️ Important: Keep index name & embedding model in sync

`ingest.py` and `agent/utils/tools.py` (`rag_tool`) **must** use the exact same:
- Pinecone index name → `mediblaze-index`
- Embedding model → `multilingual-e5-large`

If you ever change one, change the other — otherwise the chatbot will query an empty
or mismatched index and the RAG tool will return nothing useful.

## 🩺 How it works

1. User sends a message to `/chat` or `/chat/stream`.
2. The LangGraph agent (`agent/agent.py`) decides whether to call:
   - `rag_tool` — searches the Pinecone knowledge base built from your PDFs, or
   - `medical_web_search` — searches the web (restricted to trusted medical sites) via DuckDuckGo.
3. Gemini (`gemini-2.0-flash-lite`) combines the results into a structured, formatted medical answer.

See `DEPLOYMENT.md` for Docker / AWS / GCP / Azure deployment instructions.
