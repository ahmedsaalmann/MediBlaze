"""
📚 MediBlaze Ingestion Script
Loads PDF(s) from ./Data, splits them into chunks, and uploads them to Pinecone.

IMPORTANT: INDEX_NAME and EMBEDDING_MODEL below MUST match agent/utils/tools.py
(rag_tool), otherwise the chatbot's retriever will connect to an empty/wrong index.

Usage:
    1. Put your PDF file(s) inside the `Data/` folder next to this script.
    2. python ingest.py
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("❌ PINECONE_API_KEY is missing. Add it to your .env file.")

# Must match agent/utils/tools.py exactly
INDEX_NAME = "mediblaze-index"
EMBEDDING_MODEL = "multilingual-e5-large"
EMBEDDING_DIMENSION = 1024  # dimension of multilingual-e5-large

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"


def load_pdfs(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(
            f"❌ Data directory not found: {data_dir}\n"
            f"   Create it and place your PDF file(s) inside, e.g. {data_dir / 'Book.pdf'}"
        )
    loader = DirectoryLoader(str(data_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise FileNotFoundError(f"❌ No PDF files found in {data_dir}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def ensure_index_exists(pc: Pinecone):
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"✅ Index '{INDEX_NAME}' already exists, uploading into it.")
        return
    print(f"⏳ Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("⏳ Waiting for index to become ready...")
    time.sleep(60)


def upload_in_batches(chunks, embeddings, batch_size: int = 90):
    total = len(chunks)
    uploaded = 0

    # Connect directly to the existing index - avoids from_documents()
    # which calls list_indexes() internally with an old API that breaks on SDK v5+
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"⬆️  Uploading batch {batch_num}: chunks {i + 1}-{i + len(batch)} of {total}")
        try:
            vectorstore.add_documents(documents=batch)
            uploaded += len(batch)
            print(f"✅ Uploaded. Total: {uploaded}/{total}")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Batch {batch_num} failed ({e}). Retrying once after 10s...")
            time.sleep(10)
            try:
                vectorstore.add_documents(documents=batch)
                uploaded += len(batch)
                print(f"✅ Retry succeeded. Total: {uploaded}/{total}")
            except Exception as retry_error:
                print(f"❌ Retry failed for batch {batch_num}: {retry_error}")
                continue

    return uploaded


def main():
    print(f"📄 Loading PDFs from: {DATA_DIR}")
    documents = load_pdfs(DATA_DIR)
    print(f"📄 Loaded {len(documents)} page(s)")

    chunks = split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")

    embeddings = PineconeEmbeddings(model=EMBEDDING_MODEL)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index_exists(pc)

    uploaded = upload_in_batches(chunks, embeddings)
    print(f"🎉 Done! Uploaded {uploaded}/{len(chunks)} chunks to Pinecone index '{INDEX_NAME}'")


if __name__ == "__main__":
    main()
