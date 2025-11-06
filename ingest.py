"""
Builds the knowledge base for the chatbot using only local text files in /data.
Splits them into chunks, embeds them, and saves a FAISS index.
"""

import os, glob, time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def load_local_texts():
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append((text, f"file://{os.path.basename(path)}"))
    return docs

if __name__ == "__main__":
    start = time.time()

    docs = load_local_texts()
    if not docs:
        raise SystemExit("No .txt files in /data. Add them first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )

    chunks, metas = [], []
    for text, source in docs:
        for c in splitter.split_text(text):
            chunks.append(c)
            metas.append({"source": source})

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_texts(chunks, embeddings, metadatas=metas)
    db.save_local("vectordb")

    print(f"Indexed {len(chunks)} chunks from {len(docs)} text files in {time.time() - start:.1f}s")
