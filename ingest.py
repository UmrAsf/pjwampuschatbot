import os, glob, time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# variables from .env
load_dotenv()

# directory for local data and check its existence
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Loads all files from data directory and returns a list of (text, source file) tuples
def load_local_texts():
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append((text, f"file://{os.path.basename(path)}"))
    return docs


if __name__ == "__main__":
    docs = load_local_texts()
    #check for data files
    if not docs:
        raise SystemExit("No .txt files in /data. Add them first.")

    #split texts into chunks (500 characters; 60 character overlap) for processing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60
    )

    #split the document into chunks
    chunks, metas = [], []
    for text, source in docs:
        for c in splitter.split_text(text):
            chunks.append(c)
            metas.append({"source": source})

    #text chunks to vectors
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #store if FAISS local database
    db = FAISS.from_texts(chunks, embeddings, metadatas=metas)
    db.save_local("vectordb")