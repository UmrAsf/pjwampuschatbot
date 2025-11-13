import os, glob, time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

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
    # check for data files (since we aren't web scraping YET)
    if not docs:
        raise SystemExit("No .txt files in /data. Add them first.")

    # split texts into chunks (500 characters; 60 character overlap) for processing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=40
    )

    # split the document into chunks
    chunks, metas = [], []
    for text, source in docs:
        for c in splitter.split_text(text):
            chunks.append(c)
            metas.append({"source": source})

    if not chunks:
        raise SystemExit("No text chunks created from input files.")

    # text chunks to vectors
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # store in Pinecone vector database and upsert all chunks into the index
    start = time.time()
    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metas,
        index_name="projectwampus"
    )
