"""
FastAPI backend for the local Project West Campus RAG chatbot.
Retrieves embedded text chunks (FAISS) and answers with GPT-4o.

Upgrades:
- Per-IP rate limiting (default 20 requests/min; override via RATE_LIMIT_REQ_PER_MIN in .env)
- MMR retrieval for more diverse, query-specific chunks
- Show at most top-2 unique sources
- Model instructed not to print sources (UI adds them)
- Input length guardrail + LLM timeout/retries
"""

import os
import time
from collections import deque
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

app = FastAPI(title="Project West Campus Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # OK for local prototype; restrict in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# Rate limiting 
RATE_LIMIT_REQ_PER_MIN = int(os.getenv("RATE_LIMIT_REQ_PER_MIN", "20"))
RATE_LIMIT_WINDOW_SEC = 60
_ip_buckets: dict[str, deque[float]] = {}

def check_rate_limit(ip: str):
    now = time.time()
    bucket = _ip_buckets.setdefault(ip, deque())
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_REQ_PER_MIN:
        retry_after = int(RATE_LIMIT_WINDOW_SEC - (now - bucket[0]))
        raise HTTPException(status_code=429, detail=f"Rate limit: try again in ~{retry_after}s")
    bucket.append(now)

# Vector store and LLM 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    vectordb = FAISS.load_local(
        "vectordb",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    raise RuntimeError("Run `python ingest.py` first to build embeddings") from e

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    request_timeout=30,
    max_retries=2
)

ALIAS_MAP = {
    "pjwampus": "Project West Campus",
    "projectwampus": "Project West Campus",
    "pwc": "Project West Campus",
    "project wampus": "Project West Campus",
}

def normalize_aliases(text: str) -> str:
    t = text.lower()
    for k, v in ALIAS_MAP.items():
        if k in t:
            t = t.replace(k, v.lower())
    return t

SYSTEM_MESSAGE = (
    
    "You are a helpful information chatbot for Project West Campus, a student-led group "
    "that helps feed unhoused neighbors in the West Campus community. "
    "Answer using ONLY the context provided. "
    "If the context does not contain the answer, reply: "
    "'I don’t have that information yet.' "
    "Do NOT mention that you rely on context. "
    "Be friendly, concise (2–4 sentences), and focused on volunteer information. "
    "Do NOT include a 'Sources:' section or citations — the system will add sources."
    "If isn't in context, say 'Not in context.'"
    
)

class AskReq(BaseModel):
    question: str
    k: int = 3   # smaller k reduces repetitive sources

class AskResp(BaseModel):
    answer: str
    sources: List[str]

def render_context(docs):
    return "\n\n".join(
        f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}"
        for d in docs
    )

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq, request: Request):
    client_ip = request.client.host or "unknown"
    check_rate_limit(client_ip)

    question = req.question.strip()
    question = normalize_aliases(question)
    if not question:
        raise HTTPException(400, "Enter a question.")
    if len(question) > 500:
        raise HTTPException(400, "Question too long (limit 500 characters).")

    # MMR for more diverse chunks; fetch_k > k gives retriever more to choose from
    docs = vectordb.max_marginal_relevance_search(
        question,
        k=max(1, min(req.k, 5)),
        fetch_k=12,
        lambda_mult=0.3
    )

    context = render_context(docs)
    user_msg = f"Question: {question}\n\nContext:\n{context}"

    reply = llm.invoke([
        SystemMessage(content=SYSTEM_MESSAGE),
        HumanMessage(content=user_msg)
    ])

    # Keep at most 2 unique sources for a clean look
    seen = []
    for d in docs:
        s = d.metadata.get("source")
        if s and s not in seen:
            seen.append(s)
        if len(seen) >= 2:
            break

    return AskResp(answer=reply.content.strip(), sources=seen)
