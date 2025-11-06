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

# variables from .env
load_dotenv()

#FastAPI instance
app = FastAPI(title="Project West Campus Chatbot")

# accesible from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# max number of requests per RATE_LIMIT_WINDOW_SEC per IP (default: 20)
RATE_LIMIT_REQ_PER_TIME = int(os.getenv("RATE_LIMIT_REQ_PER_MIN", "20"))
# time window for rate limit in seconds
RATE_LIMIT_WINDOW_SEC = 60
# stores request timestamps per IP address
_ip_buckets: dict[str, deque[float]] = {}

# Prevents sending more than RATE_LIMIT_REQ_PER_TIME by keeping track of IP addresses and # of requests it made in the RATE_LIMIT_WINDOW_TIME time
def check_rate_limit(ip: str):
    now = time.time()
    bucket = _ip_buckets.setdefault(ip, deque())
    
    # remove timestamps outside the time window
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SEC:
        bucket.popleft()
        
    # too many requests made, block temporarily with 429 error
    if len(bucket) >= RATE_LIMIT_REQ_PER_TIME:
        retry_after = int(RATE_LIMIT_WINDOW_SEC - (now - bucket[0]))
        raise HTTPException(status_code=429, detail=f"Rate limit: try again in ~{retry_after}s")
    
    bucket.append(now)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# load FAISS vector database
try:
    vectordb = FAISS.load_local(
        "vectordb",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    # if FAISS database doesn't exist
    raise RuntimeError("Run `python ingest.py` first to build embeddings") from e

# initialization of LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    request_timeout=30,
    max_retries=2
)

# name aliases for model clarity hardcoded
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

# chatbot personality and instructions (should be very specific)
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

# expected structure of requests
class AskReq(BaseModel):
    question: str
    k: int = 3  

# structure of chatbot responses
class AskResp(BaseModel):
    answer: str
    sources: List[str]

# helper: converts retrieved documents into a formatted string
def render_context(docs):
    return "\n\n".join(
        f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}"
        for d in docs
    )

# main API endpoint that receives a question, gets relevant context, and returns an AI generated answer
@app.post("/ask", response_model=AskResp)
def ask(req: AskReq, request: Request):
    # rate limiting
    client_ip = request.client.host or "unknown"
    check_rate_limit(client_ip)

    # input validation
    question = req.question.strip()
    question = normalize_aliases(question)
    if not question:
        raise HTTPException(400, "Enter a question.")
    if len(question) > 500:
        raise HTTPException(400, "Question too long (limit 500 characters).")

    # retrieve relevant documents from FAISS DB
    docs = vectordb.max_marginal_relevance_search(
        question,
        k=max(1, min(req.k, 5)),
        fetch_k=12,
        lambda_mult=0.3
    )

    context = render_context(docs)
    # message to LLM
    user_msg = f"Question: {question}\n\nContext:\n{context}"

    # LLM response generation
    reply = llm.invoke([
        SystemMessage(content=SYSTEM_MESSAGE),
        HumanMessage(content=user_msg)
    ])

    # top 2 sources files
    seen = []
    for d in docs:
        s = d.metadata.get("source")
        if s and s not in seen:
            seen.append(s)
        if len(seen) >= 2:
            break

    # final answer with sources
    return AskResp(answer=reply.content.strip(), sources=seen)
