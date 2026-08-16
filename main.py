"""
Secure AI Backend — MyChat4 App 2 (Document Agent)
====================================================
This server holds ALL your API keys. The browser never sees them.
It automatically tries Groq -> Gemini -> OpenRouter, in that order,
so if one provider is rate-limited or down, the next one takes over.
"""

import os
import io
import httpx
import PyPDF2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
GROQ_KEY = os.environ.get("GROQ_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173"
).split(",")

app = FastAPI(title="MyChat4 Secure AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def log_usage(endpoint: str, provider: str, prompt_tokens: int, completion_tokens: int, success: bool = True):
    """Fire-and-forget usage log to Supabase. Never blocks or breaks
    the main request if Supabase is unreachable or not configured."""
    if not supabase:
        return
    try:
        supabase.table("usage_logs").insert({
            "endpoint": endpoint,
            "provider": provider,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "success": success,
        }).execute()
    except Exception as e:
        print(f"Usage logging failed (non-fatal): {e}")


# ---------------------------------------------------------------
# Request models
# ---------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str          # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    history: list[ChatMessage]
    message: str


class AgentRequest(BaseModel):
    doc_text: str
    task: str
    prior_context: str | None = None  # previous AI answer(s) on this doc, if any


# ---------------------------------------------------------------
# Provider functions — each raises an Exception on failure,
# which lets the caller move on to the next provider.
# ---------------------------------------------------------------
async def call_groq(history: list[dict], message: str) -> dict:
    if not GROQ_KEY:
        raise Exception("GROQ_KEY not set on server")

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_KEY}",
            },
            json={"model": "llama-3.3-70b-versatile", "messages": messages},
        )
    if res.status_code != 200:
        raise Exception(f"Groq {res.status_code}: {res.text[:200]}")

    data = res.json()
    usage = data.get("usage", {})
    return {
        "text": data["choices"][0]["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def call_gemini(history: list[dict], message: str) -> dict:
    if not GEMINI_KEY:
        raise Exception("GEMINI_KEY not set on server")

    contents = []
    for m in history:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
            json={"contents": contents},
        )
    if res.status_code != 200:
        raise Exception(f"Gemini {res.status_code}: {res.text[:200]}")

    data = res.json()
    usage = data.get("usageMetadata", {})
    return {
        "text": data["candidates"][0]["content"]["parts"][0]["text"],
        "prompt_tokens": usage.get("promptTokenCount", 0),
        "completion_tokens": usage.get("candidatesTokenCount", 0),
    }


async def call_openrouter(history: list[dict], message: str) -> dict:
    if not OPENROUTER_KEY:
        raise Exception("OPENROUTER_KEY not set on server")

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_KEY}",
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": messages,
            },
        )
    if res.status_code != 200:
        raise Exception(f"OpenRouter {res.status_code}: {res.text[:200]}")

    data = res.json()
    usage = data.get("usage", {})
    return {
        "text": data["choices"][0]["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def get_ai_response(history: list[dict], message: str, endpoint: str = "chat") -> dict:
    """Tries Groq -> Gemini -> OpenRouter in order. Returns which one
    answered, and logs token usage to Supabase (non-blocking)."""
    errors = []

    for name, fn in [
        ("groq", call_groq),
        ("gemini", call_gemini),
        ("openrouter", call_openrouter),
    ]:
        try:
            result = await fn(history, message)
            log_usage(
                endpoint=endpoint,
                provider=name,
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
            )
            return {"reply": result["text"], "provider": name}
        except Exception as e:
            errors.append(f"{name}: {e}")

    raise HTTPException(
        status_code=502,
        detail="All AI providers failed.\n" + "\n".join(errors),
    )


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "MyChat4 backend is running"}


@app.post("/chat")
async def chat(req: ChatRequest):
    history = [m.model_dump() for m in req.history]
    result = await get_ai_response(history, req.message, endpoint="chat")
    return result


@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()

    if file.filename.lower().endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload a .pdf or .txt file.",
            )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in this file (it may be a scanned image PDF).",
        )

    # Cap at 50k characters so we don't blow past model context limits
    return {"text": text[:50000], "truncated": len(text) > 50000}


@app.post("/agent")
async def run_agent(req: AgentRequest):
    if not req.doc_text.strip():
        raise HTTPException(status_code=400, detail="No document text provided.")
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="No task specified.")

    if req.prior_context:
        # Follow-up action on a document we've already sent once.
        # Much cheaper: reuse the AI's own prior analysis instead of
        # resending the full document text again.
        prompt = (
            "You previously analysed a document and gave this response:\n\n"
            f"{req.prior_context}\n\n"
            f"New task on the SAME document: {req.task}\n\n"
            "If your prior response already contains enough information to "
            "complete this new task, use it. If you genuinely need to see "
            "the original document again to answer accurately, say exactly: "
            "NEED_FULL_DOCUMENT and nothing else."
        )
    else:
        # First action on this document — send the full text once.
        prompt = (
            "You are a careful, precise document assistant. Here is the document:\n\n"
            f"{req.doc_text}\n\n"
            f"Task: {req.task}\n\n"
            "Complete this task thoroughly and clearly, using only information "
            "from the document above. If the document doesn't contain enough "
            "information to complete the task, say so directly."
        )

    result = await get_ai_response(history=[], message=prompt, endpoint="agent")

    # Safety net: if the model said it needs the full doc, retry once with it.
    if result["reply"].strip() == "NEED_FULL_DOCUMENT":
        fallback_prompt = (
            "You are a careful, precise document assistant. Here is the document:\n\n"
            f"{req.doc_text}\n\n"
            f"Task: {req.task}\n\n"
            "Complete this task thoroughly and clearly, using only information "
            "from the document above."
        )
        result = await get_ai_response(history=[], message=fallback_prompt, endpoint="agent")

    return {"result": result["reply"], "provider": result["provider"]}


@app.get("/usage")
async def get_usage():
    """Returns aggregated usage stats: total requests, tokens, and a
    breakdown per provider. Powers the Usage panel in the frontend."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Usage tracking not configured on server.")

    try:
        res = supabase.table("usage_logs").select("*").execute()
        rows = res.data

        total_requests = len(rows)
        total_tokens = sum(r["total_tokens"] for r in rows)
        total_prompt_tokens = sum(r["prompt_tokens"] for r in rows)
        total_completion_tokens = sum(r["completion_tokens"] for r in rows)

        by_provider = {}
        for r in rows:
            p = r["provider"]
            if p not in by_provider:
                by_provider[p] = {"requests": 0, "tokens": 0}
            by_provider[p]["requests"] += 1
            by_provider[p]["tokens"] += r["total_tokens"]

        by_endpoint = {}
        for r in rows:
            e = r["endpoint"]
            if e not in by_endpoint:
                by_endpoint[e] = {"requests": 0, "tokens": 0}
            by_endpoint[e]["requests"] += 1
            by_endpoint[e]["tokens"] += r["total_tokens"]

        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "by_provider": by_provider,
            "by_endpoint": by_endpoint,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch usage: {e}")