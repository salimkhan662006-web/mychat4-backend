"""
Secure AI Backend — MyChat4 App 2 (Document Agent)
====================================================
This server holds ALL your API keys. The browser never sees them.
It automatically tries Groq -> Gemini -> OpenRouter, in that order,
so if one provider is rate-limited or down, the next one takes over.
"""

import os
import io
import time
import httpx
import PyPDF2
import jwt
from jwt import PyJWK
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
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

# Supabase now signs user session JWTs with a rotating asymmetric key
# pair rather than one static shared secret. We verify tokens against
# Supabase's public JWKS endpoint instead of storing a secret.
#
# IMPORTANT: Supabase's JWKS endpoint, like all its /auth/v1/* routes,
# requires an `apikey` header even though the keys it returns are
# public — omitting it returns 401. We use our own small fetcher
# (below) instead of PyJWT's built-in PyJWKClient so we can attach
# that header, and we cache results briefly to avoid hitting Supabase
# on every single request.
SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json" if SUPABASE_URL else ""

_jwks_cache = {"keys": None, "fetched_at": 0}
_JWKS_CACHE_TTL_SECONDS = 600  # refetch at most every 10 minutes


def _fetch_jwks() -> dict:
    """Fetches Supabase's public signing keys, with the required apikey
    header, and caches the result briefly."""
    now = time.time()
    if _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"] < _JWKS_CACHE_TTL_SECONDS):
        return _jwks_cache["keys"]

    if not SUPABASE_JWKS_URL or not SUPABASE_KEY:
        raise Exception("Supabase URL/key not configured on server.")

    res = httpx.get(
        SUPABASE_JWKS_URL,
        headers={"apikey": SUPABASE_KEY},
        timeout=10,
    )
    if res.status_code != 200:
        raise Exception(f"Could not fetch JWKS ({res.status_code}): {res.text[:200]}")

    keys_data = res.json()
    _jwks_cache["keys"] = keys_data
    _jwks_cache["fetched_at"] = now
    return keys_data


def _get_signing_key_for_token(token: str) -> str:
    """Finds the specific public key (by 'kid') that matches the given
    JWT's header, from Supabase's JWKS."""
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    jwks = _fetch_jwks()
    for key_dict in jwks.get("keys", []):
        if key_dict.get("kid") == kid:
            return PyJWK.from_dict(key_dict).key

    raise Exception("No matching signing key found for this token.")

# Free-tier daily request caps, as published by each provider.
# Used to calculate "remaining" quota for the rate limit UI.
PROVIDER_DAILY_CAPS = {
    "groq": 14400,       # openai/gpt-oss-120b free tier
    "gemini": 1500,      # gemini-3.6-flash free tier
    "openrouter": 200,   # conservative estimate for :free models
}


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


def log_usage(endpoint: str, provider: str, prompt_tokens: int, completion_tokens: int, success: bool = True, user_id: str | None = None):
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
            "user_id": user_id,
        }).execute()
    except Exception as e:
        print(f"Usage logging failed (non-fatal): {e}")


# ---------------------------------------------------------------
# Authentication — validates the Supabase-issued JWT on protected
# routes and extracts the calling user's id.
# ---------------------------------------------------------------
async def get_current_user(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: reads the 'Authorization: Bearer <token>'
    header, verifies it was genuinely issued by Supabase for this
    project (via Supabase's public JWKS endpoint), and returns the
    user's id. Raises 401 if missing/invalid/expired."""
    if not SUPABASE_JWKS_URL:
        raise HTTPException(status_code=503, detail="Auth not configured on server.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        signing_key = _get_signing_key_for_token(token)
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["ES256", "RS256"],
            audience="authenticated",
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid session token: {e}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Could not verify token signature: {e}")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user id.")

    return user_id


async def get_optional_user(authorization: str | None = Header(default=None)) -> str | None:
    """Like get_current_user, but returns None instead of raising when
    no token is present — for routes that work for both logged-in and
    anonymous/incognito use (e.g. /chat, /agent don't require login)."""
    if not authorization or not authorization.startswith("Bearer ") or not SUPABASE_JWKS_URL:
        return None
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


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
    prior_context: str | None = None


class ConversationCreate(BaseModel):
    title: str = "New chat"
    is_incognito: bool = False


class ConversationUpdate(BaseModel):
    title: str | None = None
    pinned: bool | None = None


class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str
    is_pinned_ref: bool = False


class BoxCreateRequest(BaseModel):
    title: str = "Boxed chat"
    source_conversation_ids: list[str]  # whole conversations to pull references from  # previous AI answer(s) on this doc, if any


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
            json={"model": "openai/gpt-oss-120b", "messages": messages},
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
            f"gemini-3.6-flash:generateContent?key={GEMINI_KEY}",
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
                "model": "meta-llama/llama-3.1-8b-instruct",
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


async def get_ai_response(history: list[dict], message: str, endpoint: str = "chat", user_id: str | None = None) -> dict:
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
                user_id=user_id,
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
async def chat(req: ChatRequest, user_id: str | None = Depends(get_optional_user)):
    history = [m.model_dump() for m in req.history]
    result = await get_ai_response(history, req.message, endpoint="chat", user_id=user_id)
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
async def run_agent(req: AgentRequest, user_id: str | None = Depends(get_optional_user)):
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

    result = await get_ai_response(history=[], message=prompt, endpoint="agent", user_id=user_id)

    # Safety net: if the model said it needs the full doc, retry once with it.
    if result["reply"].strip() == "NEED_FULL_DOCUMENT":
        fallback_prompt = (
            "You are a careful, precise document assistant. Here is the document:\n\n"
            f"{req.doc_text}\n\n"
            f"Task: {req.task}\n\n"
            "Complete this task thoroughly and clearly, using only information "
            "from the document above."
        )
        result = await get_ai_response(history=[], message=fallback_prompt, endpoint="agent", user_id=user_id)

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


@app.get("/rate-limits")
async def get_rate_limits():
    """Returns today's request count per provider vs their free-tier
    daily cap, so the UI can show 'X of Y requests used today'."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Usage tracking not configured on server.")

    try:
        # "Today" boundary in UTC — simple and consistent regardless of
        # where the request comes from.
        start_of_day = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        res = (
            supabase.table("usage_logs")
            .select("provider")
            .gte("created_at", start_of_day)
            .execute()
        )
        rows = res.data

        counts_today = {}
        for r in rows:
            p = r["provider"]
            counts_today[p] = counts_today.get(p, 0) + 1

        result = {}
        for provider, cap in PROVIDER_DAILY_CAPS.items():
            used = counts_today.get(provider, 0)
            result[provider] = {
                "used": used,
                "cap": cap,
                "remaining": max(cap - used, 0),
                "percent_used": round((used / cap) * 100, 1) if cap else 0,
            }

        return {"providers": result, "reset_note": "Resets daily at midnight UTC"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch rate limits: {e}")


# ---------------------------------------------------------------
# Conversations — real chat history with pin/delete
# ---------------------------------------------------------------
@app.post("/conversations")
async def create_conversation(req: ConversationCreate, user_id: str = Depends(get_current_user)):
    if req.is_incognito:
        # Incognito conversations are never written to Supabase.
        # The frontend keeps them entirely in memory.
        raise HTTPException(
            status_code=400,
            detail="Incognito conversations should not be saved via this endpoint.",
        )
    if not supabase:
        raise HTTPException(status_code=503, detail="History storage not configured.")

    try:
        res = supabase.table("conversations").insert({
            "title": req.title,
            "user_id": user_id,
        }).execute()
        return res.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create conversation: {e}")


@app.get("/conversations")
async def list_conversations(user_id: str = Depends(get_current_user)):
    if not supabase:
        raise HTTPException(status_code=503, detail="History storage not configured.")

    try:
        res = (
            supabase.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("pinned", desc=True)
            .order("updated_at", desc=True)
            .execute()
        )
        return res.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list conversations: {e}")


@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, user_id: str = Depends(get_current_user)):
    if not supabase:
        raise HTTPException(status_code=503, detail="History storage not configured.")

    try:
        # Verify this conversation actually belongs to the requesting user
        owner_check = (
            supabase.table("conversations")
            .select("id")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not owner_check.data:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        res = (
            supabase.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )
        return res.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch messages: {e}")


@app.patch("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, req: ConversationUpdate, user_id: str = Depends(get_current_user)):
    if not supabase:
        raise HTTPException(status_code=503, detail="History storage not configured.")

    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")

    try:
        res = (
            supabase.table("conversations")
            .update(updates)
            .eq("id", conversation_id)
            .eq("user_id", user_id)  # can't update someone else's chat
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not update conversation: {e}")


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str = Depends(get_current_user)):
    if not supabase:
        raise HTTPException(status_code=503, detail="History storage not configured.")

    try:
        res = (
            supabase.table("conversations")
            .delete()
            .eq("id", conversation_id)
            .eq("user_id", user_id)  # can't delete someone else's chat
            .execute()
        )
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete conversation: {e}")


@app.post("/messages")
async def save_message(req: MessageCreate, user_id: str = Depends(get_current_user)):
    """Saves a single message to a conversation. Called by the frontend
    after every user message and every completed AI response — but
    NEVER called for incognito conversations."""
    if not supabase:
        raise HTTPException(status_code=503, detail="History storage not configured.")

    try:
        # Verify the conversation belongs to this user before writing to it
        owner_check = (
            supabase.table("conversations")
            .select("id")
            .eq("id", req.conversation_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not owner_check.data:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        res = supabase.table("messages").insert({
            "conversation_id": req.conversation_id,
            "role": req.role,
            "content": req.content,
            "is_pinned_ref": req.is_pinned_ref,
        }).execute()

        # Bump the conversation's updated_at so it sorts to the top of history
        supabase.table("conversations").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", req.conversation_id).execute()

        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save message: {e}")


@app.patch("/messages/{message_id}/pin")
async def toggle_pin_message(message_id: str, user_id: str = Depends(get_current_user)):
    """Marks/unmarks a message as a 'key message' reference — used by
    the box feature to pull precise context from a chat instead of
    replaying the whole conversation."""
    if not supabase:
        raise HTTPException(status_code=503, detail="History storage not configured.")

    try:
        current = supabase.table("messages").select("is_pinned_ref").eq("id", message_id).execute()
        if not current.data:
            raise HTTPException(status_code=404, detail="Message not found.")

        new_state = not current.data[0]["is_pinned_ref"]
        res = (
            supabase.table("messages")
            .update({"is_pinned_ref": new_state})
            .eq("id", message_id)
            .execute()
        )
        return res.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not toggle pin: {e}")


@app.post("/conversations/box")
async def create_boxed_conversation(req: BoxCreateRequest, user_id: str = Depends(get_current_user)):
    """Creates a new conversation that combines reference points from
    two or more existing chats. For each source conversation, we use
    any messages the person explicitly pinned as 'key messages' — or,
    if none were pinned, fall back to that conversation's most recent
    exchange. This keeps the merge cheap: the AI gets specific locked-in
    context instead of replaying either full chat."""
    if not supabase:
        raise HTTPException(status_code=503, detail="History storage not configured.")
    if not req.source_conversation_ids or len(req.source_conversation_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 conversations to box.")

    try:
        source_messages = []

        for conv_id in req.source_conversation_ids:
            # Confirm this source conversation actually belongs to the
            # requesting user before pulling anything from it.
            owner_check = (
                supabase.table("conversations")
                .select("id")
                .eq("id", conv_id)
                .eq("user_id", user_id)
                .execute()
            )
            if not owner_check.data:
                continue  # skip conversations that aren't theirs, silently

            # First preference: messages explicitly pinned in this chat
            pinned_res = (
                supabase.table("messages")
                .select("*")
                .eq("conversation_id", conv_id)
                .eq("is_pinned_ref", True)
                .execute()
            )

            if pinned_res.data:
                source_messages.extend(pinned_res.data)
            else:
                # Fallback: most recent message in this conversation,
                # so boxing still works even if nothing was pinned.
                recent_res = (
                    supabase.table("messages")
                    .select("*")
                    .eq("conversation_id", conv_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                source_messages.extend(recent_res.data)

        if not source_messages:
            raise HTTPException(
                status_code=404,
                detail="None of the selected conversations have any messages to reference.",
            )

        # Create the new boxed conversation, owned by this user
        conv_res = supabase.table("conversations").insert({
            "title": req.title,
            "user_id": user_id,
        }).execute()
        new_conv = conv_res.data[0]

        context_lines = "\n\n".join(
            f"[Reference from a prior chat — {m['role']}]: {m['content']}"
            for m in source_messages
        )
        seed_content = (
            "The following are key reference points carried over from "
            "earlier conversations. Treat them as already-established "
            "context:\n\n" + context_lines
        )

        supabase.table("messages").insert({
            "conversation_id": new_conv["id"],
            "role": "assistant",
            "content": f"I've combined the key points you selected. Here's what I'm carrying forward:\n\n{context_lines}",
            "is_pinned_ref": False,
        }).execute()

        return {"conversation": new_conv, "seed_context": seed_content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create boxed conversation: {e}")