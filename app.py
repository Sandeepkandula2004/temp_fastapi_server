# =========================================================
# Imports & Setup
# =========================================================
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import requests
import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from uuid import UUID
from supabase import create_client, Client
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# =========================================================
# Environment & Config
# =========================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials missing (SUPABASE_URL / SUPABASE_KEY)")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")

JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "60"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")

GOOGLE_ANDROID_CLIENT_ID = os.getenv("GOOGLE_ANDROID_CLIENT_ID")
REDIRECT_URI = os.getenv("REDIRECT_URI_ANDROID", "http://127.0.0.1:8000/auth/callback")

# Google Generative AI init (optional)
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GENAI_MODEL)
else:
    model = None

# =========================================================
# FastAPI App Init
# =========================================================
app = FastAPI()
MAX_RECENT_MESSAGES = int(os.getenv("MAX_RECENT_MESSAGES", "10"))

# =========================================================
# Helpers (Supabase, JWT, Common Functions)
# =========================================================
def check_resp(resp, raise_on_missing: bool = True):
    """Validate a supabase-py execute() response object and return data."""
    if resp is None:
        if raise_on_missing:
            raise HTTPException(status_code=500, detail="No response from Supabase (None).")
        return None
    if not hasattr(resp, "data"):
        if raise_on_missing:
            raise HTTPException(status_code=500, detail="Invalid response from Supabase.")
        return None
    return resp.data

def encode_jwt(payload: dict) -> str:
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def create_jwt_token(user_id: int, email: str, role: str):
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRY_MINUTES),
        "iat": datetime.utcnow()
    }
    return encode_jwt(payload)

def decode_jwt_token(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# =========================================================
# Dependencies (Auth Middleware)
# =========================================================
def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    token = parts[1]
    payload = decode_jwt_token(token)

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    resp = supabase.table("users").select("*").eq("id", int(user_id_str)).maybe_single().execute()
    data = check_resp(resp, raise_on_missing=False)
    if not data:
        raise HTTPException(status_code=401, detail="User not found")
    return data

# =========================================================
# Pydantic Models
# =========================================================
class NewMessage(BaseModel):
    session_id: Optional[UUID] = None
    message: str

class GoogleSignInPayload(BaseModel):
    id_token: str

# =========================================================
# Auth Routes
# =========================================================
@app.post("/auth/google")
def auth_google(payload: GoogleSignInPayload):
    # 1. Verify the token with Google
    verify_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={payload.id_token}"
    google_resp = requests.get(verify_url)
    if google_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Invalid Google ID token")

    google_data = google_resp.json()
    email = google_data.get("email")
    name = google_data.get("name") or email
    if not email:
        raise HTTPException(status_code=400, detail="Google account missing email")

    # 2. Check if user exists in Supabase, else create
    check_user_resp = supabase.table("users").select("*").eq("email", email).maybe_single().execute()
    existing = check_resp(check_user_resp, raise_on_missing=False)
    if not existing:
        insert_resp = supabase.table("users").insert({
            "name": name, "email": email, "role": "user"
        }).execute()
        inserted = check_resp(insert_resp)
        if not inserted:
            raise HTTPException(status_code=500, detail="Failed to create user")
        user_data = inserted[0]
    else:
        user_data = existing

    # 3. Create JWT for the app
    jwt_token = create_jwt_token(
        user_data["id"], 
        user_data["email"], 
        user_data.get("role", "user")
    )

    return JSONResponse({
        "message": f"Welcome, {user_data.get('name')}",
        "jwt_token": jwt_token,
        "user": user_data
    })

# =========================================================
# Session Routes
# =========================================================
@app.get("/sessions")
def list_sessions(user=Depends(get_current_user)):
    resp = supabase.table("sessions").select("*").eq("user_id", user["id"]).order("created_at", desc=True).execute()
    data = check_resp(resp, raise_on_missing=False)
    return data or []

@app.get("/messages/{session_id}")
def get_messages(session_id: UUID, user=Depends(get_current_user)):
    session_resp = supabase.table("sessions").select("*").eq("id", str(session_id)).maybe_single().execute()
    session = check_resp(session_resp, raise_on_missing=False)
    if not session or session.get("user_id") != user["id"]:
        raise HTTPException(status_code=403, detail="Not your session")
    messages_resp = supabase.table("messages").select("*").eq("session_id", str(session_id)).order("timestamp").execute()
    messages = check_resp(messages_resp, raise_on_missing=False)
    return messages or []

# =========================================================
# Message & Bot Reply Route
# =========================================================
@app.post("/message/add")
def add_message(data: NewMessage, user=Depends(get_current_user)):
    """
    Add a user message, create session if needed, generate bot reply, and update summary.
    """

    # --- Step 1: Session Handling ---
    session_id = data.session_id
    new_session_created = False
    session_data = None

    if not session_id:
        new_session_created = True
        title = data.message[:40] if data.message else "New Chat"

        if model:
            try:
                title_prompt = f"Give a short and clear 3–5 word title for this new conversation:\n\nUser: {data.message}"
                title_resp = model.generate_content(title_prompt)
                title = title_resp.text.strip()
            except Exception:
                pass

        session_insert_resp = supabase.table("sessions").insert({
            "user_id": user["id"],
            "user_email": user["email"],
            "title": title,
            "summary": ""
        }).execute()

        inserted = check_resp(session_insert_resp)
        if not inserted:
            raise HTTPException(status_code=500, detail="Failed to create session")
        session_id = inserted[0]["id"]
        session_data = inserted[0]

    else:
        session_resp = (
            supabase.table("sessions")
            .select("user_id, summary")
            .eq("id", str(session_id))
            .maybe_single()
            .execute()
        )
        session_data = check_resp(session_resp, raise_on_missing=False)
        if not session_data or session_data.get("user_id") != user["id"]:
            raise HTTPException(status_code=403, detail="Not your session")

    # --- Step 2: Save User Message ---
    supabase.table("messages").insert({
        "session_id": str(session_id),
        "sender": "user",
        "message": data.message
    }).execute()

    # --- Step 3: Prepare Conversation History ---
    messages_resp = (
        supabase.table("messages")
        .select("*")
        .eq("session_id", str(session_id))
        .order("timestamp", desc=True)
        .limit(MAX_RECENT_MESSAGES)
        .execute()
    )
    recent_messages = check_resp(messages_resp, raise_on_missing=False) or []
    recent_messages = list(reversed(recent_messages))

    # --- Step 4: System Prompt + AI Reply ---
    SYSTEM_PROMPT = """You are a friendly and helpful AI assistant.
- Be concise but clear.
- Do not prefix your answers with "Bot:" or "AI:".
- Respond naturally, like a chat conversation.
- If asked something unclear, politely ask for clarification.
"""

    history_text = "\n".join(
        [("User" if m["sender"] == "user" else "Assistant") + ": " + m["message"]
         for m in recent_messages]
    )

    prompt = SYSTEM_PROMPT + "\n\nConversation so far:\n" + history_text + \
             f"\nAssistant:"

    bot_reply = f"[local fallback reply] You said: {data.message}"
    if model:
        try:
            bot_reply = model.generate_content(prompt).text.strip()

            # Ensure no accidental "Assistant:" prefix in the reply
            if bot_reply.lower().startswith("assistant:"):
                bot_reply = bot_reply.split(":", 1)[1].strip()

        except Exception as e:
            print(f"Gemini API call failed: {e}")

    # --- Step 5: Save Bot Reply ---
    supabase.table("messages").insert({
        "session_id": str(session_id),
        "sender": "bot",
        "message": bot_reply
    }).execute()

    # --- Step 6: Update Summary (Optional) ---
    if len(recent_messages) >= MAX_RECENT_MESSAGES and model:
        try:
            summary_prompt = f"""
Conversation summary so far: {session_data.get('summary', '')}

New messages to include:
{history_text}

Update the summary concisely in 2–3 sentences.
"""
            summary_resp = model.generate_content(summary_prompt)
            new_summary = summary_resp.text.strip()
            supabase.table("sessions").update(
                {"summary": new_summary}
            ).eq("id", str(session_id)).execute()
        except Exception as e:
            print(f"AI summary update error: {e}")

    return {
        "session_id": session_id,
        "reply": bot_reply,
        "new_session": new_session_created
    }

