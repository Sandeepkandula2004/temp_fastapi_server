from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os
import uuid

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# In-memory store for sessions
chat_sessions = {}

# Model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

# Initialize the Gemini Flash model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# Input schema
class ChatInput(BaseModel):
    session_id: Optional[str] = None
    message: str

# Chat endpoint
@app.post("/chat/")
async def chat_with_bot(chat_input: ChatInput):
    # Generate a new session ID if not provided
    session_id = chat_input.session_id or str(uuid.uuid4())

    # If session is new, start a new chat
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[])

    # Get the chat object
    chat = chat_sessions[session_id]

    try:
        # Send user message to Gemini Flash
        response = chat.send_message(chat_input.message)

        # Return session ID, response, and history
        return {
            "session_id": session_id,
            "response": response.text,
            "history": [
                {"role": m.role, "text": m.parts[0].text}
                for m in chat.history
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# Optional: view full chat history
@app.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    chat = chat_sessions.get(session_id)
    if not chat:
        return {"error": "Session not found"}
    return {
        "session_id": session_id,
        "history": [
            {"role": m.role, "text": m.parts[0].text}
            for m in chat.history
        ]
    }
