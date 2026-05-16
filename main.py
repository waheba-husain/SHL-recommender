from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import uvicorn
from agent import get_agent_reply

app = FastAPI(title="SHL Assessment Recommender")

# ── Request / Response schemas (non-negotiable per spec) ────────────────────
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    
    # Convert pydantic models to plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    
    result = get_agent_reply(messages)
    
    return ChatResponse(
        reply=result["reply"],
        recommendations=result.get("recommendations", []),
        end_of_conversation=result.get("end_of_conversation", False)
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)