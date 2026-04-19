from fastapi import FastAPI
from pydantic import BaseModel
from ai.agent import agent


app = FastAPI()


class ChatRequest(BaseModel):
    user_input: str


@app.post("/chat")
async def chat(request: ChatRequest) -> str:
    response = agent(request.user_input)
    return response
