from fastapi import FastAPI
from pydantic import BaseModel
from ai.agent import agent
from db.ticket_store import TicketStore
from settings import settings


app = FastAPI()
ticket_store = TicketStore(settings.tickets_db_path)


class ChatRequest(BaseModel):
    user_input: str


@app.post("/chat")
async def chat(request: ChatRequest) -> str:
    response = agent(request.user_input)
    return response
