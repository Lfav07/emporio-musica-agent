"""API HTTP do agente Empório da Música.

Inicie na raiz do projeto com:
    uv run uvicorn emporio_musica_agent.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from langchain_core.messages import AIMessage, HumanMessage
from ollama import ResponseError
from pydantic import BaseModel, Field

from .agent import get_agent
from .formatting import normalize_answer
from .memory import add_messages, get_history


class ChatRequest(BaseModel):
    session_id: Annotated[str, Field(min_length=1, max_length=200)]
    message: Annotated[str, Field(min_length=1, max_length=10_000)]


class ChatResponse(BaseModel):
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cria uma instância do agente para reutilizar durante a vida da API."""
    app.state.agent = get_agent()
    yield


app = FastAPI(
    title="Empório da Música — API do agente",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    agent = request.app.state.agent
    user_message = HumanMessage(content=payload.message)
    history = get_history(payload.session_id)

    try:
        result = agent.invoke({"messages": [*history, user_message]})
    except (httpx.RequestError, ResponseError, ConnectionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Não foi possível acessar o Ollama. Verifique se o serviço está "
                "ativo e se o modelo configurado está disponível."
            ),
        ) from exc

    messages = result.get("messages", [])
    final_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, AIMessage)
            and message.content
            and not message.tool_calls
        ),
        None,
    )
    if final_message is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="O agente não retornou uma resposta.",
        )

    answer = final_message.content
    if not isinstance(answer, str):
        answer = "".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict)
        )

    add_messages(payload.session_id, [user_message, final_message])
    return ChatResponse(answer=normalize_answer(answer))
