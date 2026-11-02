"""Chat interativo de terminal para o agente Empório da Música.

Execute com `uv run emporio-musica-agent` ou
`uv run python -m emporio_musica_agent.chat_cli` na raiz do projeto.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from ollama import ResponseError

from .agent import get_agent
from .formatting import normalize_answer
from .memory import add_messages, get_history


def _message_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
    )


def _new_session_id() -> str:
    return f"cli-{uuid4().hex[:12]}"


def main() -> None:
    print("Empório da Música — chat")
    print("Converse livremente. Digite /ajuda para ver os comandos.\n")
    session_id = _new_session_id()

    try:
        agent = get_agent()
    except Exception as exc:
        print(f"Não foi possível iniciar o agente: {exc}")
        return

    print("Empório da Música: Olá! Como posso ajudar?\n")

    try:
        while True:
            try:
                user_text = input("Você: ").strip()
            except EOFError:
                print("\nConversa encerrada.")
                break

            if not user_text:
                continue
            command = user_text.casefold()
            if command in {"/sair", "/exit", "/quit"}:
                print("Conversa encerrada.")
                break
            if command == "/ajuda":
                print("Comandos: /ajuda mostra esta mensagem; /sair encerra.\n")
                continue

            user_message = HumanMessage(content=user_text)
            conversation = [*get_history(session_id), user_message]
            print("Empório da Música: pensando...", end="\r", flush=True)

            try:
                result = agent.invoke({"messages": conversation})
            except (httpx.RequestError, ResponseError, ConnectionError):
                print(
                    "Empório da Música: Não foi possível acessar o Ollama. "
                    "Verifique se o serviço está ativo e se o modelo configurado está disponível."
                )
                continue
            except Exception as exc:
                print(f"Empório da Música: Não consegui processar a mensagem ({exc}).")
                continue

            final_message = next(
                (
                    message
                    for message in reversed(result.get("messages", []))
                    if isinstance(message, AIMessage)
                    and message.content
                    and not message.tool_calls
                ),
                None,
            )
            if final_message is None:
                print("Empório da Música: Não consegui gerar uma resposta.\n")
                continue

            answer = normalize_answer(_message_text(final_message.content))
            add_messages(session_id, [user_message, final_message])
            print(f"\rEmpório da Música: {answer}\n")
    except KeyboardInterrupt:
        print("\nConversa encerrada.")


if __name__ == "__main__":
    main()
