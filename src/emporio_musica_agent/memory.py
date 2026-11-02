"""Memória de conversas em processo, organizada por sessão."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MAX_HISTORY_MESSAGES = 12

# Memória volátil: é limpa ao encerrar o processo. Operações simultâneas na
# mesma sessão não são sincronizadas; a ordem pode se misturar em um app concorrente.
_sessions: dict[str, list[BaseMessage]] = {}


def get_history(session_id: str) -> list[BaseMessage]:
    """Retorna uma cópia do histórico da sessão, sem mensagens intermediárias de tools."""
    return list(_sessions.get(session_id, []))


def add_messages(
    session_id: str,
    messages: list[BaseMessage],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> None:
    """Acrescenta mensagens de usuário e respostas finais e limita o histórico.

    Mensagens de tool e mensagens do assistente que iniciam tool calls são
    descartadas para nunca deixar uma chamada sem a resposta correspondente.
    O limite é arredondado para baixo a um número par, preservando turnos completos.
    """
    history = _sessions.setdefault(session_id, [])
    history.extend(
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (isinstance(message, AIMessage) and not message.tool_calls)
    )

    limit = max(0, max_messages)
    limit -= limit % 2
    if len(history) > limit:
        del history[: len(history) - limit]
