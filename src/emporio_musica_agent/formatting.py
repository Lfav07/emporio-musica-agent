"""Normalização das respostas expostas pela API e pelo CLI."""

import re


def normalize_answer(answer: str) -> str:
    """Remove Markdown e espaços de controle para retornar texto simples em uma linha."""
    answer = re.sub(r"```(?:[\w+-]+)?\s*([\s\S]*?)```", r"\1", answer)
    answer = re.sub(r"`([^`]+)`", r"\1", answer)
    answer = re.sub(r"\[([^\]]+)\]\((?:[^()]|\([^()]*\))*\)", r"\1", answer)
    answer = re.sub(r"^\s{0,3}#{1,6}\s+", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"^[ \t]*[-*+][ \t]+", "; ", answer, flags=re.MULTILINE)
    answer = re.sub(r"(?<!\w)(\*\*|__)(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", answer)
    answer = re.sub(r"(?<!\w)(\*|_)(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", answer)
    answer = re.sub(r"\s*\n\s*", " ", answer)
    answer = re.sub(r":\s*;", ":", answer)
    answer = re.sub(r"[.!?]\s*;\s*", "; ", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    return re.sub(r"^(?:;\s*)+", "", answer)
