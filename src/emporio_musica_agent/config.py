import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
MODEL = os.getenv('OLLAMA_MODEL', 'qwen3.5:9b')
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PDF = DATA_DIR / "políticas_da_loja.pdf"
CHROMA_DIR = DATA_DIR / "chroma"
STAGING_CHROMA_DIR = DATA_DIR / "chroma_staging"
BACKUP_CHROMA_DIR = DATA_DIR / "chroma_previous"
COLLECTION_NAME = "politicas"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

def get_llm(**kwargs) -> ChatOllama:
    return ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0, reasoning=False, **kwargs)


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model='nomic-embed-text', base_url=BASE_URL)
