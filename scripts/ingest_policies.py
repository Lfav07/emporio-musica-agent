from pathlib import Path
import gc
import re
import shutil
import unicodedata

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from emporio_musica_agent.config import BACKUP_CHROMA_DIR, CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, PDF, STAGING_CHROMA_DIR, get_embeddings



SANITY_CASES = (
    {
        "query": "troca por preferência cor modelo tamanho 7 dias",
        "page": 4,
        "k": 4,
        "phrases": (
            "Trocas por preferência",
            "até 7 dias",
            "disponibilidade do produto desejado",
            "perfeito estado",
            "embalagem original",
        ),
    },
    {
        "query": "arrependimento",
        "page": 4,
        "k": 5,
        "phrases": (
            "7 (sete) dias corridos",
            "embalagem original",
            "mesma forma de pagamento original",
            "frete de devolução",
        ),
    },
    {
        "query": "frete",
        "page": 4,
        "k": 5,
        "phrases": (
            "frete grátis",
            "taxa fixa de R$ 35,00",
            "1 a 3 dias úteis",
        ),
    },
    {
        "query": "frete para outras cidades PAC SEDEX Jadlog",
        "page": 5,
        "k": 5,
        "phrases": (
            "5 a 12 dias úteis",
            "2 a 5 dias úteis",
            "3 a 8 dias úteis",
        ),
    },
    {
        "query": "garantia",
        "page": 7,
        "k": 5,
        "phrases": (
            "90 (noventa) dias",
            "6 meses a 2 anos",
            "certificado de garantia",
        ),
    },
    {
        "query": "produto com defeito de fabricação prazo para troca 30 dias",
        "page": 4,
        "k": 5,
        "phrases": (
            "30 (trinta) dias corridos após a compra",
            "acionar a garantia diretamente com o fabricante",
        ),
    },
    {
        "query": "garantia não cobre desgaste natural mau uso queda umidade",
        "page": 7,
        "k": 5,
        "phrases": (
            "desgaste natural",
            "mau uso",
            "terceiros não autorizados",
        ),
    },
)


def load_pdf():
    loader = PyPDFLoader(str(PDF), extraction_mode="layout")
    docs = loader.load()
    return docs


def clean_docs(docs: list[Document]) -> list[Document]:
    cleaned_docs = []
    for doc in docs:
        content = unicodedata.normalize("NFKC", doc.page_content)
        lines = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue

            if line.casefold() == "empório da música manual de políticas e procedimentos":
                continue
            if re.fullmatch(r"página\s+\d+", line, flags=re.IGNORECASE):
                continue

            # Layout extraction aligns table columns with repeated spaces.
            if re.match(r"^(?:[•▪◦]|\d+[.)])\s", line):
                line = re.sub(r"[ \t]+", " ", line)
            else:
                cells = re.split(r"[ \t]{2,}", line)
                if len(cells) > 1:
                    line = " | ".join(
                        re.sub(r"[ \t]+", " ", cell).strip()
                        for cell in cells
                        if cell.strip()
                    )
                else:
                    line = re.sub(r"[ \t]+", " ", line)
            lines.append(line)
        content = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        cleaned_docs.append(
            Document(page_content=content, metadata=doc.metadata.copy())
        )

    # Keep a section heading with the content when a page break separates them.
    section_heading = re.compile(r"^\d{1,2}(?:\.\d{1,2})?\s+\D.{0,100}$")
    for index, doc in enumerate(cleaned_docs[:-1]):
        nonempty_lines = [
            line.strip() for line in doc.page_content.splitlines() if line.strip()
        ]
        if not nonempty_lines:
            continue
        trailing_line = nonempty_lines[-1]
        if not section_heading.fullmatch(trailing_line):
            continue

        next_doc = cleaned_docs[index + 1]
        if trailing_line.casefold() in next_doc.page_content.casefold():
            continue
        cleaned_docs[index + 1] = Document(
            page_content=f"{trailing_line}\n\n{next_doc.page_content}",
            metadata=next_doc.metadata.copy(),
        )

    return cleaned_docs


def split_docs(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def clear_index(index_dir: Path = CHROMA_DIR) -> None:
    if index_dir.exists():
        try:
            shutil.rmtree(index_dir)
        except PermissionError as exc:
            raise RuntimeError(
                f"Não foi possível remover {index_dir}. "
                "Feche chat_cli/API que possam estar usando o índice; "
                "a remoção não foi forçada."
            ) from exc


def index_chunks(
    chunks: list[Document], persist_directory: Path = CHROMA_DIR
) -> Chroma:
    return Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )


def _page_label(doc: Document) -> str:
    label = doc.metadata.get("page_label")
    if label:
        return str(label)
    page = doc.metadata.get("page")
    try:
        return str(int(page) + 1)
    except (TypeError, ValueError):
        return str(page if page is not None else "?")


def sanity_check(
    chroma: Chroma,
    query: str = "troca por preferência cor modelo tamanho 7 dias",
    k: int = 4,
    expected_page: int | None = None,
    expected_phrases: tuple[str, ...] = (),
) -> bool:
    results = chroma.similarity_search(query, k=k)
    print(f"\nBusca de sanidade: {query!r} (k={k})")
    if not results:
        print("Nenhum resultado. Algo deu errado na indexação.")
        return False

    passed = False
    for i, doc in enumerate(results, start=1):
        page = _page_label(doc)
        preview = re.sub(r"\s+", " ", doc.page_content).strip()[:350]
        print(f"\n[{i}] página {page}\n{preview}...")
        normalized_content = re.sub(r"\s+", " ", doc.page_content).casefold()
        phrases_found = all(
            phrase.casefold() in normalized_content
            for phrase in expected_phrases
        )
        if (expected_page is None or page == str(expected_page)) and phrases_found:
            passed = True

    if expected_phrases:
        print("Resultado do caso:", "PASSOU" if passed else "NÃO PASSOU")
    return passed if expected_phrases else bool(results)


def run_sanity_suite(chroma: Chroma) -> bool:
    results = []
    for case in SANITY_CASES:
        results.append(
            sanity_check(
                chroma,
                query=case["query"],
                k=case["k"],
                expected_page=case["page"],
                expected_phrases=case["phrases"],
            )
        )
    passed = sum(results)
    print(f"\nResumo do sanity check: {passed}/{len(results)} casos passaram")
    return all(results)


def promote_staging_index() -> None:
    if BACKUP_CHROMA_DIR.exists():
        raise RuntimeError(
            f"Backup já existe em {BACKUP_CHROMA_DIR}; verifique-o antes de continuar."
        )

    moved_old_index = False
    try:
        if CHROMA_DIR.exists():
            CHROMA_DIR.replace(BACKUP_CHROMA_DIR)
            moved_old_index = True
        STAGING_CHROMA_DIR.replace(CHROMA_DIR)
    except OSError as exc:
        if moved_old_index and BACKUP_CHROMA_DIR.exists() and not CHROMA_DIR.exists():
            try:
                BACKUP_CHROMA_DIR.replace(CHROMA_DIR)
            except OSError as rollback_exc:
                raise RuntimeError(
                    "Falha ao promover o novo índice e também ao restaurar o anterior. "
                    f"O índice anterior permanece em {BACKUP_CHROMA_DIR}."
                ) from rollback_exc
        raise RuntimeError(
            "Não foi possível trocar o índice. Feche chat_cli/API que o estejam "
            "usando; o índice anterior foi preservado quando possível."
        ) from exc

    if BACKUP_CHROMA_DIR.exists():
        try:
            shutil.rmtree(BACKUP_CHROMA_DIR)
        except PermissionError:
            print(
                "Aviso: o novo índice foi instalado; o backup anterior continua em "
                f"{BACKUP_CHROMA_DIR} porque algum processo ainda o mantém aberto."
            )


def main() -> None:
    docs = load_pdf()
    print(f"{len(docs)} páginas carregadas")

    docs = clean_docs(docs)
    if docs:
        print("\nPrévia limpa da página 1 (primeiros 300 caracteres):")
        print(docs[0].page_content[:300])

    chunks = split_docs(docs)
    print(f"{len(chunks)} chunks gerados")

    clear_index(STAGING_CHROMA_DIR)
    chroma = index_chunks(chunks, persist_directory=STAGING_CHROMA_DIR)
    print("Índice temporário gravado em", STAGING_CHROMA_DIR)

    if not run_sanity_suite(chroma):
        raise RuntimeError(
            "A avaliação de retrieval falhou; o índice anterior foi mantido e "
            f"o candidato ficou em {STAGING_CHROMA_DIR}."
        )

    client = getattr(chroma, "_client", None)
    close_client = getattr(client, "close", None)
    if callable(close_client):
        close_client()
    del chroma
    gc.collect()
    promote_staging_index()
    print("Índice aprovado e instalado em", CHROMA_DIR)


if __name__ == "__main__":
    main()
