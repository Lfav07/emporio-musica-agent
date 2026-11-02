from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import unicodedata
import pandas as pd
from langchain_core.tools import tool
from .config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, get_embeddings
from langchain_chroma import Chroma
from functools import lru_cache
from langchain_core.documents import Document

products = pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - products.csv")
categories = pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - categories.csv")
promotions = pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - promotions.csv")
orders = pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - orders.csv")
order_items = pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - order_items.csv")
customers = pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - customers.csv")

ORDER_STATUS_PT = {
    "shipped": "enviado",
    "delivered": "entregue",
    "confirmed": "confirmado",
    "pending": "pendente",
    "cancelled": "cancelado",
}


def _records_without_nan(dataframe: pd.DataFrame) -> list[dict]:
    """Converte as linhas em dicionários sem valores NaN, próprios do pandas."""
    clean_dataframe = dataframe.astype(object).where(pd.notna(dataframe), None)
    return clean_dataframe.to_dict(orient="records")


def _normalize_text(value: object) -> str:
    """Normaliza texto para buscas sem diferenciar maiúsculas ou acentos."""
    text = "" if pd.isna(value) else str(value)
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return without_accents.casefold()


def _token_variants(token: str) -> set[str]:
    """Gera formas singular/plural simples para buscas em português."""
    variants = {token}
    if token.endswith("ao"):
        variants.update({f"{token[:-2]}oes", f"{token[:-2]}aes"})
    elif token.endswith(("oes", "aes")):
        variants.add(f"{token[:-3]}ao")
    elif token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    else:
        variants.add(f"{token}s")
    return variants


def _matches_query(searchable_text: pd.Series, query: str) -> pd.Series:
    """Exige que cada palavra da consulta apareça, aceitando flexões simples."""
    matches = pd.Series(True, index=searchable_text.index)
    for token in _normalize_text(query).split():
        token_matches = pd.Series(False, index=searchable_text.index)
        for variant in _token_variants(token):
            token_matches |= searchable_text.str.contains(variant, regex=False)
        matches &= token_matches
    return matches


@tool
def buscar_produtos(
    termo: str = "",
    categoria: str = "",
    preco_maximo: float | None = None,
) -> list[dict]:
    """Busca produtos por palavras-chave, categoria e preço máximo.

    Inclui estoque e promoções ativas para ajudar o cliente a comparar opções.
    O preço máximo considera o preço promocional quando houver promoção ativa.
    Use termo para o nome, tipo ou descrição do produto; categoria é um filtro opcional.
    """
    category_lookup = categories.rename(
        columns={"name": "category_name", "description": "category_description"}
    )
    catalog = products.merge(category_lookup, on="category_id", how="left")

    active_promotions = promotions.copy()
    active_promotions["is_active"] = (
        pd.to_numeric(active_promotions["is_active"], errors="coerce").fillna(0).eq(1)
    )
    active_promotions = active_promotions[active_promotions["is_active"]]
    active_promotions["discount_percent"] = pd.to_numeric(
        active_promotions["discount_percent"], errors="coerce"
    ).fillna(0)
    # Se houver mais de uma promoção ativa para um produto, aplica a maior.
    active_promotions = active_promotions.loc[
        active_promotions.groupby("product_id")["discount_percent"].idxmax(),
        ["product_id", "discount_percent", "description"],
    ].rename(
        columns={
            "discount_percent": "desconto_percentual",
            "description": "promocao",
        }
    )
    catalog = catalog.merge(active_promotions, on="product_id", how="left")

    catalog["price_brl"] = pd.to_numeric(catalog["price_brl"], errors="coerce")
    catalog["stock_quantity"] = pd.to_numeric(
        catalog["stock_quantity"], errors="coerce"
    )
    catalog["desconto_percentual"] = pd.to_numeric(
        catalog["desconto_percentual"], errors="coerce"
    ).fillna(0)
    catalog["preco_promocional_brl"] = (
        (catalog["price_brl"] * (1 - catalog["desconto_percentual"] / 100))
        .round(2)
        .where(catalog["desconto_percentual"].gt(0))
    )
    catalog["preco_final_brl"] = catalog["preco_promocional_brl"].where(
        catalog["desconto_percentual"].gt(0), catalog["price_brl"]
    )

    # Recomenda apenas produtos que podem ser comprados agora.
    catalog = catalog[
        catalog["status"].fillna("").astype(str).str.casefold().eq("active")
        & catalog["stock_quantity"].gt(0)
    ]

    if termo.strip():
        search_columns = [
            "name",
            "description",
            "category_name",
            "category_description",
            "specs",
        ]
        searchable_text = (
            catalog[search_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .map(_normalize_text)
        )
        # Todas as palavras precisam aparecer, mas podem estar em qualquer ordem.
        catalog = catalog[_matches_query(searchable_text, termo)]

    if categoria.strip():
        normalized_categories = catalog["category_name"].map(_normalize_text)
        catalog = catalog[_matches_query(normalized_categories, categoria)]

    if preco_maximo is not None:
        catalog = catalog[catalog["preco_final_brl"] <= preco_maximo]

    catalog = catalog.sort_values("preco_final_brl").head(10)
    result = catalog[
        [
            "product_id",
            "name",
            "category_name",
            "description",
            "price_brl",
            "desconto_percentual",
            "promocao",
            "preco_promocional_brl",
            "preco_final_brl",
            "stock_quantity",
            "status",
            "specs",
        ]
    ].rename(
        columns={
            "product_id": "id_produto",
            "name": "nome",
            "category_name": "categoria",
            "description": "descricao",
            "price_brl": "preco_brl",
            "stock_quantity": "estoque",
            "status": "status",
            "specs": "especificacoes",
        }
    )
    return _records_without_nan(result)


@tool
def consultar_pedido(order_id: int) -> list[dict]:
    """Consulta status, entrega e itens pelo número do pedido.

    Se o pedido não existir nos dados, retorna um registro com a chave "erro".
    """
    matching_orders = orders[orders["order_id"] == order_id]
    if matching_orders.empty:
        return [{"erro": f"Pedido {order_id} não encontrado nos dados da loja."}]

    order = matching_orders.iloc[0]
    customer = customers[customers["customer_id"] == order["customer_id"]]
    customer_name = customer.iloc[0]["name"] if not customer.empty else None

    items = order_items[order_items["order_id"] == order_id].merge(
        products[["product_id", "name"]], on="product_id", how="left"
    )
    items = items.rename(
        columns={
            "product_id": "id_produto",
            "name": "produto",
            "quantity": "quantidade",
        }
    )[["id_produto", "produto", "quantidade"]]

    result = {
        "numero_pedido": int(order["order_id"]),
        "cliente": customer_name,
        "data_pedido": order["order_date"],
        "status": ORDER_STATUS_PT.get(
            str(order["status"]).casefold(), order["status"]
        ),
        "total_brl": order["total_brl"],
        "forma_pagamento": order["payment_method"],
        "codigo_rastreio": order["tracking_code"],
        "previsao_entrega": order["estimated_delivery"],
        "observacoes": order["notes"],
        "itens": _records_without_nan(items),
    }
    return _records_without_nan(pd.DataFrame([result]))


@lru_cache(maxsize=1)
def _get_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
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


@tool
def consulta_pdf(query: str) -> str:
    """Consulta o manual e os dados institucionais da loja.

    Use para perguntas sobre endereço, localização, telefone, WhatsApp, e-mail,
    horário de funcionamento, trocas, devoluções, arrependimento, frete,
    prazos de entrega, garantia e outras informações presentes no manual.
    """
    results = _get_db().similarity_search(query, k=4)
    if not results:
        return "Nenhum trecho encontrado no manual de políticas."
    return "\n\n".join(
        f"[Página {_page_label(doc)}] {doc.page_content}" for doc in results
    )


def get_csv_tools() -> list:
    return [buscar_produtos, consultar_pedido]


def get_policy_tools() -> list:
    return [consulta_pdf]


@tool
def consultar_data_atual() -> str:
    """Retorna a data de hoje no fuso horário de São Paulo.

    Use esta ferramenta sempre que a pergunta envolver prazos, datas-limite,
    arrependimento, troca ou devolução de um pedido, ou quando precisar comparar
    a data atual com a data de compra ou entrega. Combine a data retornada com
    as regras obtidas na ferramenta de políticas e com os dados do pedido.
    """
    hoje = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    return hoje.strftime("%Y-%m-%d (%d/%m/%Y)")


def get_date_tools() -> list:
    return [consultar_data_atual]
