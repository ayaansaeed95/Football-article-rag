import re
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_EMBEDDING,
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
)

_embeddings = OpenAIEmbeddings(
    model=LLM_EMBEDDING,
    openai_api_base=LLM_API_BASE,
    api_key=LLM_API_KEY,
)

vector_store = Chroma(
    collection_name=CHROMA_COLLECTION,
    embedding_function=_embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
)


def related_titles(docs: list) -> list[str]:
    titles: list[str] = []
    for doc in docs:
        title = doc.metadata.get("title")
        if title and title not in titles:
            titles.append(title)
    return titles


def format_docs_with_metadata(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        year, month, day = meta.get("year", ""), meta.get("month", ""), meta.get("day", "")

        if year and month and day:
            date_str = f" (Published: {year}-{int(month):02d}-{int(day):02d})"
        elif year:
            date_str = f" (Published: {year})"
        else:
            date_str = ""

        parts.append(
            f"Document {i}:\n"
            f"Title: {meta.get('title', 'Unknown')}{date_str}\n"
            f"Content: {doc.page_content}\n"
        )
    return "\n".join(parts)


def parse_date_filter(query: str) -> dict | None:
    filter_dict: dict = {}

    year_match = re.search(r"\b(19|20)\d{2}\b", query)
    if year_match:
        filter_dict["year"] = int(year_match.group())

    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for name, num in months.items():
        if name in query.lower():
            filter_dict["month"] = num
            break

    return filter_dict or None


def get_filtered_retriever(query: str, k: int = 3):
    metadata_filter = parse_date_filter(query)
    search_kwargs: dict = {"k": k}

    if metadata_filter:
        conditions = [{key: {"$eq": val}} for key, val in metadata_filter.items()]
        search_kwargs["filter"] = (
            {"$and": conditions} if len(conditions) > 1 else conditions[0]
        )
        print(f"Applying metadata filter: {metadata_filter}")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
