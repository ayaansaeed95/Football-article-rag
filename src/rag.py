"""RAG pipeline: retrieve → format → generate."""

import sys
from langchain_openai import ChatOpenAI

from src.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from src.retrieval import get_filtered_retriever, related_titles, format_docs_with_metadata

_llm = ChatOpenAI(
    openai_api_base=LLM_API_BASE,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
    temperature=0.1,
)

_PROMPT_TEMPLATE = """\
The following documents come from football articles (originally written in French).

Answer the question in English using only the information in the documents.
If the answer is not in the documents, reply exactly: I don't know

Documents:
{context}

Question: {query}

Answer:"""


def rag_answer(query: str, k: int = 3) -> dict:
    retriever = get_filtered_retriever(query, k=k)
    docs = retriever.invoke(query)

    if not docs:
        return {"answer": "I don't know", "titles": []}

    prompt = _PROMPT_TEMPLATE.format(
        context=format_docs_with_metadata(docs),
        query=query,
    )
    response = _llm.invoke([{"role": "user", "content": prompt}])

    return {
        "answer": response.content.strip(),
        "titles": related_titles(docs),
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is video assisted refereeing in football?"
    result = rag_answer(query)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for title in result["titles"]:
        print(f"  - {title}")
