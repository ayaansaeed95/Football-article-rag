import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_BASE = os.getenv("LLM_API_BASE", "https://llmproxy.uva.nl")
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
LLM_EMBEDDING = os.getenv("LLM_EMBEDDING", "text-embedding-3-large")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "cahiers-chunk-08")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".")
