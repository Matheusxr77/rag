from langchain_huggingface import HuggingFaceEmbeddings

METRICS_CSV = "./files/metrics_results.csv"
LLM_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 256
CHUNK_OVERLAP = 25
TOP_K = 5
MAX_TOKENS = 512