# Importar bibliotecas
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Instanciar o modelo de incorporação
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Instanciar o vector store com o modelo de incorporação
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Função para indexar um documento PDF
def index_pdf(pdf_path):
    # Importar a função load_pdf e chunk_text
    from asserts.pdf_processor import load_pdf, chunk_text

    # Verificar o diretório
    if not os.path.exists("./chroma_db"):
        text = load_pdf(pdf_path)
        chunks = chunk_text(text)
        vector_store.add_texts(chunks)

# Função para pesquisar documentos
def search_documents(query, top_k=5):
    # Pesquisar os documentos no vector store
    results = vector_store.similarity_search(query, k=top_k)

    # Retornar os documentos
    return [res.page_content for res in results]