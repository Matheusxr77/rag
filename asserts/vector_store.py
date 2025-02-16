# Importar bibliotecas
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Instanciar o modelo de incorporação
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Instanciar o vector store com o modelo de incorporação
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

def index_pdf(pdf_path):
    """Carrega e indexa um documento PDF."""
    from asserts.pdf_processor import load_pdf, chunk_text

    # Verifica se o diretório do banco de vetores existe
    # if not os.path.exists("./chroma_db"):
    text = load_pdf("./files/imposto_renda.pdf")  # Texto já limpo
    chunks = chunk_text(text)
    vector_store.add_texts(chunks)


# Função para pesquisar documentos
def search_documents(query, top_k=5, similarity_search=0.85):

    print("Número de documentos indexados:", vector_store._collection.count())

    # Pesquisar os documentos no vector store
    results = vector_store.similarity_search_with_score(query, k=top_k)
    if not results:
        print("No results")
    filtered_results = [res[0].page_content for res in results if res[1] >= similarity_search]

    # Retornar os documentos
    #return [res.page_content for res in results]
   

    return filtered_results