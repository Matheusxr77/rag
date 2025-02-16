# Importar bibliotecas
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Instanciar o modelo de incorporação
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_directory = "./chroma_db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Instancia o vector store com o modelo de incorporação
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def index_pdf(pdf_path="C:/Users/mayeu/git/rag/files/imposto_renda.pdf"):
    """Carrega e indexa um documento PDF."""
    from asserts.pdf_processor import load_pdf, chunk_text

    # Verifica se o diretório do banco de vetores existe
    if not os.path.exists("./chroma_db"):
        text = load_pdf(pdf_path)  # Texto já limpo
        chunks = chunk_text(text)
        
        print(f"Total de chunks gerados: {len(chunks)}")
        print(f"Primeiro chunk: {chunks[0] if chunks else 'Nenhum chunk gerado'}")

        vector_store.add_texts(chunks)
        print(f"Documentos indexados: {len(chunks)}")

# Não está funcionando corretamente!!!
def split_query(query, chunk_size=256, chunk_overlap=25):
    """Divide a query em pedaços para busca eficiente."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(query)

def search_documents(query, top_k=5):
    print(f"Buscando documentos com a query: {query}")
    
    # Divide a query em chunks
    query_chunks = split_query(query)
    
    # Gera embeddings para cada chunk da query
    query_embeddings = [embedding_model.embed_documents(chunk) for chunk in query_chunks]
    
    # Compara cada chunk da query com os documentos indexados
    results = []
    for embedding in query_embeddings:
        # Compara com os documentos no vector store
        result = vector_store.similarity_search_by_vector(embedding, k=top_k)
        results.extend(result)

    # Remove duplicados (caso haja)
    results = list({res['page_content']: res for res in results}.values())

    if not results:
        print("Nenhum resultado encontrado. Verifique a indexação e embeddings.")

    print(f"Total de resultados encontrados: {len(results)}")

    # Mostra uma amostra dos resultados encontrados
    for res in results:
        print(f"Resultado encontrado: {res['page_content'][:500]}...")  # Mostrar os primeiros 500 caracteres

    # Retorna os documentos
    return [res['page_content'] for res in results]
