# Importar bibliotecas
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from asserts.pdf_processor import load_pdf, chunk_text

# Instancia o modelo de incorporação
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_directory = "./chroma_db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Instancia o vector store com o modelo de incorporação
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def index_pdf(pdf_path="./files/imposto_renda.pdf"):  
    """Indexa um documento PDF no banco de vetores"""
    print(f"📂 Carregando PDF do caminho: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"❌ Erro: O arquivo {pdf_path} não foi encontrado.")
        return

    text = load_pdf(pdf_path)
    if not text:
        print("❌ Erro: Nenhum texto para indexar.")
        return

    print(f"📄 Texto extraído do PDF: {text[:500]}...")

    chunks = chunk_text(text)
    if not chunks:
        print("❌ Erro: Nenhum chunk gerado.")
        return

    print(f"✂️ Total de chunks gerados: {len(chunks)}")
    print(f"📝 Exemplo de chunk: {chunks[0]}")

    print("🚀 Indexando documento no banco de vetores...")
    try:
        vector_store.add_texts(chunks)
    except Exception as e:
        print(f"❌ Erro ao adicionar textos ao banco de vetores: {e}")
        return

    indexed_count = vector_store._collection.count()
    print(f"📚 Documentos indexados no banco: {indexed_count}")

    if indexed_count == 0:
        print("❌ Erro: Nenhum documento foi indexado! Verifique a conexão com o banco de vetores.")
    
    print("🔎 Verificando conteúdo no banco de vetores...")
    stored_texts = vector_store._collection.peek(3)  
    print(f"📄 Textos armazenados: {stored_texts}")

def split_query(query, chunk_size=256, chunk_overlap=25):
    """Divide a query em pedaços menores para busca eficiente."""
    if len(query) > chunk_size:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(query)
    return [query]

def search_documents(query, top_k=5):
    print(f"🔍 Buscando documentos com a query: {query}")

    indexed_count = vector_store._collection.count()
    if indexed_count == 0:
        print("❌ Erro: Nenhum documento está indexado. Certifique-se de que o PDF foi processado corretamente.")
        return []

    query_chunks = split_query(query)

    try:
        query_embeddings = [embedding_model.embed_query(chunk) for chunk in query_chunks]
    except Exception as e:
        print(f"❌ Erro ao gerar embeddings da query: {e}")
        return []

    results = []
    for embedding in query_embeddings:
        try:
            result = vector_store.similarity_search_by_vector(embedding, k=top_k)
            results.extend(result)
        except Exception as e:
            print(f"❌ Erro na busca por similaridade: {e}")
            return []

    results = list({res.page_content: res for res in results}.values())

    if not results:
        print("⚠️ Nenhum resultado encontrado. Verifique a indexação.")

    print(f"📌 Total de resultados encontrados: {len(results)}")
    for res in results[:3]:  
        print(f"📄 Resultado encontrado: {res.page_content[:500]}...")

    return [res.page_content for res in results]
