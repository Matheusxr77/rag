import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(pdf_path="C:/Users/mayeu/git/rag/imposto_renda.pdf"):
    """Carrega o texto do PDF, limpa espaços extras e caracteres corrompidos."""
    try:
        doc = fitz.open(pdf_path)
        print(f"📄 PDF carregado: {pdf_path}")
        print(f"📜 Total de páginas: {len(doc)}")
        
        # Extrair texto de todas as páginas e juntar
        extracted_text = []
        for i, page in enumerate(doc):
            page_text = page.get_text("text")  # Extrai texto puro
            if not page_text.strip():
                print(f"⚠️ Aviso: Página {i + 1} está vazia ou não contém texto extraível.")
            extracted_text.append(page_text)

        # Unir todo o texto
        text = "\n".join(extracted_text)

        # Verificar se algum texto foi extraído
        if not text.strip():
            print("❌ Erro: Nenhum texto foi extraído do PDF!")
            return ""

        # Normalizar formatação
        text = re.sub(r'\s+', ' ', text).strip()  # Remove múltiplos espaços

        print(f"✅ Texto extraído com sucesso! Total de caracteres: {len(text)}")
        print(f"📝 Prévia do texto extraído:\n{text[:500]}...")

        return text

    except Exception as e:
        print(f"❌ Erro ao carregar o PDF: {e}")
        return ""

# Função para dividir o texto em chunks
def chunk_text(text, chunk_size=256, chunk_overlap=25):
    """Divide o texto em chunks menores para processamento"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    
    print(f"🔹 Total de chunks gerados: {len(chunks)}")
    if chunks:
        print(f"📌 Primeiro chunk: {chunks[0]}")
    
    return chunks
