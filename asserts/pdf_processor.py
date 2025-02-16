# Importar bibliotecas
import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):
    """Carrega o texto do PDF, limpa espaços extras e caracteres corrompidos."""
    doc = fitz.open(pdf_path)
    
    # Extrair texto de todas as páginas e juntar
    text = "\n".join([page.get_text("text") for page in doc])

    # Normalizar formatação para evitar problemas de exibição
    text = re.sub(r'\s+', ' ', text)  # Remove múltiplos espaços
    text = text.strip()  # Remove espaços extras no início e fim

    print(f"Texto extraído: {text[:500]}...") 

    return text

# Função para dividir o texto em chunks
def chunk_text(text, chunk_size=256, chunk_overlap=25):
    """Divide o texto em chunks menores para processamento"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)