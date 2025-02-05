# Importar bibliotecas
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Função para carregar o texto de um arquivo PDF
def load_pdf(pdf_path):
    # Abrir o arquivo PDF
    doc = fitz.open(pdf_path)

    # Inicializar a variável de texto
    text = "\n".join([page.get_text() for page in doc])
    
    # Retornar o arquivo PDF
    return text

# Função para dividir o texto em chunks
def chunk_text(text, chunk_size=256, chunk_overlap=25):
    # Instanciar o text_splitter com os parâmetros
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Dividir o texto
    return text_splitter.split_text(text)