# Importar bibliotecas
import os
from groq import Groq
from asserts.vector_store import search_documents
from dotenv import load_dotenv

# Carregar as variáveis do arquivo .env
load_dotenv()

# Obter a variável de ambiente
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Instanciar o cliente Groq
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Função para gerar a resposta
def generate_response(query):
    # Pesquisar os documentos
    retrieved_docs = search_documents(query)

    if not retrieved_docs:
        print("DEBUG: Resposta não gerada a partir do PDF indexado.")
        return "Desculpe, não encontrei informações sobre isso."

    # Concatenar os documentos
    context = "\n".join(retrieved_docs)

    # Formatar a entrada para o modelo
    prompt = f"""
        Você é um assistente de IA que só pode responder perguntas com base no documento fornecido.
        Não informe ao usuário que você pesquisou no documento, nem mencione o documento.
        Se a pergunta não puder ser respondida com as informações do documento, apenas responda:
        "Desculpe, não encontrei informações sobre isso. "

        Contexto:\n{context}\n
        Pergunta: {query}\n
        Resposta:
        """

    # Gerar a resposta
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    resposta = response.choices[0].message.content

    # Verificar se a resposta contém o contexto do documento indexado
    if any(doc.lower() in resposta.lower() for doc in retrieved_docs):
        print("DEBUG: Resposta gerada a partir do PDF indexado.")
    else:
        print("DEBUG: Resposta pode ter sido gerada a partir da internet ou extrapolada.")

    # Retornar a resposta
    return resposta
