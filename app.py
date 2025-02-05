# Importar bibliotecas
import streamlit as st
from asserts.llm_interface import generate_response

# Título e descrição
st.title("📄 Chat RAG - Imposto de Renda")
st.write("Faça perguntas sobre o imposto de renda e obtenha respostas baseadas nos dados do governo federal!")

# Campo de entrada
query = st.chat_input()

# Verificar se a consulta não está vazia
if query:
    # Solicitar a resposta
    st.chat_message("user").write(query)

    # Obter a resposta
    response = generate_response(query)

    # Exibir a resposta
    st.chat_message("assistant").write(response)