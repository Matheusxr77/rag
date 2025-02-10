# Importar bibliotecas
import streamlit as st
from asserts.llm_interface import generate_response
from asserts.pdf_processor import load_pdf  # Importa a função para carregar o PDF

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Título e descrição
st.title("📄 Chat RAG - Imposto de Renda")
st.write("Faça perguntas sobre o imposto de renda e obtenha respostas baseadas nos dados do governo federal!")

# Campo de entrada
query = st.chat_input()

# Verificar se a consulta não está vazia
if query:
    # Adicionar pergunta do usuário ao histórico
    st.session_state.chat_history.append(("user", query))

    # Obter a resposta do modelo
    response = generate_response(query)

    # Adicionar resposta do modelo ao histórico
    if response:
        st.session_state.chat_history.append(("assistant", response))

# Exibir todo o histórico de chat
for role, response in st.session_state.chat_history:
    with st.chat_message(role):
        if role == "assistant":
            # Formatar a resposta do assistente com markdown
            st.markdown(response)
        else:
            # Mostrar a pergunta do usuário
            st.markdown(response)
