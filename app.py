import csv
import time
import os
import streamlit as st
from asserts.llm_interface import generate_response
from asserts.pdf_processor import load_pdf  
from asserts.vector_store import embedding_model  # Para capturar modelo de embedding

METRICS_CSV = "C:/Users/sueht/Documents/GitHub/rag/files/metrics_results.csv"  # Arquivo de métricas

# Configurações do modelo
LLM_MODEL = "llama3-8b-8192"  # Modelo de linguagem usado
EMBEDDING_MODEL = embedding_model.model_name  # Nome do modelo de embeddings
CHUNK_SIZE = 256
CHUNK_OVERLAP = 25

def save_metrics_to_csv(query, response, response_time):
    """Salva cada interação no arquivo de métricas com informações detalhadas."""
    file_exists = os.path.exists(METRICS_CSV)

    with open(METRICS_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Pergunta", "Resposta Gerada", "Tempo de Resposta (s)", 
                "Modelo de Linguagem", "Modelo de Embedding", 
                "Chunk Size", "Chunk Overlap"
            ])

        writer.writerow([
            query, response, round(response_time, 4),
            LLM_MODEL, EMBEDDING_MODEL,
            CHUNK_SIZE, CHUNK_OVERLAP
        ])

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
    start_time = time.time()
    response = generate_response(query)
    response_time = time.time() - start_time

    # Adicionar resposta do modelo ao histórico
    if response:
        st.session_state.chat_history.append(("assistant", response))
        save_metrics_to_csv(query, response, response_time)

# Exibir todo o histórico de chat
for role, response in st.session_state.chat_history:
    with st.chat_message(role):
        if role == "assistant":
            # Formatar a resposta do assistente com markdown
            st.markdown(response)
        else:
            # Mostrar a pergunta do usuário
            st.markdown(response)
