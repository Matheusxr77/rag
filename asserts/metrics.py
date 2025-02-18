import time
import csv
import os
import numpy as np
from asserts.llm_interface import generate_response
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from asserts.config import METRICS_CSV

LLM_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "paraphrase-xlm-r-multilingual-v1"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 25

def load_questions_answers(csv_path="./files/questions.csv"):
    """Carrega perguntas e respostas esperadas de um arquivo CSV."""
    questions = []
    expected_answers = []

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Ignora cabeçalho

            for row in reader:
                if len(row) >= 2:
                    questions.append(row[0])
                    expected_answers.append(row[1])
                else:
                    print(f"Aviso: Linha ignorada (formato inválido): {row}")
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado: {csv_path}")
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {e}")

    return questions, expected_answers

def calculate_relevance(expected, predicted):
    """Calcula a relevância com base na interseção das palavras-chave."""
    expected_words = set(expected.lower().split())
    predicted_words = set(predicted.lower().split())
    return len(expected_words & predicted_words) / len(expected_words) if expected_words else 0

def save_metrics_to_csv(query, response, response_time, expected_answer, relevance, precision, accuracy):
    """Salva métricas detalhadas no CSV."""
    file_exists = os.path.exists(METRICS_CSV)
    
    with open(METRICS_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow([  # Adiciona cabeçalho apenas se o arquivo não existir
                "Pergunta", "Resposta Esperada", "Resposta Gerada", 
                "Tempo de Resposta (s)", "Relevância", "Precisão", "Acurácia",
                "Modelo de Linguagem", "Modelo de Embedding", 
                "Chunk Size", "Chunk Overlap"
            ])
        
        writer.writerow([  # Adiciona dados de cada interação
            query, expected_answer, response, round(response_time, 4), 
            round(relevance, 4), round(precision, 4), round(accuracy, 4),
            LLM_MODEL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
        ])

def tokenize(text):
    """Divide o texto em palavras únicas para comparar precisão e acurácia."""
    return list(set(text.lower().split()))

def calculate_metrics(questions, expected_answers):
    """Calcula métricas de desempenho e salva os resultados."""
    response_times = []
    relevances = []
    precisions = []
    accuracies = []
    predicted_answers = []

    for question, expected in zip(questions, expected_answers):
        start_time = time.time()
        predicted = generate_response(question)
        end_time = time.time()
        
        response_time = end_time - start_time
        response_times.append(response_time)
        predicted_answers.append(predicted)

        # Cálculo de relevância
        relevance = calculate_relevance(expected, predicted)
        relevances.append(relevance)

        # Tokenização das respostas
        expected_tokens = tokenize(expected)
        predicted_tokens = tokenize(predicted)

        # Certifique-se de que há pelo menos um rótulo em comum
        all_labels = list(set(expected_tokens + predicted_tokens))
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)

        # Converter para valores numéricos
        y_true = label_encoder.transform(expected_tokens) if expected_tokens else []
        y_pred = label_encoder.transform(predicted_tokens) if predicted_tokens else []

        # Cálculo da precisão
        if len(y_true) > 0 and len(y_pred) > 0:
            precision = precision_score([expected.lower()], [predicted.lower()], average='weighted', zero_division=1)
        else:
            precision = 0
        precisions.append(precision)
        
        # Cálculo da acurácia
        accuracy = accuracy_score([expected.lower()], [predicted.lower()])
        accuracies.append(accuracy)

        # Salva métricas detalhadas para cada interação
        save_metrics_to_csv(question, predicted, response_time, expected, relevance, precision, accuracy)

    if not expected_answers or not predicted_answers:
        print("Aviso: Listas de respostas vazias. Não é possível calcular métricas globais.")
        return

    # Calcula o comprimento das respostas esperadas e previstas
    expected_lengths = [len(ans) for ans in expected_answers if ans]
    predicted_lengths = [len(ans) for ans in predicted_answers if ans]

    # Evita erro caso listas estejam vazias
    if expected_lengths and predicted_lengths:
        mae = mean_absolute_error(expected_lengths, predicted_lengths)
        mape = mean_absolute_percentage_error(expected_lengths, predicted_lengths)
    else:
        mae = 0
        mape = 0
    
    # Calcula métricas globais
    avg_response_time = np.mean(response_times)
    avg_relevance = np.mean(relevances)
    avg_precision = np.mean(precisions)
    avg_accuracy = np.mean(accuracies)
    
    # Salvar métricas globais no arquivo CSV
    with open(METRICS_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([])  # Linha em branco
        writer.writerow(["Métricas Globais"])  # Título das métricas globais
        writer.writerow(["Tempo Médio de Resposta (s)", round(avg_response_time, 4)])
        writer.writerow(["Relevância Média", round(avg_relevance, 4)])
        writer.writerow(["Precisão Média", round(avg_precision, 4)])
        writer.writerow(["Acurácia Média", round(avg_accuracy, 4)])
        writer.writerow(["MAE", round(mae, 4)])
        writer.writerow(["MAPE", round(mape, 4)])
    
    # Também salva as métricas globais detalhadas
    save_metrics_to_csv("Métricas Globais", "", 0, "", avg_relevance, avg_precision, avg_accuracy)

    print(f"📊 Métricas salvas em {METRICS_CSV}")
