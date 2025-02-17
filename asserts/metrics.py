import time
import csv
import os
import numpy as np
from llm_interface import generate_response
from sklearn.metrics import precision_score, accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from app import METRICS_CSV

def load_questions_answers(csv_path="questions.csv"):
    """Carrega perguntas e respostas esperadas de um arquivo CSV."""
    questions = []
    expected_answers = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Ignora cabeçalho
        for row in reader:
            questions.append(row[0])
            expected_answers.append(row[1])
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
            writer.writerow([
                "Pergunta", "Resposta Esperada", "Resposta Gerada", 
                "Tempo de Resposta (s)", "Relevância", "Precisão", "Acurácia"
            ])
        
        writer.writerow([
            query, expected_answer, response, round(response_time, 4), 
            round(relevance, 4), round(precision, 4), round(accuracy, 4)
        ])

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

        # Cálculo de métricas
        relevance = calculate_relevance(expected, predicted)
        relevances.append(relevance)

        precision = precision_score(
            list(expected.lower()), list(predicted.lower()), average='weighted', zero_division=1
        )
        precisions.append(precision)
        
        accuracy = accuracy_score(list(expected.lower()), list(predicted.lower()))
        accuracies.append(accuracy)

        save_metrics_to_csv(question, predicted, response_time, expected, relevance, precision, accuracy)
    
    # Calcular métricas globais
    mae = mean_absolute_error([len(ans) for ans in expected_answers], [len(ans) for ans in predicted_answers])
    mape = mean_absolute_percentage_error([len(ans) for ans in expected_answers], [len(ans) for ans in predicted_answers])
    avg_response_time = np.mean(response_times)
    avg_relevance = np.mean(relevances)
    avg_precision = np.mean(precisions)
    avg_accuracy = np.mean(accuracies)
    
    # Salvar métricas globais
    with open(METRICS_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Métricas Globais"])
        writer.writerow(["Tempo Médio de Resposta (s)", round(avg_response_time, 4)])
        writer.writerow(["Relevância Média", round(avg_relevance, 4)])
        writer.writerow(["Precisão Média", round(avg_precision, 4)])
        writer.writerow(["Acurácia Média", round(avg_accuracy, 4)])
        writer.writerow(["MAE", round(mae, 4)])
        writer.writerow(["MAPE", round(mape, 4)])
    
    print(f"📊 Métricas salvas em metrics_results.csv")

if __name__ == "__main__":
    questions, expected_answers = load_questions_answers()
    calculate_metrics(questions, expected_answers)

    print("✅ Métricas calculadas com sucesso!")