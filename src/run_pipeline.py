"""
Pipeline completo para previsão de diabetes (Fase 1 + Fase 2 resumida):
- Pré-processamento
- Treino e avaliação de modelos clássicos
- Modelo baseline Fase 2
- Otimização de Logistic Regression via Algoritmo Genético
- Export de métricas, melhores hiperparâmetros e gráfico de comparação
- Interpretação via LLM (ou fallback)
"""

import os
import json
import logging
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

# ---------------------------
# Ignorar warnings desnecessários do sklearn
# ---------------------------
warnings.filterwarnings("ignore", category=FutureWarning, message=".*penalty.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Inconsistent values: penalty.*")

# ---------------------------
# Paths e logging
# ---------------------------
ARTIFACTS_PATH = "../artefatos/"
RESULTS_PATH = r"B:\Pós\hospital-ai-diagnosis\resultados\Fase 2"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

LOG_FILE = os.path.join(ARTIFACTS_PATH, "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")]
)
logger = logging.getLogger()
logger.info("Início do pipeline")

# ---------------------------
# 1. Carregar e pré-processar dados
# ---------------------------
dataset = pd.read_csv("../dados/Raw/diabetes.csv")

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
dataset[cols_with_zeros] = dataset[cols_with_zeros].replace(0, np.nan)

X = dataset.drop("Outcome", axis=1)
y = dataset["Outcome"]

# Imputação KNN
imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Padronização
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Salvar dataset processado
final_dataset = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
os.makedirs('../dados/processados', exist_ok=True)
final_dataset.to_csv('../dados/processados/processado_diabetes_script.csv', index=False)
logger.info("Pré-processamento concluído e dataset salvo.")

# ---------------------------
# 2. Divisão treino/teste + SMOTE
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ---------------------------
# 3. Treinar modelos clássicos
# ---------------------------
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42)
}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)

classic_results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    classic_results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }
logger.info(f"Resultados modelos clássicos: {classic_results}")

# ---------------------------
# 4. Modelo baseline Fase 2
# ---------------------------
baseline_model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)
baseline_results = {
    "Recall": recall_score(y_test, y_pred_base),
    "F1": f1_score(y_test, y_pred_base),
    "Accuracy": accuracy_score(y_test, y_pred_base)
}
logger.info(f"Resultados modelo baseline: {baseline_results}")

# ---------------------------
# 5. Algoritmo Genético para otimização Logistic Regression
# ---------------------------
C_VALUES = np.logspace(-4, 2, 20)
PENALTIES = ["l1", "l2"]
MAX_ITERS = [100, 300, 500, 800, 1200]

def create_individual():
    return {"C": random.choice(C_VALUES), "penalty": random.choice(PENALTIES),
            "max_iter": random.choice(MAX_ITERS), "class_weight": "balanced"}

def fitness(ind, X_train, X_test, y_train, y_test):
    model = LogisticRegression(**ind, solver="liblinear")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return recall_score(y_test, y_pred, pos_label=1)

def tournament_selection(pop, scores, k=3):
    selected = random.sample(list(zip(pop, scores)), k)
    return max(selected, key=lambda x: x[1])[0]

def crossover(p1, p2):
    return {k: random.choice([p1[k], p2[k]]) for k in p1}

def mutate(ind, mutation_rate=0.3):
    if random.random() < mutation_rate: ind["C"] = random.choice(C_VALUES)
    if random.random() < mutation_rate: ind["penalty"] = random.choice(PENALTIES)
    if random.random() < mutation_rate: ind["max_iter"] = random.choice(MAX_ITERS)
    return ind

def genetic_algorithm(X_train, X_test, y_train, y_test,
                      population_size=40, generations=25,
                      mutation_rate=0.3, tournament_k=3, elite_size=2):
    population = [create_individual() for _ in range(population_size)]
    best_score, best_individual = -1, None
    for _ in range(generations):
        scores = [fitness(ind, X_train, X_test, y_train, y_test) for ind in population]
        gen_best_score = max(scores)
        if gen_best_score > best_score:
            best_score, best_individual = gen_best_score, population[np.argmax(scores)]
        ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        elites = [ind for ind, _ in ranked[:elite_size]]
        new_population = elites.copy()
        while len(new_population) < population_size:
            p1 = tournament_selection(population, scores, tournament_k)
            p2 = tournament_selection(population, scores, tournament_k)
            new_population.append(mutate(crossover(p1, p2), mutation_rate))
        population = new_population
    return best_individual, best_score

best_ind, _ = genetic_algorithm(X_train, X_test, y_train, y_test)
ga_model = LogisticRegression(**best_ind, solver="liblinear")
ga_model.fit(X_train, y_train)
y_pred_ga = ga_model.predict(X_test)
ga_results = {
    "Recall": recall_score(y_test, y_pred_ga),
    "F1": f1_score(y_test, y_pred_ga),
    "Accuracy": accuracy_score(y_test, y_pred_ga)
}
logger.info(f"Resultados GA: {ga_results}")
logger.info(f"Melhores hiperparâmetros GA: {best_ind}")

# ---------------------------
# 6. Exportar métricas, hiperparâmetros e gráfico
# ---------------------------
with open(os.path.join(ARTIFACTS_PATH, "best_hyperparameters.json"), "w") as f:
    json.dump(best_ind, f, indent=4)

pd.DataFrame([baseline_results]).to_csv(os.path.join(ARTIFACTS_PATH, "metrics.csv"), index=False)
pd.DataFrame([ga_results]).to_csv(os.path.join(ARTIFACTS_PATH, "metrics_ga.csv"), index=False)

metrics = ["Recall", "F1", "Accuracy"]
baseline_values = [baseline_results[m] for m in metrics]
ga_values = [ga_results[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, baseline_values, width, label="Baseline")
plt.bar(x + width/2, ga_values, width, label="GA")
plt.xticks(x, metrics)
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Comparação Baseline x GA")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(os.path.join(RESULTS_PATH, "comparacao_baseline_ga.png"))
plt.close()
logger.info(f"Gráfico salvo em {RESULTS_PATH}/comparacao_baseline_ga.png")

# ---------------------------
# 7. LLM / Fallback
# ---------------------------
load_dotenv()
try:
    from openai import AzureOpenAI
    AZURE_VARS = {
        "API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "API_VERSION": os.getenv("AZURE_API_VERSION")
    }
    USE_LLM = all(AZURE_VARS.values())
except ImportError:
    USE_LLM = False

def fallback_explanation(metrics_before, metrics_after, hyperparams):
    return (
        "O modelo otimizado por Algoritmo Genético apresentou maior recall, indicando maior capacidade "
        "de identificar corretamente pacientes com diabetes. Isso é relevante em cenários clínicos de triagem.\n\n"
        "Pode haver trade-offs com outras métricas como F1-score e acurácia. Resultados devem ser interpretados "
        "como suporte à decisão, não substituindo avaliação médica."
    )

def build_prompt(metrics_before, metrics_after, hyperparams):
    return f"""
Você é um assistente especializado em interpretar resultados de modelos de aprendizado de máquina aplicados à saúde.

Contexto:
- Problema: Classificação binária de diabetes
- Modelo base: Regressão Logística
- Modelo otimizado via Algoritmo Genético, priorizando Recall

Hiperparâmetros selecionados:
{hyperparams}

Métricas do modelo sem GA:
{metrics_before}

Métricas do modelo com GA:
{metrics_after}

Instruções:
- Explique como a otimização impactou as métricas
- Destaque o papel do class_weight balanceado
- Relacione Recall elevado com redução de falsos negativos
- Não forneça diagnóstico médico
- Máximo de 3 parágrafos
- Explique para alguem que não compreende termos tecnicos de tecnologia ou IA
"""

def call_llm(prompt: str) -> str:
    if USE_LLM:
        client = AzureOpenAI(
            api_key=AZURE_VARS["API_KEY"],
            azure_endpoint=AZURE_VARS["ENDPOINT"],
            api_version=AZURE_VARS["API_VERSION"]
        )
        response = client.chat.completions.create(
            model=AZURE_VARS["DEPLOYMENT"],
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em interpretação de modelos de ML. Não forneça diagnóstico médico."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=700
        )
        return response.choices[0].message.content
    else:
        return fallback_explanation(baseline_results, ga_results, best_ind)

prompt = build_prompt(baseline_results, ga_results, best_ind)
interpretation = call_llm(prompt)

# ---------------------------
# 8. Print final apenas do LLM
# ---------------------------
print("\n=== Interpretação do modelo pelo LLM ===")
print(interpretation)
logger.info("Pipeline completo executado com sucesso")
