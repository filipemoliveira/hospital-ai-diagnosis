# 🏥 Diagnóstico Hospitalar com IA
# 🩺 Predição de Diabetes com Machine Learning, Otimização e LLMs

Este projeto tem como objetivo aplicar técnicas de Machine Learning para prever a presença de **diabetes** a partir de variáveis clínicas, evoluindo o modelo base com otimização por **Algoritmos Genéticos** e interpretação de resultados com **Large Language Models (LLMs)**.

O trabalho está organizado em duas fases, seguindo uma abordagem incremental e acadêmica.  

## 🔬 Visão Geral das Fases
### ✅ Fase 1 — Modelagem Base

- Análise Exploratória dos Dados (EDA)
- Pré-processamento e balanceamento de classes
- Treinamento de modelos clássicos de classificação
- Avaliação com métricas tradicionais
- Geração de resultados e relatório técnico

### 🚀 Fase 2 — Otimização e Interpretabilidade (Projeto 1)

- Otimização de hiperparâmetros via Algoritmos Genéticos
- Priorização de métricas clínicas (ex: Recall)
- Comparação entre modelo base e modelo otimizado
- Registro estruturado de métricas e logs
- Integração com LLMs para interpretação automática dos resultados

---

## 📂 Estrutura do Projeto

- `dados/` → datasets utilizados (ou links se forem muito grandes)  
- `notebooks/` → notebooks Jupyter para experimentos e análises  
- `src/` → scripts Python (pré-processamento, treinamento, avaliação)  
- `resultados/` → gráficos, capturas de tela e métricas de avaliação  
- `requisitos.txt` → dependências do projeto  
- `Dockerfile` → configuração do container  

---

## ⚙️ Configuração do Ambiente
Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

---

## Instale as dependências
```bash
pip install -r requirements.txt
```
---

## 🚀 Como Executar o Projeto
### ▶️ Execução Automatizada (Pipeline)

Executa a Fase 1 de ponta a ponta:
```bash
python src/pipeline.py
```
---

## 📓 Execução Manual via Notebooks
### Fase 1 — Modelagem Base

Execute os notebooks em ordem:
- EDA.ipynb
- PreProcessamento.ipynb
- Modelagem.ipynb

💡 O dataset gerado no pré-processamento é reutilizado na modelagem.

### Fase 2 — Otimização e LLM

- Apresentacao_Definicao_Estrutura.ipynb
→ Contextualização teórica e definição das etapas

- Implementacao_GA.ipynb
→ Otimização dos hiperparâmetros com Algoritmo Genético

- Integracao_LLM.ipynb
→ Interpretação automática dos resultados com LLM

---

### 🐳 Executar via Docker (Container)
#### Caso não queira instalar nada localmente, você pode rodar todo o projeto dentro de um container Docker:
1- Construir a imagem  
```bash
docker build -t diabetes-ml .
```
2- Executar o container  
```bash
docker run --rm -it diabetes-ml
```   

---

## 📊 Resultados

#### Métricas da Fase 1 e Fase 2 estão salvas em:

- artefatos/
- resultados/

#### Comparações entre modelo base e otimizado incluem:

- Recall
- F1-score
- Acurácia

Gráficos e análises completas estão documentados no relatório PDF.

---

### 🤖 Técnicas e Tecnologias Utilizadas
#### Modelagem

- Regressão Logística
- Árvores de Decisão
- Random Forest

#### Otimização

- Algoritmos Genéticos para ajuste de hiperparâmetros
- Função fitness priorizando Recall

#### Avaliação
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

#### Interpretabilidade
- Feature Importance
- SHAP Values
- Interpretação textual via LLM

---

### 🧠 Observações Acadêmicas

Este projeto foi desenvolvido com foco em:

- Evolução incremental do modelo
- Reprodutibilidade
- Clareza metodológica
- Separação entre teoria, experimentação e análise
- Boas práticas de projetos acadêmicos em IA e ML

---

## 👤 Autor
Filipe Mendes  
🌐 GitHub: https://github.com/filipemoliveira  
🔗 LinkedIn: https://www.linkedin.com/in/filipecrm  

