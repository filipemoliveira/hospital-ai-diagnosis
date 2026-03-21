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


### 🧠 Fase 3 — Assistente Clínico com LLM

- Fine-tuning de LLM com dados médicos (dataset público e dados sintéticos)
- Construção de assistente clínico com LangChain
- Orquestração de fluxo clínico com LangGraph
- Contextualização com dados do paciente
- Implementação de segurança e governança:
  - Logging estruturado e auditoria
  - Explainability (justificativa das respostas)
  - Validação humana no fluxo
- Avaliação do modelo com cenários clínicos simulados

---

## 📂 Estrutura do Projeto

- `dados/` → datasets utilizados (ou links se forem muito grandes)  
- `notebooks/` → notebooks Jupyter organizados por fase  
- `src/` → scripts Python para execução dos pipelines  
- `resultados/` → métricas e outputs dos modelos  
- `artefatos/` → imagens, diagramas e materiais auxiliares  
- `logs/` → logs estruturados do sistema clínico  
- `requirements.txt` → dependências do projeto  
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
### ▶️ Execução Automatizada (Ajustado para o Pipeline da Fase 3)
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


### Fase 3 — Assistente Clínico

Execute os notebooks em ordem:

- `01_llm/01_fine_tuning_processamento.ipynb`  
→ Preparação e curadoria dos dados

- `01_llm/02_fine_tuning_azure.ipynb`  
→ Execução do fine-tuning

- `02_assistente/assistente_langchain.ipynb`  
→ Construção do assistente clínico

- `03_fluxo_langgraph/fluxo_langgraph.ipynb`  
→ Orquestração do fluxo clínico

- `05_avaliacao_modelo/avaliacao_modelo.ipynb`  
→ Avaliação do modelo com cenários clínicos simulados

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


## 🧩 Arquitetura da Fase 3

A solução da Fase 3 foi estruturada em etapas:

1. Preparação dos dados médicos  
2. Fine-tuning da LLM  
3. Construção do assistente com LangChain  
4. Orquestração do fluxo clínico com LangGraph  
5. Validação humana e geração de resposta final  
6. Registro de logs para auditoria  

Os diagramas estão disponíveis em:

- `artefatos/Fase 3/Diagrama Completo.png`  
- `artefatos/Fase 3/Diagrama Video.png`  

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

#### IA Generativa e Orquestração

- LangChain  
- LangGraph  
- Azure OpenAI / LLMs  

---

## 🔐 Segurança e Governança

A solução implementa mecanismos para garantir uso seguro:

- Não prescrição automática de medicamentos  
- Validação humana obrigatória antes da resposta final  
- Logging estruturado de todas as etapas do fluxo  
- Rastreabilidade completa (input, output e decisões)  
- Explainability com justificativa baseada nos dados do paciente  

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

