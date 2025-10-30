# 🏥 Diagnóstico Hospitalar com IA

# 🩺 Diagnóstico de Diabetes — Projeto de Machine Learning

Este projeto tem como objetivo aplicar técnicas de aprendizado de máquina para prever a presença de **diabetes** com base em variáveis clínicas.

Foram desenvolvidos **notebooks Jupyter** para análise detalhada e discussão dos resultados, e **scripts Python** que permitem a execução completa do pipeline de forma automatizada.

---

## 📂 Estrutura do Projeto

- `dados/` → datasets utilizados (ou links se forem muito grandes)  
- `notebooks/` → notebooks Jupyter para experimentos e análises  
- `src/` → scripts Python (pré-processamento, treinamento, avaliação)  
- `resultados/` → gráficos, capturas de tela e métricas de avaliação  
- `requisitos.txt` → dependências do projeto  
- `Dockerfile` → configuração do container  

---

## ⚙️ Como Configurar o Ambiente

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

## 🚀 Como Executar

### 1️⃣ Executar o projeto completo de forma automatica (script Python)
#### O script abaixo executa as 3 etapas (EDA - Pré-Processamento - Modelagem) em sequência:
```bash
python src/pipeline.py
```

### 2️⃣ Executar o projeto manualmente (via notebooks)
#### Para ver os resultados passo a passo e a analise completa:
##### - Inicie o Jupyter
```bash
jupyter notebook
```   
##### - Abra e execute os notebooks em ordem:
1- EDA.ipynb  
2- PreProcessamento.ipynb  
3- Modelagem.ipynb

##### 💡 O dataset gerado no notebook de pré-processamento é utilizado no notebook de modelagem  

### 3️⃣ Executar via Docker (Container)
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

Os resultados incluindo graficos e metricas de avaliação estão descritos no arquivo pdf.  
Uma copia do arquivo pode ser encontrada na pasta resultados.  

---

## 👨‍💻 Modelos e Técnicas Utilizadas

- Regressão Logística
- Árvore de Decisão
- Random Forest
- Normalização e Padronização
- Balanceamento de Classes (SMOTE)
- Avaliação: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Interpretação: Feature Importance e SHAP Values

---

## 📚 Requisitos

As bibliotecas necessárias estão listadas em requirements.txt.

---

## 🧠 Observação Final

Este projeto foi desenvolvido para fins acadêmicos, com foco em:
- Clareza e estruturação do código
- Reprodutibilidade do pipeline
- Comparação de algoritmos de classificação
- Documentação técnica e visual (notebooks e relatório PDF)

---

## 👤 Autor
Filipe Mendes  
🌐 GitHub: https://github.com/filipemoliveira  
🔗 LinkedIn: https://www.linkedin.com/in/filipecrm  

