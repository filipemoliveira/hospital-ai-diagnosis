# 1️⃣ Escolher uma imagem base com Python
FROM python:3.10-slim

# 2️⃣ Definir diretório de trabalho dentro do container
WORKDIR /app

# 3️⃣ Copiar arquivos de requirements e instalar dependências
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Copiar todo o código do projeto para o container
COPY . .

# 5️⃣ Expor a porta padrão do Jupyter (opcional, se for usar notebooks)
EXPOSE 8888

# 6️⃣ Comando padrão para rodar o script pipeline
CMD ["python", "src/run_pipeline.py"]
