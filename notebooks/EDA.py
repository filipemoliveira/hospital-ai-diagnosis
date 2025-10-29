# Exploração de dados

import pandas as pd
import numpy as np

# Carregar dataset
dataset = pd.read_csv("../dados/Raw/diabetes.csv")

# Identificar valores zero inválidos
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zeros_count = {col: (dataset[col] == 0).sum() for col in zero_cols}

# Estatísticas básicas
dataset_info = dataset.info()
dataset_desc = dataset.describe()

# Distribuição do target
class_counts = dataset['Outcome'].value_counts(normalize=True)

# Correlação entre variáveis
corr_matrix = dataset.corr()
