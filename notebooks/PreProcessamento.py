import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Carregar dataset
dataset = pd.read_csv("../dados/Raw/diabetes.csv")

# Substituir zeros por NaN em colunas específicas
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
dataset[cols_with_zeros] = dataset[cols_with_zeros].replace(0, np.nan)

# Separar features e target
X = dataset.drop("Outcome", axis=1)
y = dataset["Outcome"]

# Imputação KNN
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Dataset final
final_dataset = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
final_dataset.to_csv('../dados/processados/processado_diabetes_script.csv', index=False)
