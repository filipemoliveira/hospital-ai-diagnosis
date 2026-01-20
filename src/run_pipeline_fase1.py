"""
Pipeline completo para previsão de diabetes:
1. Pré-processamento
2. Treino e avaliação de modelos
3. Feature Importance e SHAP
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap as sh

# ------------------------------------------
# 1. Carregar dataset bruto
# ------------------------------------------
dataset = pd.read_csv("../dados/Raw/diabetes.csv")

# ------------------------------------------
# 2. Pré-processamento
# ------------------------------------------
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

# Criar dataset final
final_dataset = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
final_dataset.to_csv('../dados/processados/processado_diabetes_script.csv', index=False)
print("✅ Pré-processamento concluído e dataset salvo.")

# ------------------------------------------
# 3. Divisão treino/teste e balanceamento SMOTE
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ------------------------------------------
# 4. Treinar modelos
# ------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42)
}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)

# ------------------------------------------
# 5. Avaliar modelos
# ------------------------------------------
model_results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    model_results[name] = report

print("\n--- Resultados dos modelos ---")
for name, metrics in model_results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    if '1' in metrics:  # classe positiva
        print(f"Recall (Classe 1): {metrics['1']['recall']:.2f}")
        print(f"Precision (Classe 1): {metrics['1']['precision']:.2f}")
        print(f"F1-score (Classe 1): {metrics['1']['f1-score']:.2f}")

# # 6. Feature Importance
# for name, model in models.items():
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#     elif hasattr(model, 'coef_'):
#         importances = model.coef_[0]
#     else:
#         continue

#     feat_imp = pd.DataFrame({
#         'Feature': X_train.columns,
#         'Importance': importances
#     }).sort_values(by='Importance', ascending=False)
    
#     print(f"\nFeature Importance - {name}")
#     print(feat_imp.head(10))

# # 7. SHAP
# final_model = models['Logistic Regression']
# explainer = sh.Explainer(final_model, X_train_res)
# sh_values = explainer(X_test)
# sh.summary_plot(sh_values, X_test, plot_type="bar")  # pode ser comentado se quiser sem gráfico