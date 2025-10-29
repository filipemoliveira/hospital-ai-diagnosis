import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar dataset processado
dataset = pd.read_csv("../dados/processados/processado_diabetes_script.csv")

# Separar features e target
X = dataset.drop("Outcome", axis=1)
y = dataset["Outcome"]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Definir modelos
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42)
}

# Treinar modelos
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))

# Feature Importance / Coeficientes (somente valores, sem plot)
feature_importances = {}
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances[name] = model.coef_[0]

# SHAP apenas para Logistic Regression (valores)
import shap as sh
final_model = models['Logistic Regression']
explainer = sh.Explainer(final_model, X_train_res)
sh_values = explainer(X_test)
