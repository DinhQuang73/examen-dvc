import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Chargement des données
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Sélection uniquement des colonnes numériques
X_train_num = X_train.select_dtypes(include=["number"])
X_test_num = X_test.select_dtypes(include=["number"])

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)

# Sauvegarde
pd.DataFrame(X_train_scaled, columns=X_train_num.columns).to_csv("data/processed/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test_num.columns).to_csv("data/processed/X_test_scaled.csv", index=False)
