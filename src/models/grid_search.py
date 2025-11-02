import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20]
}

grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(grid.best_params_, "models/best_params.pkl")

