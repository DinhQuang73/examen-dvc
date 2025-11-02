import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
model = joblib.load("models/model.pkl")

preds = model.predict(X_test)
pd.DataFrame({"prediction": preds}).to_csv("data/predictions.csv", index=False)

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

os.makedirs("metrics", exist_ok=True)
with open("metrics/scores.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f)

