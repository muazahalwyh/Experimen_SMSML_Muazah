import pandas as pd
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

mlflow.set_tracking_uri("https://dagshub.com/muazahalwyh/my-first-repo.mlflow")
mlflow.set_experiment("Experiment Customer Churn")

# Load data hasil preprocessing
X_train = pd.read_csv("Dataset/X_train_resampled_20250530_004103.csv")
y_train = pd.read_csv("Dataset/y_train_resampled_20250530_004103.csv").squeeze()
X_test = pd.read_csv("Dataset/X_test_20250530_004103.csv")
y_test = pd.read_csv("Dataset/y_test_20250530_004103.csv").squeeze()

input_example = X_train.iloc[0:5]

# Range hyperparameter
n_estimators_range = np.linspace(100, 1000, 5, dtype=int)
max_depth_range = np.linspace(5, 50, 5, dtype=int)

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"Tuning_{n_estimators}_{max_depth}"):
            
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(X_test, y_test)

            # Log parameter dan metric manual
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)

            # Simpan model terbaik
            if acc > best_accuracy:
                best_accuracy = acc
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="best_model",
                    input_example=input_example
                )

print("Tuning selesai.")
print(f"Model terbaik: {best_params}, Akurasi: {best_accuracy:.4f}")
