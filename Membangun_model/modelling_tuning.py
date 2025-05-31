import pandas as pd
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

mlflow.set_tracking_uri("https://dagshub.com/muazahalwyh/Experimen_SMSML_Muazah.mlflow")
mlflow.set_experiment("Experiment Customer Churn")

# Load data hasil preprocessing
X_train = pd.read_csv("Membangun_model/Dataset/X_train_resampled.csv")
y_train = pd.read_csv("Membangun_model/Dataset/y_train_resampled.csv").squeeze()
X_test = pd.read_csv("Membangun_model/Dataset/X_test.csv")
y_test = pd.read_csv("Membangun_model/Dataset/y_test.csv").squeeze()

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

            # Log parameter dan metric manual
            # Hitungan metrik
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log metrik manual
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
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
