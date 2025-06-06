import pandas as pd
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data hasil preprocessing
X_train = pd.read_csv("Membangun_model/Dataset/X_train_resampled.csv")
y_train = pd.read_csv("Membangun_model/Dataset/y_train_resampled.csv").squeeze()
X_test = pd.read_csv("Membangun_model/Dataset/X_test.csv")
y_test = pd.read_csv("Membangun_model/Dataset/y_test.csv").squeeze()

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("Experiment Customer Churn")

# Ambil contoh input untuk log model (harus DataFrame)
input_example = X_train.iloc[0:5]

with mlflow.start_run():
    # Set parameter model
    n_estimators = 505
    max_depth = 37
    
    # Aktifkan autolog untuk otomatis logging param, metric, model
    mlflow.autolog() 

    # model Random Forest
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Prediksi untuk evaluasi
    y_pred = model.predict(X_test)

    # Hitungan metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrik manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan model dengan input_example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    print("Model training dan logging selesai.")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")