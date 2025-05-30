import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # type: ignore
import os
import joblib
import shutil
from datetime import datetime

def preprocess_data(df, target_col):    
    df_copy = df.copy()
    
    # Timestamp untuk versi file/folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Label Encoding untuk semua kolom kategorikal (tipe object)
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_copy[col] = encoder.fit_transform(df_copy[col].astype(str))
        encoders[col] = encoder

    # Standarisasi untuk kolom numerik (kecuali target)
    numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns.drop(target_col)
    scaler = StandardScaler()
    df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

    # Pisahkan fitur dan target
    X = df_copy.drop(target_col, axis=1)
    y = df_copy[target_col]

    # Split data train-test stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Oversampling SMOTE di data train
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

   # Buat ulang folder output preprocessing
    preprocessing_dir = f"Preprocessing/Dataset_{timestamp}"
    joblib_dir = "Preprocessing/Joblib"
    if os.path.exists(preprocessing_dir):
        shutil.rmtree(preprocessing_dir)
    os.makedirs(preprocessing_dir)

    if os.path.exists(joblib_dir):
        shutil.rmtree(joblib_dir)
    os.makedirs(joblib_dir)

    # Simpan file dataset di preprocessing
    X_train_resampled.to_csv(f"{preprocessing_dir}/Dataset/X_train_resampled.csv", index=False)
    X_test.to_csv(f"{preprocessing_dir}/Dataset/X_test.csv", index=False)
    y_train_resampled.to_csv(f"{preprocessing_dir}/Dataset/y_train_resampled.csv", index=False)
    y_test.to_csv(f"{preprocessing_dir}/Dataset/y_test.csv", index=False)

    # Simpan artefak/joblib di preprocessing
    joblib.dump(encoders, f"{joblib_dir}/encoders_{timestamp}.joblib")
    joblib.dump(scaler, f"{joblib_dir}/scaler_{timestamp}.joblib")
    joblib.dump(smote, f"{joblib_dir}/smote_{timestamp}.joblib")

    # Salin ke folder Membangun_model/Dataset
    model_dataset_dir = f"Membangun_model/Dataset_{timestamp}"
    if os.path.exists(model_dataset_dir):
        shutil.rmtree(model_dataset_dir)
    os.makedirs(model_dataset_dir)

    for filename in os.listdir(preprocessing_dir):
        src_file = os.path.join(preprocessing_dir, filename)
        dst_file = os.path.join(model_dataset_dir, filename)
        shutil.copyfile(src_file, dst_file)

    print(f"Preprocessing selesai. Output disimpan di:\n- {preprocessing_dir}\n- {model_dataset_dir}")

    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    df_raw = pd.read_csv("Dataset/data_bersih_preprocessing.csv")
    preprocess_data(df_raw, target_col="Churn Label") 
