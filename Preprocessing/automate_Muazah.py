import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # type: ignore

def preprocess_data(df, target_col):
    df_copy = df.copy()

    # Label Encoding untuk semua kolom kategorikal (tipe object)
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_copy[col] = encoder.fit_transform(df_copy[col].astype(str))

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

    X_train_resampled.to_csv("Preprocessing/Dataset/X_train.csv", index=False)
    X_test.to_csv("Preprocessing/Dataset/X_test.csv", index=False)
    y_train_resampled.to_csv("Preprocessing/Dataset/y_train.csv", index=False)
    y_test.to_csv("Preprocessing/Dataset/y_test.csv", index=False)
    
    return X_train_resampled, X_test, y_train_resampled, y_test
