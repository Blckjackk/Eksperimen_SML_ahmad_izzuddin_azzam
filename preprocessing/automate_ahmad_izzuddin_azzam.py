import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path: str = 'obesity_raw/obesity_raw.csv') -> pd.DataFrame:
    """
    Memuat raw dataset dari file CSV lokal.
    Jika file tidak ditemukan di folder obesity_raw, fallback ke root directory atau ucimlrepo.
    """
    print(f"[1/6] Memuat dataset dari: {file_path}...")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    elif os.path.exists('ObesityDataSet_raw_and_data_sinthetic.csv'):
        df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    else:
        from ucimlrepo import fetch_ucirepo
        print("  Downloading dataset dari UCI Repository (ID: 544)...")
        obesity = fetch_ucirepo(id=544)
        X = obesity.data.features
        y = obesity.data.targets
        df = pd.concat([X, y], axis=1)
        
    print(f"  Berhasil memuat {len(df)} baris data dan {len(df.columns)} kolom.")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan missing values dan duplikat dari DataFrame.
    """
    print("[2/6] Membersihkan data (missing values & duplikat)...")
    initial_count = len(df)
    
    # Cek & handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df = df.dropna()
        print(f"  Ditemukan {missing_count} missing values. Baris telah dihapus.")
    else:
        print("  Tidak ada missing values ditemukan.")
        
    # Hapus duplikat
    df = df.drop_duplicates()
    dedup_count = initial_count - len(df)
    print(f"  Dihapus {dedup_count} baris duplikat. Sisa total data: {len(df)} baris.")
    
    return df

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Meng-encode fitur biner (LabelEncoder), nominal (One-Hot Encoding), dan target.
    Returns X (features) dan y (target).
    """
    print("[3/6] Meng-encode fitur kategorikal & target...")
    df = df.copy()
    
    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    nominal_cols = ['CAEC', 'CALC', 'MTRANS']
    target_col = 'NObeyesdad'
    
    # Label Encoding fitur biner
    le_bin = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le_bin.fit_transform(df[col])
            
    # One-Hot Encoding fitur nominal
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # Label Encoding target
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"  Encoding selesai. Jumlah fitur akhir (X): {X.shape[1]} kolom.")
    return X, y

def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan standarisasi (StandardScaler) pada kolom numerik kontinu.
    """
    print("[4/6] Melakukan standarisasi (StandardScaler) fitur numerik...")
    X_scaled = X.copy()
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    scaler = StandardScaler()
    X_scaled[numerical_cols] = scaler.fit_transform(X_scaled[numerical_cols])
    
    print("  Standarisasi fitur numerik berhasil.")
    return X_scaled

def split_and_save_data(X: pd.DataFrame, y: pd.Series, output_dir: str = 'preprocessing/obesity_preprocessing') -> None:
    """
    Melakukan train-test split (80:20) dan menyimpan hasilnya ke folder output.
    """
    print(f"[5/6] Melakukan Train-Test Split (80:20) & menyimpan ke folder '{output_dir}'...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"  [BERHASIL] File tersimpan:")
    print(f"   - {output_dir}/X_train.csv ({X_train.shape})")
    print(f"   - {output_dir}/X_test.csv ({X_test.shape})")
    print(f"   - {output_dir}/y_train.csv ({y_train.shape})")
    print(f"   - {output_dir}/y_test.csv ({y_test.shape})")

def main():
    print("=== MULAI PROSES OTOMATISASI PREPROCESSING DATASET OBESITY ===")
    df_raw = load_data('obesity_raw/obesity_raw.csv')
    df_clean = clean_data(df_raw)
    X, y = encode_features(df_clean)
    X_scaled = scale_features(X)
    split_and_save_data(X_scaled, y, output_dir='preprocessing/obesity_preprocessing')
    print("=== PROSES OTOMATISASI SELESAI DENGAN SUKSES ===")

if __name__ == '__main__':
    main()
