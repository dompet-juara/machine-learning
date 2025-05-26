# modules/preprocessing.py

"""
Modul Pra-pemrosesan Data untuk Proyek Machine Learning.

Modul ini menyediakan fungsi-fungsi untuk mempersiapkan data mentah agar siap
digunakan dalam pelatihan model machine learning. Langkah-langkah pra-pemrosesan
yang dicakup meliputi:
1. Encoding variabel target kategorikal menjadi representasi numerik
   (Label Encoding dan One-Hot Encoding).
2. Pembagian dataset menjadi set pelatihan dan set pengujian.
3. Scaling fitur-fitur numerik menggunakan StandardScaler.
4. Penyimpanan objek scaler dan label encoder ke file pickle.

Fungsi utama `preprocess_data` mengorkestrasi semua langkah ini.
Modul ini bergantung pada pandas, numpy, scikit-learn, TensorFlow, pickle, os,
serta modul lokal `config` untuk parameter konfigurasi.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Any, Optional
import pickle
import os

from . import config


def encode_target_variable(
    y_series: pd.Series,
) -> Tuple[np.ndarray, LabelEncoder, List[str], int]:
    """Melakukan encoding pada variabel target kategorikal menggunakan LabelEncoder.

    Fungsi ini mengubah label target dari format string atau kategorikal menjadi
    representasi integer. Informasi mengenai kelas asli dan jumlah kelas juga
    diekstraksi.

    Args:
        y_series (pd.Series): Series pandas yang berisi variabel target
                               (sebelum di-encode).

    Returns:
        Tuple[np.ndarray, LabelEncoder, List[str], int]: Sebuah tuple berisi:
            - y_encoded (np.ndarray): Array NumPy dari target yang telah
              di-encode secara numerik (misalnya, 0, 1, 2, ...).
            - label_encoder (LabelEncoder): Objek LabelEncoder yang telah di-fit,
              dapat digunakan untuk `inverse_transform` dan disimpan.
            - class_names (List[str]): Daftar nama kelas unik dalam urutan
              sesuai dengan encoding numeriknya.
            - num_classes (int): Jumlah total kelas unik.
    """
    print("\n--- Encoding Variabel Target (Label Encoding) ---")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    print(f"Kelas target asli terdeteksi: {class_names}")
    print(f"Jumlah kelas unik: {num_classes}")
    print(f"Contoh 5 nilai target sebelum encoding: {y_series.iloc[:5].values}")
    print(f"Contoh 5 nilai target setelah LabelEncoding: {y_encoded[:5]}")

    return y_encoded, label_encoder, class_names, num_classes


def one_hot_encode_target(y_encoded: np.ndarray, num_classes: int) -> np.ndarray:
    """Mengonversi target yang sudah di-encode secara numerik menjadi format one-hot.

    Fungsi ini mengambil array target yang telah di-label-encode dan mengubahnya
    menjadi representasi biner (one-hot encoding), yang seringkali dibutuhkan
    untuk fungsi loss seperti 'categorical_crossentropy' pada model neural network.

    Args:
        y_encoded (np.ndarray): Array NumPy dari target yang telah di-encode
                                secara numerik (output dari `encode_target_variable`).
        num_classes (int): Jumlah kelas unik, digunakan untuk menentukan dimensi
                           output one-hot encoding.

    Returns:
        np.ndarray: Array NumPy dari target dalam format one-hot encoding.
                    Shape-nya akan menjadi (jumlah_sampel, num_classes).
    """
    print("\n--- One-Hot Encoding Variabel Target ---")
    y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    print(f"Shape target setelah One-Hot Encoding: {y_one_hot.shape}")
    print(f"Contoh 5 nilai target setelah One-Hot Encoding:\n{y_one_hot[:5]}")
    return y_one_hot


def scale_numerical_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Melakukan scaling pada fitur numerik menggunakan StandardScaler.

    StandardScaler menstandarisasi fitur dengan menghilangkan mean dan menskalakan
    ke unit varians. Scaler di-fit hanya pada data pelatihan (`X_train`) dan
    kemudian digunakan untuk mentransformasi baik `X_train` maupun `X_test`
    untuk mencegah kebocoran data (data leakage) dari set pengujian.

    Asumsi: Semua kolom dalam `X_train` dan `X_test` adalah fitur numerik
            yang memerlukan scaling.

    Args:
        X_train (pd.DataFrame): DataFrame yang berisi fitur-fitur numerik
                                dari set pelatihan.
        X_test (pd.DataFrame): DataFrame yang berisi fitur-fitur numerik
                               dari set pengujian.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]: Sebuah tuple berisi:
            - X_train_scaled (pd.DataFrame): DataFrame fitur pelatihan yang
              telah di-scale.
            - X_test_scaled (pd.DataFrame): DataFrame fitur pengujian yang
              telah di-scale.
            - scaler (StandardScaler): Objek StandardScaler yang telah di-fit,
              dapat digunakan untuk mentransformasi data baru dan disimpan.
    """
    print("\n--- Scaling Fitur Numerik (StandardScaler) ---")
    numerical_cols = X_train.columns.tolist()  # Asumsi semua kolom adalah numerik
    scaler = StandardScaler()
    X_train_scaled_np = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled_np = scaler.transform(X_test[numerical_cols])

    X_train_scaled = pd.DataFrame(
        X_train_scaled_np, columns=numerical_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_np, columns=numerical_cols, index=X_test.index
    )

    print("Fitur numerik telah di-scale menggunakan StandardScaler.")
    print("Contoh 5 baris X_train setelah scaling:")
    print(X_train_scaled.head())

    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(
    df: Optional[pd.DataFrame],
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[StandardScaler],
    Optional[LabelEncoder],
    Optional[List[str]],
    Optional[int],
    Optional[np.ndarray],
]:
    """Melakukan pipeline pra-pemrosesan data secara lengkap.

    Langkah-langkah yang dilakukan:
    1. Memisahkan fitur (X) dan target (y) dari DataFrame input.
    2. Melakukan Label Encoding pada variabel target.
    3. Menyimpan objek LabelEncoder yang sudah di-fit ke file pickle.
    4. Melakukan One-Hot Encoding pada variabel target yang sudah di-label-encode.
    5. Membagi data menjadi set pelatihan dan pengujian (X_train, X_test, y_train, y_test).
       Pembagian dilakukan secara stratified untuk menjaga proporsi kelas.
    6. Melakukan scaling pada fitur-fitur numerik (X_train, X_test) menggunakan StandardScaler.
    7. Menyimpan objek StandardScaler yang sudah di-fit ke file pickle.

    Jika DataFrame input adalah None, fungsi akan mengembalikan tuple berisi None
    untuk semua output yang diharapkan.

    Args:
        df (Optional[pd.DataFrame]): DataFrame input yang berisi fitur dan target.
                                     Jika None, pra-pemrosesan dibatalkan.

    Returns:
        Tuple: Sebuah tuple berisi sembilan elemen:
            - X_train_scaled (Optional[pd.DataFrame]): Fitur pelatihan yang sudah di-scale.
            - X_test_scaled (Optional[pd.DataFrame]): Fitur pengujian yang sudah di-scale.
            - y_train_one_hot (Optional[np.ndarray]): Target pelatihan dalam format one-hot.
            - y_test_one_hot (Optional[np.ndarray]): Target pengujian dalam format one-hot.
            - scaler (Optional[StandardScaler]): Objek StandardScaler yang telah di-fit.
            - label_encoder (Optional[LabelEncoder]): Objek LabelEncoder yang telah di-fit.
            - class_names (Optional[List[str]]): Daftar nama kelas target.
            - num_classes (Optional[int]): Jumlah kelas target.
            - y_test_labels_encoded (Optional[np.ndarray]): Target pengujian dalam format
              label encoded (misalnya, untuk metrik sklearn).
            Jika input `df` adalah None, semua elemen dalam tuple akan menjadi None.
    """
    if df is None:
        print(
            "DataFrame input tidak tersedia (None). Proses pra-pemrosesan dibatalkan."
        )
        return (None,) * 9

    print("\n" + "=" * 20 + " MEMULAI PRA-PEMROSESAN DATA " + "=" * 20)

    try:
        os.makedirs(config.TOOLS_DIR, exist_ok=True)
        print(f"Direktori '{config.TOOLS_DIR}' siap digunakan.")
    except Exception as e:
        print(f"Gagal membuat direktori '{config.TOOLS_DIR}': {e}")
        return (None,) * 9

    # 1. Pisahkan Fitur dan Target
    if config.TARGET_COLUMN not in df.columns:
        print(
            f"Error: Kolom target '{config.TARGET_COLUMN}' tidak ditemukan dalam DataFrame."
        )
        return (None,) * 9
    X = df.drop(config.TARGET_COLUMN, axis=1)
    y = df[config.TARGET_COLUMN]
    print(f"Shape fitur (X) awal: {X.shape}")
    print(f"Shape target (y) awal: {y.shape}")

    # 2. Encoding Variabel Target
    y_encoded, label_encoder, class_names, num_classes = encode_target_variable(y)

    # 3. Simpan LabelEncoder
    if label_encoder:
        try:
            with open(config.LABEL_ENCODER_FILEPATH, "wb") as f:
                pickle.dump(label_encoder, f)
            print(
                f"Label encoder berhasil disimpan ke: {config.LABEL_ENCODER_FILEPATH}"
            )
        except Exception as e:
            print(f"Error saat menyimpan LabelEncoder: {e}")
            return (None,) * 9

    # 4. One-Hot Encoding Target
    y_one_hot = one_hot_encode_target(y_encoded, num_classes)

    # 5. Pembagian Data menjadi Training dan Test Set
    print("\n--- Pembagian Data (Train-Test Split) ---")
    (
        X_train,
        X_test,
        y_train_one_hot,
        y_test_one_hot,
        y_train_labels_encoded,
        y_test_labels_encoded,
    ) = train_test_split(
        X,
        y_one_hot,
        y_encoded,
        test_size=config.TEST_SPLIT_SIZE,
        random_state=config.SEED,
        stratify=y_encoded,
    )
    print(f"Shape X_train: {X_train.shape}, Shape X_test: {X_test.shape}")
    print(
        f"Shape y_train_one_hot: {y_train_one_hot.shape}, Shape y_test_one_hot: {y_test_one_hot.shape}"
    )
    print(
        f"Shape y_train_labels_encoded: {y_train_labels_encoded.shape}, Shape y_test_labels_encoded: {y_test_labels_encoded.shape}"
    )

    # 6. Scaling Fitur Numerik
    X_train_scaled, X_test_scaled, scaler = scale_numerical_features(X_train, X_test)

    # 7. Simpan StandardScaler
    if scaler:
        try:
            with open(config.SCALER_FILEPATH, "wb") as f:
                pickle.dump(scaler, f)
            print(f"Scaler berhasil disimpan ke: {config.SCALER_FILEPATH}")
        except Exception as e:
            print(f"Error saat menyimpan Scaler: {e}")
            return (None,) * 9

    print("\n" + "=" * 20 + " PRA-PEMROSESAN DATA SELESAI " + "=" * 20)
    return (
        X_train_scaled,
        X_test_scaled,
        y_train_one_hot,
        y_test_one_hot,
        scaler,
        label_encoder,
        class_names,
        num_classes,
        y_test_labels_encoded,
    )
