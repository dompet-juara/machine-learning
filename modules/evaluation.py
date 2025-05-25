# modules/evaluation.py

"""
Modul Evaluasi Model Machine Learning.

Modul ini menyediakan serangkaian fungsi untuk mengevaluasi performa
model machine learning yang telah dilatih, khususnya model TensorFlow/Keras.
Fungsi-fungsi ini mencakup:
- Pemuatan model yang telah disimpan.
- Visualisasi riwayat pelatihan (akurasi dan loss).
- Pembuatan laporan klasifikasi dan confusion matrix.
- Evaluasi metrik standar pada data uji.
- Pengujian model dengan data input manual untuk demonstrasi atau validasi cepat.

Modul ini bergantung pada TensorFlow, Scikit-learn, Matplotlib, Seaborn,
serta modul lokal `config` untuk path default dan `utils` untuk utilitas
seperti penyimpanan plot.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Any, Optional

from . import config
from . import utils


def load_trained_model(
    model_path: str = config.BEST_MODEL_PATH,
) -> Optional[tf.keras.Model]:
    """Memuat model Keras yang telah dilatih dari file .keras.

    Args:
        model_path (str, optional): Path ke file model (.keras) yang disimpan.
            Defaultnya adalah `config.BEST_MODEL_PATH`.

    Returns:
        Optional[tf.keras.Model]: Objek model Keras yang dimuat jika berhasil,
                                   atau None jika terjadi kesalahan saat memuat.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model berhasil dimuat dari: {model_path}")
        return model
    except Exception as e:
        print(f"Error saat memuat model dari '{model_path}': {e}")
        return None


def plot_training_history(history: Optional[tf.keras.callbacks.History]) -> None:
    """Memplot dan menyimpan riwayat akurasi dan loss dari proses pelatihan model.

    Fungsi ini akan membuat dua plot: satu untuk akurasi (training vs validation)
    dan satu lagi untuk loss (training vs validation) selama epoch pelatihan.
    Plot akan disimpan menggunakan `utils.save_plot()`.

    Args:
        history (Optional[tf.keras.callbacks.History]): Objek History yang dikembalikan
            oleh `model.fit()`. Jika None atau tidak valid, fungsi akan keluar.

    Returns:
        None
    """
    if history is None or not hasattr(history, "history") or not history.history:
        print("Riwayat pelatihan tidak valid atau tidak tersedia untuk diplot.")
        return

    metrics_to_plot = [
        ("accuracy", "val_accuracy", "Akurasi"),
        ("loss", "val_loss", "Loss"),
    ]

    for train_metric_key, val_metric_key, title_prefix in metrics_to_plot:
        if (
            train_metric_key not in history.history
            or val_metric_key not in history.history
        ):
            print(
                f"Metrik '{train_metric_key}' atau '{val_metric_key}' tidak ditemukan dalam riwayat pelatihan."
            )
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            history.history[train_metric_key],
            label=f"Training {title_prefix}",
            marker="o",
            linestyle="-",
        )
        ax.plot(
            history.history[val_metric_key],
            label=f"Validation {title_prefix}",
            marker="x",
            linestyle="--",
        )
        ax.set_title(f"Riwayat {title_prefix} Model Selama Pelatihan")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title_prefix)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        filename = f'training_{title_prefix.lower().replace(" ", "_")}_history.png'
        utils.save_plot(fig, filename)
        plt.show()
        plt.close(fig)


def generate_classification_report_and_cm(
    model: tf.keras.Model,
    X_test: pd.DataFrame,
    y_test_labels_encoded: np.ndarray,
    class_names: List[str],
) -> None:
    """Menghasilkan, mencetak laporan klasifikasi, dan memplot/menyimpan confusion matrix.

    Args:
        model (tf.keras.Model): Model Keras yang telah dilatih dan akan dievaluasi.
        X_test (pd.DataFrame): Fitur-fitur dari data uji.
        y_test_labels_encoded (np.ndarray): Label target sebenarnya dari data uji,
                                            dalam format integer (hasil label encoding).
        class_names (List[str]): Daftar nama kelas asli (misalnya, ['boros', 'hemat']).

    Returns:
        None
    """
    print("\n" + "=" * 15 + " LAPORAN KLASIFIKASI & CONFUSION MATRIX " + "=" * 15)
    y_pred_proba = model.predict(X_test)
    y_pred_classes_encoded = np.argmax(y_pred_proba, axis=1)

    print("\nLaporan Klasifikasi Model:")
    try:
        report = classification_report(
            y_test_labels_encoded,
            y_pred_classes_encoded,
            target_names=class_names,
            zero_division=0,
        )
        print(report)
    except ValueError as e:
        print(f"Error saat membuat laporan klasifikasi: {e}")
        print(
            "Pastikan 'class_names' sesuai dengan label yang ada di 'y_test_labels_encoded'."
        )
        report = classification_report(
            y_test_labels_encoded, y_pred_classes_encoded, zero_division=0
        )
        print(report)

    cm = confusion_matrix(y_test_labels_encoded, y_pred_classes_encoded)
    figsize_x = max(7, len(class_names) * 1.2)
    figsize_y = max(5, len(class_names) * 0.9)

    fig_cm, ax_cm = plt.subplots(figsize=(figsize_x, figsize_y))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 10},
    )
    ax_cm.set_title("Confusion Matrix", fontsize=14)
    ax_cm.set_xlabel("Predicted Label", fontsize=12)
    ax_cm.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    utils.save_plot(fig_cm, "confusion_matrix.png")
    plt.show()
    plt.close(fig_cm)


def evaluate_model(
    model: Optional[tf.keras.Model],
    X_test: pd.DataFrame,
    y_test_one_hot: np.ndarray,
    y_test_labels_encoded: np.ndarray,
    class_names: List[str],
    history: Optional[tf.keras.callbacks.History],
) -> None:
    """Mengevaluasi model machine learning secara komprehensif.

    Fungsi ini melakukan:
    1. Evaluasi loss dan akurasi dasar pada data uji menggunakan `model.evaluate()`.
    2. Menghasilkan dan menampilkan laporan klasifikasi serta confusion matrix.
    3. Memplot riwayat pelatihan (akurasi dan loss).

    Args:
        model (Optional[tf.keras.Model]): Model Keras yang telah dilatih. Jika None, evaluasi dibatalkan.
        X_test (pd.DataFrame): Fitur-fitur dari data uji.
        y_test_one_hot (np.ndarray): Label target sebenarnya dari data uji,
                                     dalam format one-hot encoding (untuk `model.evaluate()`).
        y_test_labels_encoded (np.ndarray): Label target sebenarnya dari data uji,
                                            dalam format integer (untuk scikit-learn metrics).
        class_names (List[str]): Daftar nama kelas asli.
        history (Optional[tf.keras.callbacks.History]): Objek History dari `model.fit()`.

    Returns:
        None
    """
    if model is None:
        print("Model tidak tersedia (None). Proses evaluasi dibatalkan.")
        return

    print("\n" + "=" * 20 + " MEMULAI EVALUASI MODEL " + "=" * 20)

    # 1. Evaluasi dasar (loss, accuracy) pada data uji
    print("\nEvaluasi Performa Model pada Data Uji:")
    try:
        loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
        print(f"  - Akurasi Model : {accuracy:.4f} ({(accuracy*100):.2f}%)")
        print(f"  - Loss Model    : {loss:.4f}")
    except Exception as e:
        print(f"  Error saat `model.evaluate()`: {e}")

    # 2. Laporan Klasifikasi dan Confusion Matrix
    generate_classification_report_and_cm(
        model, X_test, y_test_labels_encoded, class_names
    )

    # 3. Visualisasi Riwayat Pelatihan
    print("\nVisualisasi Riwayat Pelatihan:")
    plot_training_history(history)

    print("\n" + "=" * 20 + " EVALUASI MODEL SELESAI " + "=" * 20)


def test_with_manual_data(
    model_path: str,
    scaler: Optional[StandardScaler],
    label_encoder: Optional[LabelEncoder],
    feature_names: List[str],
    class_names: List[str],
) -> None:
    """Melakukan pengujian model dengan data sampel yang dibuat secara manual.

    Proses ini mencakup:
    1. Membuat beberapa sampel data manual.
    2. Memuat model yang telah dilatih dari `model_path`.
    3. Melakukan pra-pemrosesan (scaling) pada data manual menggunakan `scaler` yang telah di-fit.
    4. Melakukan prediksi menggunakan model yang dimuat.
    5. Mengubah hasil prediksi dari format numerik (indeks kelas) kembali ke label kelas asli.
    6. Menampilkan data input manual beserta hasil prediksinya dan probabilitas kelas.

    Fungsi ini berguna untuk sanity check atau demonstrasi cepat kemampuan model
    pada input spesifik.

    Args:
        model_path (str): Path ke file model .keras yang telah disimpan.
        scaler (Optional[StandardScaler]): Objek StandardScaler yang telah di-fit pada data training.
                                           Jika None, scaling dilewati (tidak direkomendasikan).
        label_encoder (Optional[LabelEncoder]): Objek LabelEncoder yang telah di-fit.
                                                Jika None, prediksi akan berupa indeks kelas.
        feature_names (List[str]): Daftar nama fitur sesuai dengan urutan yang digunakan
                                   saat melatih model. Penting untuk pembuatan DataFrame manual.
        class_names (List[str]): Daftar nama kelas asli untuk interpretasi hasil.

    Returns:
        None
    """
    print("\n" + "=" * 20 + " TESTING DENGAN DATA BUATAN MANUAL " + "=" * 20)

    # 1. Contoh Data Buatan Manual
    # PENTING: Sesuaikan nilai dan jumlah sampel di bawah ini agar relevan
    # dengan domain masalah Anda dan struktur fitur yang digunakan model.
    # Pastikan urutan fitur dalam list di bawah ini SAMA PERSIS dengan `feature_names`.
    # Contoh untuk dataset keuangan (15 fitur numerik):
    # Fitur: 'Gaji', 'Tabungan Lama', 'Investasi', 'Pemasukan Lainnya', 'Bahan Pokok',
    # 'Protein & Gizi Tambahan', 'Tempat Tinggal', 'Sandang', 'Konsumsi Praktis',
    # 'Barang & Jasa Sekunder', 'Pengeluaran Tidak Esensial', 'Pajak', 'Asuransi',
    # 'Sosial & Budaya', 'Tabungan / Investasi'
    manual_data_samples = [
        # Skenario 1: Potensi "Hemat"
        [
            15000000,
            50000000,
            20000000,
            1000000,
            2000000,
            1000000,
            3000000,
            500000,
            500000,
            500000,
            200000,
            1500000,
            500000,
            300000,
            5000000,
        ],
        # Skenario 2: Potensi "Boros"
        [
            7000000,
            5000000,
            1000000,
            500000,
            3000000,
            1500000,
            2500000,
            1000000,
            1500000,
            2000000,
            1000000,
            500000,
            200000,
            100000,
            500000,
        ],
        # Skenario 3: Potensi "Normal"
        [
            10000000,
            20000000,
            5000000,
            500000,
            2500000,
            1200000,
            2000000,
            700000,
            800000,
            1000000,
            500000,
            1000000,
            300000,
            200000,
            2000000,
        ],
    ]
    if len(manual_data_samples[0]) != len(feature_names):
        print(
            f"Error: Jumlah nilai dalam sampel data manual ({len(manual_data_samples[0])}) "
            f"tidak cocok dengan jumlah feature_names ({len(feature_names)})."
        )
        print("Harap periksa 'manual_data_samples' dan 'feature_names'.")
        return

    df_manual_data = pd.DataFrame(manual_data_samples, columns=feature_names)
    print("\nData Buatan (Sebelum Pra-pemrosesan):")
    print(df_manual_data)

    # 2. Muat Model
    loaded_model = load_trained_model(model_path)
    if loaded_model is None:
        print("Gagal memuat model. Pengujian dengan data buatan dibatalkan.")
        return

    # 3. Lakukan Pra-pemrosesan (Scaling)
    if scaler is None:
        print(
            "Peringatan: Scaler tidak tersedia. Data manual tidak di-scale. Hasil mungkin tidak akurat."
        )
        manual_data_processed = df_manual_data.values
    else:
        try:
            manual_data_processed = scaler.transform(df_manual_data)
            df_manual_data_scaled = pd.DataFrame(
                manual_data_processed, columns=feature_names
            )
            print("\nData Buatan (Setelah Scaling):")
            print(df_manual_data_scaled.head())
        except Exception as e:
            print(f"Error saat melakukan scaling pada data manual: {e}")
            print(
                "Pastikan scaler telah di-fit dengan benar dan fitur data manual sesuai."
            )
            return

    # 4. Lakukan Prediksi
    try:
        predictions_proba = loaded_model.predict(manual_data_processed)
        predicted_indices = np.argmax(predictions_proba, axis=1)
    except Exception as e:
        print(f"Error saat melakukan prediksi pada data manual: {e}")
        return

    # 5. Ubah Hasil Prediksi ke Label Kelas Asli
    if label_encoder is None:
        print(
            "Peringatan: Label encoder tidak tersedia. Menampilkan indeks kelas sebagai prediksi."
        )
        predicted_labels = [f"Indeks Kelas: {idx}" for idx in predicted_indices]
    else:
        try:
            predicted_labels = label_encoder.inverse_transform(predicted_indices)
        except Exception as e:
            print(
                f"Error saat mengubah prediksi ke label asli dengan label_encoder: {e}"
            )
            predicted_labels = [
                f"Indeks Kelas: {idx} (Error LE)" for idx in predicted_indices
            ]

    # 6. Tampilkan Hasil
    print("\n--- Hasil Prediksi pada Data Buatan ---")
    results_df = df_manual_data.copy()
    results_df["Predicted_Tipe"] = predicted_labels

    for i, class_name in enumerate(class_names):
        if i < predictions_proba.shape[1]:
            results_df[f"Prob_{class_name}"] = predictions_proba[:, i]
        else:
            results_df[f"Prob_{class_name}"] = np.nan

    print(results_df.to_string())
    print("\n" + "=" * 20 + " TESTING DENGAN DATA BUATAN MANUAL SELESAI " + "=" * 20)
