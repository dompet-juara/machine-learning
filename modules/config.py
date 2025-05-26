# modules/config.py

"""
File Konfigurasi Proyek Machine Learning.

Modul ini berisi semua variabel konfigurasi global yang digunakan di seluruh
proyek machine learning. Variabel-variabel ini mencakup path direktori,
nama file, hyperparameter model, pengaturan Keras Tuner, dan parameter
lainnya yang terkait dengan eksperimen dan pelatihan model.

Menggunakan file konfigurasi terpusat seperti ini membantu dalam:
- Menjaga konsistensi pengaturan di seluruh proyek.
- Memudahkan modifikasi parameter tanpa harus mengubah kode di banyak tempat.
- Meningkatkan keterbacaan dan pemeliharaan kode.
"""

import os

# ==============================================================================
# PENGATURAN UMUM PROYEK
# ==============================================================================
SEED = 42
"""int: Seed untuk generator angka acak guna memastikan reproduktifitas eksperimen."""

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
"""str: Path absolut ke direktori root proyek. 
Dihitung relatif terhadap lokasi file config.py ini.
"""

# ==============================================================================
# PENGATURAN DATA
# ==============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
"""str: Path ke direktori tempat dataset disimpan."""

DATA_FILEPATH = os.path.join(DATA_DIR, "dataset_keuangan.csv")
"""str: Path lengkap menuju file dataset utama yang akan digunakan."""

TARGET_COLUMN = "Tipe"
"""str: Nama kolom target dalam dataset yang akan diprediksi."""

TEST_SPLIT_SIZE = 0.2
"""float: Proporsi dataset yang akan dialokasikan untuk set pengujian (test set). 
Nilai antara 0.0 dan 1.0."""

TOOLS_DIR = "tools"
"""str: Direktori untuk menyimpan alat bantu seperti scaler, encoder, dll."""

SCALER_FILEPATH = os.path.join(TOOLS_DIR, "scaler.pkl")
"""str: Path lengkap ke file scaler yang digunakan untuk normalisasi atau
standarisasi fitur numerik dalam dataset."""

LABEL_ENCODER_FILEPATH = os.path.join(TOOLS_DIR, "label_encoder.pkl")
"""str: Path lengkap ke file LabelEncoder yang digunakan untuk mengubah
label kategorikal menjadi format numerik."""

# ==============================================================================
# PENGATURAN DIREKTORI OUTPUT
# ==============================================================================
OUTPUT_DIR_BASE = PROJECT_ROOT_DIR
"""str: Direktori dasar untuk semua output yang dihasilkan oleh proyek 
(model, gambar, log, dll.). Secara default sama dengan root proyek.
"""

IMAGE_DIR = os.path.join(OUTPUT_DIR_BASE, "image")
"""str: Path ke direktori untuk menyimpan gambar atau plot yang dihasilkan 
(misalnya, dari Exploratory Data Analysis - EDA).
"""

MODEL_DIR = os.path.join(OUTPUT_DIR_BASE, "model")
"""str: Path ke direktori untuk menyimpan model machine learning yang telah dilatih."""

TUNER_DIR_BASE = os.path.join(OUTPUT_DIR_BASE, "keras_tuner_dir")
"""str: Path ke direktori dasar untuk menyimpan output dari Keras Tuner 
(hasil tuning hyperparameter).
"""

LOG_DIR_BASE = os.path.join(OUTPUT_DIR_BASE, "logs")
"""str: Path ke direktori dasar untuk menyimpan log, misalnya log TensorBoard."""

# ==============================================================================
# PENGATURAN MODEL
# ==============================================================================
BEST_MODEL_FILENAME = "best_model.keras"
"""str: Nama file standar untuk model terbaik yang disimpan setelah pelatihan atau tuning."""

BEST_MODEL_PATH = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
"""str: Path lengkap menuju file model terbaik yang disimpan."""

# ==============================================================================
# PENGATURAN HYPERPARAMETER TUNING (KERAS TUNER)
# ==============================================================================
TUNER_OBJECTIVE = "val_accuracy"
"""str: Metrik yang akan dioptimalkan oleh Keras Tuner selama proses pencarian 
hyperparameter. Contoh: 'val_accuracy', 'val_loss'.
"""

TUNER_MAX_EPOCHS_PER_TRIAL = 10
"""int: Jumlah epoch maksimum untuk setiap percobaan (trial) individual dalam Keras Tuner."""

TUNER_FACTOR = 3
"""int: Faktor yang digunakan oleh beberapa algoritma tuner (seperti Hyperband) untuk 
mengurangi jumlah model atau epoch antar bracket.
"""

TUNER_PROJECT_NAME = "financial_classification"
"""str: Nama proyek untuk Keras Tuner. Digunakan untuk mengorganisir direktori 
hasil tuning.
"""

TUNER_DIR_PROJECT = os.path.join(TUNER_DIR_BASE, TUNER_PROJECT_NAME)
"""str: Path ke direktori spesifik untuk menyimpan hasil dari proyek Keras Tuner ini."""

TUNER_SEARCH_EPOCHS = 500
"""int: Jumlah epoch total yang akan dijalankan oleh Keras Tuner untuk keseluruhan
proses pencarian hyperparameter (misalnya, dalam `tuner.search()`).
Tergantung pada jenis tuner, ini bisa berarti total epoch atau jumlah trial
yang dievaluasi.
"""

TUNER_EARLY_STOPPING_PATIENCE = 5
"""int: Jumlah epoch kesabaran (patience) untuk mekanisme early stopping 
*dalam setiap trial* Keras Tuner. Jika metrik yang dipantau tidak membaik
selama jumlah epoch ini, trial akan dihentikan lebih awal.
"""

# ==============================================================================
# PENGATURAN PELATIHAN MODEL FINAL
# ==============================================================================
FINAL_MODEL_EPOCHS = 100
"""int: Jumlah epoch untuk melatih model final setelah hyperparameter optimal ditemukan."""

FINAL_MODEL_BATCH_SIZE = 32
"""int: Ukuran batch yang digunakan saat melatih model final."""

# Pengaturan Callbacks untuk Pelatihan Model Final
CALLBACK_MODEL_CHECKPOINT_MONITOR = "val_accuracy"
"""str: Metrik yang dipantau oleh callback ModelCheckpoint untuk menyimpan 
model dengan performa terbaik pada data validasi.
"""

CALLBACK_EARLY_STOPPING_MONITOR = "val_loss"
"""str: Metrik yang dipantau oleh callback EarlyStopping untuk menghentikan 
pelatihan jika tidak ada peningkatan.
"""

CALLBACK_EARLY_STOPPING_PATIENCE = 7
"""int: Jumlah epoch kesabaran (patience) untuk callback EarlyStopping. Pelatihan
akan berhenti jika metrik yang dipantau tidak membaik selama jumlah epoch ini.
"""

CALLBACK_REDUCE_LR_MONITOR = "val_loss"
"""str: Metrik yang dipantau oleh callback ReduceLROnPlateau untuk mengurangi 
learning rate jika tidak ada peningkatan.
"""

CALLBACK_REDUCE_LR_FACTOR = 0.16667
"""float: Faktor pengurangan learning rate (new_lr = lr * factor) oleh callback 
ReduceLROnPlateau.
"""

CALLBACK_REDUCE_LR_PATIENCE = 3
"""int: Jumlah epoch kesabaran (patience) untuk callback ReduceLROnPlateau. 
Learning rate akan dikurangi jika metrik yang dipantau tidak membaik selama 
jumlah epoch ini.
"""

CALLBACK_REDUCE_LR_MIN_LR = 1e-6
"""float: Batas bawah learning rate. Callback ReduceLROnPlateau tidak akan 
mengurangi learning rate di bawah nilai ini.
"""

LOG_DIR_FIT = os.path.join(LOG_DIR_BASE, "fit", TUNER_PROJECT_NAME)
"""str: Direktori spesifik untuk menyimpan log TensorBoard dari proses pelatihan 
model final (hasil dari `model.fit()`). Disarankan untuk menyertakan nama proyek
agar log terorganisir.
"""

# ==============================================================================
# PENGATURAN EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
EDA_MAX_NUMERICAL_DIST_PLOTS = 5
"""int: Jumlah maksimum plot distribusi untuk fitur numerik yang akan ditampilkan 
atau disimpan selama fase EDA. Berguna untuk menghindari terlalu banyak plot
jika dataset memiliki banyak fitur numerik.
"""
