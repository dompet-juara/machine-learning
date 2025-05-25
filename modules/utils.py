# modules/utils.py

"""
Modul Utilitas Umum untuk Proyek Machine Learning.

Modul ini berisi fungsi-fungsi pendukung yang digunakan di berbagai bagian
proyek. Fungsi-fungsi ini mencakup tugas-tugas umum seperti:
- Pengaturan seed untuk generator angka acak guna memastikan reproduktifitas.
- Pembuatan direktori output yang diperlukan oleh proyek.
- Penyimpanan plot Matplotlib ke file.

Modul ini bergantung pada os, matplotlib, numpy, tensorflow, dan modul
lokal `config` untuk parameter konfigurasi.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import config


def set_seeds(seed_value: int = config.SEED) -> None:
    """Mengatur seed untuk generator angka acak di NumPy dan TensorFlow.

    Pengaturan seed ini bertujuan untuk memastikan bahwa eksperimen machine learning
    dapat direproduksi. Dengan seed yang sama, urutan angka acak yang dihasilkan
    akan konsisten, yang mengarah pada hasil yang sama pada inisialisasi bobot,
    pembagian data (jika `random_state` menggunakan seed ini), dan operasi
    stokastik lainnya.

    Args:
        seed_value (int, optional): Nilai integer yang akan digunakan sebagai seed.
            Defaultnya adalah nilai dari `config.SEED`.

    Returns:
        None
    """
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f"Seed untuk NumPy dan TensorFlow telah diatur ke: {seed_value}")


def create_output_directories() -> None:
    """Membuat semua direktori output yang diperlukan oleh proyek jika belum ada.

    Fungsi ini membaca path direktori dari modul `config` dan menggunakan
    `os.makedirs` dengan `exist_ok=True` untuk membuat direktori tersebut.
    Ini memastikan bahwa direktori tujuan untuk menyimpan model, gambar,
    log, dan hasil tuner tersedia sebelum digunakan.

    Direktori yang dibuat meliputi:
    - `config.IMAGE_DIR`
    - `config.MODEL_DIR`
    - `config.TUNER_DIR_PROJECT` (direktori spesifik untuk proyek Keras Tuner)
    - `config.LOG_DIR_FIT` (direktori untuk log pelatihan TensorBoard)

    Args:
        None

    Returns:
        None
    """
    print("\n--- Memastikan Direktori Output Tersedia ---")

    dirs_to_create = [
        config.IMAGE_DIR,
        config.MODEL_DIR,
        config.TUNER_DIR_PROJECT,
        config.LOG_DIR_FIT,
    ]

    for dir_path in dirs_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Direktori dipastikan/dibuat: {dir_path}")
        except OSError as e:
            print(f"Error saat membuat direktori {dir_path}: {e}")


def save_plot(
    fig: plt.Figure, filename: str, directory: str = config.IMAGE_DIR
) -> None:
    """Menyimpan objek figure Matplotlib ke file dalam direktori yang ditentukan.

    Fungsi ini mengambil objek `matplotlib.figure.Figure`, nama file, dan
    direktori tujuan. Plot akan disimpan dengan `bbox_inches='tight'` untuk
    memastikan tidak ada bagian plot yang terpotong. Setelah disimpan,
    figure akan ditutup (`plt.close(fig)`) untuk melepaskan memori.

    Args:
        fig (plt.Figure): Objek figure Matplotlib yang akan disimpan.
        filename (str): Nama file untuk menyimpan plot (misalnya, 'my_plot.png').
                        Ekstensi file (seperti .png, .jpg, .pdf) akan menentukan format.
        directory (str, optional): Path ke direktori tempat plot akan disimpan.
            Defaultnya adalah `config.IMAGE_DIR`.

    Returns:
        None
    """
    if not filename:
        print("Error: Nama file tidak boleh kosong untuk menyimpan plot.")
        plt.close(fig)
        return

    filepath = os.path.join(directory, filename)
    try:
        fig.savefig(filepath, bbox_inches="tight", dpi=150)
        print(f"Plot berhasil disimpan di: {filepath}")
    except Exception as e:
        print(f"Error saat menyimpan plot ke '{filepath}': {e}")
    finally:
        plt.close(fig)


def sanitize_filename(filename_str: str) -> str:
    """Membersihkan string agar aman digunakan sebagai nama file.

    Fungsi ini akan:
    1. Mengubah string menjadi huruf kecil.
    2. Menghapus karakter selain alfanumerik, spasi, dan tanda hubung.
    3. Mengganti spasi atau tanda hubung berulang dengan satu tanda hubung.
    4. Menghapus tanda hubung di awal atau akhir string.
    5. Mengembalikan "untitled_plot" jika string menjadi kosong setelah sanitasi.

    Args:
        filename_str (str): String nama file yang akan disanitasi.

    Returns:
        str: String nama file yang sudah aman.
    """
    if not isinstance(filename_str, str):
        return f"invalid_filename_type_{type(filename_str).__name__}"

    s = re.sub(r"[^\w\s-]", "", filename_str.lower())
    s = re.sub(r"[-\s]+", "-", s).strip("-_")
    return s if s else "untitled_plot"
