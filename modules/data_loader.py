# modules/data_loader.py

"""
Modul Data Loader untuk Proyek Machine Learning.

Modul ini bertanggung jawab untuk:
1. Memuat dataset dari sumber yang ditentukan (umumnya file CSV).
2. Menyediakan fungsi untuk inspeksi data awal, seperti menampilkan
   beberapa baris pertama, informasi tipe data, statistik deskriptif,
   dan jumlah missing values.

Modul ini menggunakan pandas untuk manipulasi DataFrame dan mengacu pada
variabel konfigurasi dari `modules.config` untuk path default dataset.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional

from . import config


def load_data(filepath: str = config.DATA_FILEPATH) -> Union[pd.DataFrame, None]:
    """Memuat dataset dari file CSV ke dalam pandas DataFrame.

    Fungsi ini mencoba memuat data dari `filepath`. Jika file tidak ditemukan,
    ia akan mencetak pesan error dan, untuk tujuan demonstrasi atau
    pengembangan berkelanjutan tanpa menghentikan alur kerja,
    mengembalikan DataFrame dummy dengan struktur yang mirip.
    Jika terjadi kesalahan lain selama pemuatan, ia akan mengembalikan None.

    Args:
        filepath (str, optional): Path absolut atau relatif ke file CSV dataset.
            Defaultnya adalah nilai dari `config.DATA_FILEPATH`.

    Returns:
        Union[pd.DataFrame, None]:
            - DataFrame yang berisi data dari file CSV jika berhasil dimuat.
            - DataFrame dummy jika file asli tidak ditemukan (untuk tujuan demo).
            - None jika terjadi error lain saat proses pemuatan data.
        # Jika menggunakan Optional[pd.DataFrame]:
        # Optional[pd.DataFrame]: DataFrame yang dimuat, atau None jika gagal.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset berhasil dimuat dari: {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File dataset di '{filepath}' tidak ditemukan.")
        print(
            "Membuat DataFrame dummy agar eksekusi dapat dilanjutkan (HANYA UNTUK DEMO)."
        )
        print(
            "Pastikan file dataset tersedia di path yang benar untuk penggunaan aktual."
        )
        data_dummy = {
            "Gaji": np.random.uniform(1000000, 20000000, 100).astype(int),
            "Tabungan Lama": np.random.uniform(500000, 10000000, 100).astype(int),
            "Investasi": np.random.uniform(0, 5000000, 100).astype(int),
            "Pemasukan Lainnya": np.random.uniform(0, 2000000, 100).astype(int),
            "Tipe": np.random.choice(
                ["boros", "hemat", "normal"], 100, p=[0.3, 0.4, 0.3]
            ),
            "Bahan Pokok": np.random.uniform(500000, 3000000, 100).astype(int),
            "Protein & Gizi Tambahan": np.random.uniform(200000, 1500000, 100).astype(
                int
            ),
            "Tempat Tinggal": np.random.uniform(1000000, 5000000, 100).astype(int),
            "Sandang": np.random.uniform(100000, 1000000, 100).astype(int),
            "Konsumsi Praktis": np.random.uniform(100000, 1200000, 100).astype(int),
            "Barang & Jasa Sekunder": np.random.uniform(0, 2000000, 100).astype(int),
            "Pengeluaran Tidak Esensial": np.random.uniform(0, 1000000, 100).astype(
                int
            ),
            "Pajak": np.random.uniform(50000, 500000, 100).astype(int),
            "Asuransi": np.random.uniform(0, 800000, 100).astype(int),
            "Sosial & Budaya": np.random.uniform(0, 500000, 100).astype(int),
            "Tabungan / Investasi": np.random.uniform(100000, 2000000, 100).astype(int),
        }
        return pd.DataFrame(data_dummy)
    except Exception as e:
        print(
            f"Terjadi error yang tidak terduga saat memuat data dari '{filepath}': {e}"
        )
        return None


def initial_data_inspection(df: Optional[pd.DataFrame]) -> None:
    """Melakukan dan mencetak hasil inspeksi data awal pada DataFrame.

    Fungsi ini menampilkan informasi dasar mengenai DataFrame yang diberikan,
    termasuk:
    - Lima baris pertama data (head).
    - Informasi umum DataFrame (info), yang mencakup tipe data setiap kolom
      dan jumlah entri non-null.
    - Statistik deskriptif untuk semua kolom, termasuk data numerik dan kategorikal.
    - Jumlah missing values (nilai yang hilang) per kolom dan totalnya.

    Fungsi ini tidak mengembalikan nilai, melainkan mencetak output ke konsol.
    Jika DataFrame yang diberikan adalah None, fungsi akan mencetak pesan
    dan tidak melakukan inspeksi.

    Args:
        df (Optional[pd.DataFrame]): DataFrame yang akan diinspeksi.
                                  Jika None, fungsi akan menangani ini dengan baik.
    Returns:
        None
    """
    if df is None:
        print(
            "DataFrame tidak tersedia (None). Inspeksi data awal tidak dapat dilakukan."
        )
        return

    print("\n" + "=" * 15 + " HEAD DATA (5 BARIS PERTAMA) " + "=" * 15)
    print(df.head())

    print("\n" + "=" * 15 + " INFORMASI UMUM DATA " + "=" * 15)
    df.info()

    print("\n" + "=" * 15 + " DESKRIPSI STATISTIK DATA " + "=" * 15)
    print(df.describe(include="all").T)

    print("\n" + "=" * 15 + " JUMLAH MISSING VALUES PER KOLOM " + "=" * 15)
    missing_values = df.isnull().sum()
    print(missing_values)

    total_missing = missing_values.sum()
    if total_missing == 0:
        print("\nTidak ditemukan missing values dalam dataset. Bagus!")
    else:
        print(f"\nTotal missing values dalam dataset: {total_missing}.")
        print(
            "Perlu dilakukan penanganan missing values (misalnya imputasi atau penghapusan)."
        )
    print("=" * 50)
