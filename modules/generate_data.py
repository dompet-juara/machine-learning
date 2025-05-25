# modules/generate_data.py

"""
Skrip untuk menghasilkan dataset keuangan sintetis dengan variasi tipe pengguna.
Dataset ini mencakup sumber pemasukan dan rincian pengeluaran,
dengan upaya untuk membuat distribusi digit pertama mendekati Hukum Benford
melalui penambahan noise dan pembulatan.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 16000
OUTPUT_DIR = Path("data")
OUTPUT_IMG = Path("image")
OUTPUT_FILENAME = "dataset_keuangan.csv"

USER_TYPES_CONFIG: Dict[str, np.ndarray] = {
    "boros": np.array([0.40, 0.50, 0.10]),
    "normal": np.array([0.50, 0.30, 0.20]),
    "hemat": np.array([0.50, 0.20, 0.30]),
}
USER_TYPE_PROBS: List[float] = [0.33, 0.34, 0.33]

INCOME_SOURCE_ALPHA: np.ndarray = np.array([60, 10, 20, 10])
MIN_INCOME = 2_000_000
MAX_INCOME = 100_000_000

NEEDS_ALPHA_RAW: np.ndarray = np.array([8.63, 15.38, 27.83, 2.32])
WANTS_ALPHA_RAW: np.ndarray = np.array([16.45, 17.00, 5.29])
SAVINGS_ALPHA_RAW: np.ndarray = np.array([2.20, 2.15, 1.91, 13.74])
EXPENSE_ALPHA_CONFIGS: Dict[str, np.ndarray] = {
    "needs": NEEDS_ALPHA_RAW,
    "wants": WANTS_ALPHA_RAW,
    "savings": SAVINGS_ALPHA_RAW,
}

ADJUSTMENT_NOISE_SCALE = 0.01
ROUNDING_UNIT = 1000
MINIMUM_ADJUSTED_VALUE = 1000

INCOME_COLUMNS: List[str] = ["Gaji", "Tabungan Lama", "Investasi", "Pemasukan Lainnya"]
EXPENSE_COLUMNS: List[str] = [
    "Bahan Pokok",
    "Protein & Gizi Tambahan",
    "Tempat Tinggal",
    "Sandang",
    "Konsumsi Praktis",
    "Barang & Jasa Sekunder",
    "Pengeluaran Tidak Esensial",
    "Pajak",
    "Asuransi",
    "Sosial & Budaya",
    "Tabungan / Investasi",
]
USER_TYPE_COLUMN = "Tipe"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_digits(number: float) -> Tuple[Optional[int], Optional[int]]:
    """
    Mengambil digit pertama dan kedua dari sebuah angka.
    """
    s_num = str(int(number)).lstrip("0")
    if not s_num:
        return None, None
    first_digit = int(s_num[0])
    second_digit = int(s_num[1]) if len(s_num) > 1 else 0
    return first_digit, second_digit


def adjust_financial_values(
    numbers: np.ndarray,
    noise_scale: float = ADJUSTMENT_NOISE_SCALE,
    rounding_unit: int = ROUNDING_UNIT,
    min_value: int = MINIMUM_ADJUSTED_VALUE,
) -> np.ndarray:
    """
    Menyesuaikan nilai finansial dengan menambahkan noise dan melakukan pembulatan.
    """
    adjusted_numbers = []
    for num in numbers:
        if num <= 0:
            adjusted_numbers.append(float(min_value))
            continue
        noise = np.random.normal(0, num * noise_scale)
        new_num = num + noise
        if new_num <= 0:
            new_num = max(num / 2, float(min_value))
        new_num_rounded = round(new_num / rounding_unit) * rounding_unit
        if new_num_rounded < min_value:
            new_num_rounded = float(min_value)
        adjusted_numbers.append(new_num_rounded)
    return np.array(adjusted_numbers)


def generate_user_types_and_spending_proportions(
    n_samples: int,
    user_types_config: Dict[str, np.ndarray],
    user_type_probs: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menghasilkan tipe pengguna acak dan proporsi pengeluaran mereka.
    """
    user_type_keys = list(user_types_config.keys())
    chosen_user_types = np.random.choice(
        user_type_keys, size=n_samples, p=user_type_probs
    )
    spending_proportions = np.zeros((n_samples, 3))
    for i, user_type in enumerate(chosen_user_types):
        base_proportions = user_types_config[user_type]
        noisy_proportions = np.random.dirichlet(base_proportions * 100)
        spending_proportions[i] = noisy_proportions
    return chosen_user_types, spending_proportions


def generate_income_data(
    n_samples: int,
    min_income: float,
    max_income: float,
    income_source_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menghasilkan total pemasukan dan rincian sumber pemasukan.
    """
    total_income = np.random.uniform(min_income, max_income, size=n_samples)
    income_sources_proportions = np.random.dirichlet(
        income_source_alpha, size=n_samples
    )
    income_sources_nominal = total_income[:, np.newaxis] * income_sources_proportions
    return total_income, income_sources_nominal


def _calculate_one_category_nominals(
    category_total_allocation: np.ndarray,
    category_alpha_raw: np.ndarray,
    dirichlet_multiplier: float,
    n_samples: int,
) -> np.ndarray:
    """
    Helper: Menghitung nominal subkategori untuk satu kategori utama (misal, kebutuhan).
    """
    category_alpha_normalized = category_alpha_raw / category_alpha_raw.sum()
    category_sub_prop = np.random.dirichlet(
        category_alpha_normalized * dirichlet_multiplier, size=n_samples
    )
    category_nominal = category_sub_prop * category_total_allocation[:, np.newaxis]
    return category_nominal


def generate_expense_subcategories(
    total_income: np.ndarray,
    spending_proportions: np.ndarray,
    expense_alpha_configs: Dict[str, np.ndarray],
    dirichlet_multiplier: float = 100.0,
) -> np.ndarray:
    """
    Menghasilkan nominal untuk setiap subkategori pengeluaran.
    Args:
        total_income: Array total pemasukan per sampel.
        spending_proportions: Array proporsi pengeluaran utama (kebutuhan, keinginan, tabungan).
        expense_alpha_configs: Dict berisi alpha mentah untuk 'needs', 'wants', 'savings'.
        dirichlet_multiplier: Pengali untuk alpha distribusi Dirichlet subkategori.

    Returns:
        Array (n_samples, n_total_subcategories) nominal untuk semua subkategori pengeluaran.
    """
    n_samples = total_income.shape[0]

    needs_total_allocation = total_income * spending_proportions[:, 0]
    wants_total_allocation = total_income * spending_proportions[:, 1]
    savings_total_allocation = total_income * spending_proportions[:, 2]

    needs_nominal = _calculate_one_category_nominals(
        needs_total_allocation,
        expense_alpha_configs["needs"],
        dirichlet_multiplier,
        n_samples,
    )
    wants_nominal = _calculate_one_category_nominals(
        wants_total_allocation,
        expense_alpha_configs["wants"],
        dirichlet_multiplier,
        n_samples,
    )
    savings_nominal = _calculate_one_category_nominals(
        savings_total_allocation,
        expense_alpha_configs["savings"],
        dirichlet_multiplier,
        n_samples,
    )

    all_expense_nominal = np.hstack([needs_nominal, wants_nominal, savings_nominal])
    return all_expense_nominal


def adjust_all_nominal_values(
    nominal_array: np.ndarray, noise_scale: float, rounding_unit: int, min_value: int
) -> np.ndarray:
    """
    Menerapkan penyesuaian (noise, pembulatan) ke setiap kolom dari array nominal.
    """
    adjusted_array = np.zeros_like(nominal_array)
    for i in range(nominal_array.shape[1]):
        column_data = nominal_array[:, i]
        adjusted_array[:, i] = adjust_financial_values(
            column_data, noise_scale, rounding_unit, min_value
        )
    return adjusted_array


def create_dataframes(
    income_sources_nominal: np.ndarray,
    all_expense_nominal_adjusted: np.ndarray,
    chosen_user_types: np.ndarray,
) -> pd.DataFrame:
    """
    Membuat DataFrame gabungan dari data pemasukan dan pengeluaran.
    Menggunakan konstanta global untuk nama kolom (INCOME_COLUMNS, dll.).
    """
    df_income = pd.DataFrame(income_sources_nominal, columns=INCOME_COLUMNS)
    df_expenses = pd.DataFrame(all_expense_nominal_adjusted, columns=EXPENSE_COLUMNS)

    df_income[USER_TYPE_COLUMN] = chosen_user_types

    df_final = pd.concat([df_income, df_expenses], axis=1)
    return df_final


def save_dataframe_to_csv(df: pd.DataFrame, dir_path: Path, filename: str) -> None:
    """
    Menyimpan DataFrame ke file CSV.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path / filename
    df.to_csv(filepath, index=False)
    logging.info("Dataset lengkap disimpan di '%s'", filepath)


def plot_first_digit_distribution(
    data_series: pd.Series, title: str, save_path: Optional[Path] = None
) -> None:
    """
    Membuat plot distribusi digit pertama dari sebuah series data.
    """
    first_digits: List[int] = []
    for value in data_series:
        if pd.notna(value) and value > 0:
            digit1, _ = get_digits(value)
            if digit1 is not None:
                first_digits.append(digit1)
    if not first_digits:
        logging.warning(
            "Tidak ada digit pertama yang valid ditemukan untuk plot: %s", title
        )
        return

    count = Counter(first_digits)
    total_valid_digits = sum(count.values())
    frequencies = [
        count.get(i, 0) / total_valid_digits if total_valid_digits > 0 else 0
        for i in range(1, 10)
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 10), frequencies, color="skyblue")
    benford_expected = [np.log10(1 + 1 / d) for d in range(1, 10)]
    plt.plot(
        range(1, 10),
        benford_expected,
        color="red",
        linestyle="--",
        marker="o",
        label="Benford's Law",
    )
    plt.title(title)
    plt.xlabel("Digit Pertama")
    plt.ylabel("Frekuensi Relatif")
    plt.xticks(range(1, 10))
    plt.legend()
    plt.grid(axis="y", linestyle="--")
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logging.info("Plot disimpan di %s", save_path)
        plt.close()
    else:
        plt.show()


def print_average_spending_by_type(df: pd.DataFrame) -> None:
    """
    Mencetak rata-rata pembagian pengeluaran per tipe pengguna.
    Menggunakan konstanta global USER_TYPE_COLUMN dan EXPENSE_COLUMNS.
    """
    logging.info("\nRata-rata pembagian pengeluaran per tipe pengguna:")
    valid_expense_cols = [col for col in EXPENSE_COLUMNS if col in df.columns]
    if not valid_expense_cols:
        logging.warning("Tidak ada kolom pengeluaran yang valid untuk dianalisis.")
        return
    avg_spending = df.groupby(USER_TYPE_COLUMN)[valid_expense_cols].mean()
    logging.info("\n%s", avg_spending.to_string())


def main() -> None:
    """
    Fungsi utama untuk menjalankan seluruh proses pembuatan dataset.
    """
    np.random.seed(RANDOM_SEED)

    logging.info("Menghasilkan tipe pengguna dan proporsi pengeluaran...")
    (
        chosen_user_types,
        spending_proportions,
    ) = generate_user_types_and_spending_proportions(
        N_SAMPLES, USER_TYPES_CONFIG, USER_TYPE_PROBS
    )

    logging.info("Menghasilkan data pemasukan...")
    total_income, income_sources_nominal_raw = generate_income_data(
        N_SAMPLES, MIN_INCOME, MAX_INCOME, INCOME_SOURCE_ALPHA
    )
    income_sources_nominal_adjusted = adjust_all_nominal_values(
        income_sources_nominal_raw,
        ADJUSTMENT_NOISE_SCALE,
        ROUNDING_UNIT,
        MINIMUM_ADJUSTED_VALUE,
    )

    logging.info("Menghasilkan rincian pengeluaran...")
    all_expense_nominal_raw = generate_expense_subcategories(
        total_income, spending_proportions, EXPENSE_ALPHA_CONFIGS
    )
    all_expense_nominal_adjusted = adjust_all_nominal_values(
        all_expense_nominal_raw,
        ADJUSTMENT_NOISE_SCALE,
        ROUNDING_UNIT,
        MINIMUM_ADJUSTED_VALUE,
    )

    logging.info("Membuat DataFrame...")
    df_final = create_dataframes(
        income_sources_nominal_adjusted, all_expense_nominal_adjusted, chosen_user_types
    )

    save_dataframe_to_csv(df_final, OUTPUT_DIR, OUTPUT_FILENAME)

    logging.info("Membuat visualisasi distribusi digit pertama...")
    if "Bahan Pokok" in df_final.columns:
        plot_first_digit_distribution(
            df_final["Bahan Pokok"],
            "Distribusi Digit Pertama - Bahan Pokok (Setelah Penyesuaian)",
            save_path=OUTPUT_IMG / "distribusi_digit_bahan_pokok.png",
        )
    else:
        logging.warning("Kolom 'Bahan Pokok' tidak ditemukan untuk visualisasi.")

    print_average_spending_by_type(df_final)

    logging.info("Proses pembuatan dataset selesai.")


if __name__ == "__main__":
    main()
