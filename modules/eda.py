# modules/eda.py

"""
Modul Exploratory Data Analysis (EDA).

Modul ini menyediakan fungsi-fungsi untuk melakukan analisis data eksploratif
pada sebuah DataFrame. Tujuannya adalah untuk memahami dataset lebih baik,
mengidentifikasi pola, anomali, dan hubungan antar variabel melalui
visualisasi dan statistik ringkas.

Fungsi-fungsi di dalamnya mencakup:
- Visualisasi distribusi variabel target.
- Visualisasi distribusi fitur-fitur numerik.
- Visualisasi hubungan antara fitur numerik dan variabel target.
- Visualisasi matriks korelasi antar fitur numerik.

Plot yang dihasilkan akan disimpan ke direktori gambar yang ditentukan
dalam modul `config` menggunakan fungsi dari modul `utils`.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List

from . import config
from . import utils


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> None:
    """Memvisualisasikan distribusi variabel target menggunakan countplot dan pie chart.

    Fungsi ini menghasilkan dua plot:
    1. Countplot: Menunjukkan jumlah absolut setiap kelas dalam variabel target.
    2. Pie chart: Menunjukkan proporsi (persentase) setiap kelas dalam variabel target.

    Plot akan disimpan secara otomatis menggunakan `utils.save_plot()`.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data.
        target_col (str): Nama kolom yang dijadikan sebagai variabel target.

    Returns:
        None
    """
    print(f"\nAnalisis Variabel Target: '{target_col}'")

    fig_countplot, ax_countplot = plt.subplots(figsize=(8, 6))
    sns.countplot(
        x=target_col,
        data=df,
        ax=ax_countplot,
        palette="viridis",
        order=df[target_col].value_counts().index,
    )
    ax_countplot.set_title(f"Distribusi Kelas Target ({target_col})")
    ax_countplot.set_xlabel("Kategori Target")
    ax_countplot.set_ylabel("Jumlah")
    for p in ax_countplot.patches:
        ax_countplot.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )
    utils.save_plot(fig_countplot, "target_distribution_countplot.png")
    plt.show()

    target_counts = df[target_col].value_counts()
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(
        target_counts,
        labels=target_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("viridis", len(target_counts)),
    )
    ax_pie.set_title(f"Proporsi Kelas Target ({target_col})")
    ax_pie.axis("equal")
    utils.save_plot(fig_pie, "target_distribution_pie.png")
    plt.show()


def plot_numerical_feature_distributions(
    df: pd.DataFrame, numerical_features: List[str]
) -> None:
    """Memvisualisasikan distribusi fitur numerik menggunakan histogram dan density plot.

    Untuk setiap fitur numerik yang diberikan, fungsi ini akan membuat plot
    yang menunjukkan distribusi nilainya. Jika jumlah fitur numerik melebihi
    batas yang ditentukan dalam `config.EDA_MAX_NUMERICAL_DIST_PLOTS`,
    maka hanya sejumlah fitur awal yang akan diplot.

    Plot akan disimpan secara otomatis menggunakan `utils.save_plot()`.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data.
        numerical_features (List[str]): Daftar nama kolom fitur numerik
                                         yang akan divisualisasikan.

    Returns:
        None
    """
    print("\nDistribusi Fitur Numerik (Histogram & Density Plot):")
    features_to_plot = numerical_features
    if (
        config.EDA_MAX_NUMERICAL_DIST_PLOTS is not None
        and len(numerical_features) > config.EDA_MAX_NUMERICAL_DIST_PLOTS
        and config.EDA_MAX_NUMERICAL_DIST_PLOTS > 0
    ):
        features_to_plot = numerical_features[: config.EDA_MAX_NUMERICAL_DIST_PLOTS]
        print(
            f"Menampilkan distribusi untuk {config.EDA_MAX_NUMERICAL_DIST_PLOTS} "
            f"fitur numerik pertama dari total {len(numerical_features)} fitur."
        )
    elif config.EDA_MAX_NUMERICAL_DIST_PLOTS == 0:
        print("Plotting distribusi fitur numerik dilewati sesuai konfigurasi.")
        return

    for col in features_to_plot:
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col], kde=True, ax=ax_dist, color="skyblue")
        ax_dist.set_title(f"Distribusi Fitur: {col}")
        ax_dist.set_xlabel(col)
        ax_dist.set_ylabel("Frekuensi")
        safe_col_name = utils.sanitize_filename(col)
        utils.save_plot(fig_dist, f"dist_{safe_col_name}.png")
        plt.show()


def plot_numerical_features_vs_target(
    df: pd.DataFrame, numerical_features: List[str], target_col: str
) -> None:
    """Memvisualisasikan hubungan fitur numerik dengan variabel target menggunakan boxplot.

    Fungsi ini membuat boxplot untuk setiap fitur numerik, dikelompokkan berdasarkan
    kategori pada variabel target. Ini membantu untuk melihat bagaimana distribusi
    fitur numerik berbeda antar kelas target.

    Plot akan diatur dalam subplot dan disimpan sebagai satu gambar
    menggunakan `utils.save_plot()`.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data.
        numerical_features (List[str]): Daftar nama kolom fitur numerik.
        target_col (str): Nama kolom yang dijadikan sebagai variabel target.

    Returns:
        None
    """
    print(f"\nBoxplot Fitur Numerik vs. Target ('{target_col}'):")
    num_cols_for_boxplot = [col for col in numerical_features if col != target_col]

    if not num_cols_for_boxplot:
        print(
            f"Tidak ada fitur numerik (selain '{target_col}') untuk diplot dengan boxplot."
        )
        return

    n_features = len(num_cols_for_boxplot)
    n_cols_subplot = 3
    n_rows_subplot = (n_features + n_cols_subplot - 1) // n_cols_subplot

    fig_boxplots, axes_boxplots = plt.subplots(
        n_rows_subplot, n_cols_subplot, figsize=(18, n_rows_subplot * 5), squeeze=False
    )
    axes_boxplots_flat = axes_boxplots.flatten()

    for i, col in enumerate(num_cols_for_boxplot):
        if i < len(axes_boxplots_flat):
            sns.boxplot(
                x=target_col,
                y=col,
                data=df,
                ax=axes_boxplots_flat[i],
                palette="viridis",
            )
            axes_boxplots_flat[i].set_title(f"{col} vs. {target_col}")
            axes_boxplots_flat[i].tick_params(axis="x", rotation=45)

    for j in range(n_features, len(axes_boxplots_flat)):
        fig_boxplots.delaxes(axes_boxplots_flat[j])

    plt.tight_layout()
    utils.save_plot(fig_boxplots, "numerical_features_boxplot_vs_target.png")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, numerical_features: List[str]) -> None:
    """Memvisualisasikan matriks korelasi antar fitur numerik menggunakan heatmap.

    Heatmap ini menunjukkan koefisien korelasi Pearson antar pasangan
    fitur numerik. Nilai korelasi berkisar antara -1 dan 1.
    Berguna untuk mengidentifikasi multikolinearitas.

    Plot akan disimpan secara otomatis menggunakan `utils.save_plot()`.

    Args:
        df (pd.DataFrame): DataFrame yang berisi data.
        numerical_features (List[str]): Daftar nama kolom fitur numerik
                                         untuk dihitung korelasinya.

    Returns:
        None
    """
    print("\nAnalisis Korelasi Fitur Numerik (Heatmap):")
    if not numerical_features:
        print(
            "Tidak ada fitur numerik yang tersedia untuk menghitung matriks korelasi."
        )
        return

    correlation_matrix = df[numerical_features].corr()

    figsize_width = max(12, len(numerical_features) * 0.8)
    figsize_height = max(10, len(numerical_features) * 0.7)

    fig_corr, ax_corr = plt.subplots(figsize=(figsize_width, figsize_height))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax_corr,
        annot_kws={"size": 8},
    )
    ax_corr.set_title("Matriks Korelasi Fitur Numerik")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    utils.save_plot(fig_corr, "correlation_heatmap.png")
    plt.show()


def perform_eda(df: pd.DataFrame, target_col: str) -> None:
    """Fungsi utama untuk melakukan serangkaian analisis data eksploratif (EDA).

    Fungsi ini akan:
    1. Memvisualisasikan distribusi variabel target.
    2. Mengidentifikasi fitur numerik dalam DataFrame.
    3. Memvisualisasikan distribusi fitur-fitur numerik (jika diaktifkan di config).
    4. Memvisualisasikan hubungan antara fitur numerik dan variabel target (boxplot).
    5. Memvisualisasikan matriks korelasi antar fitur numerik.

    Semua plot yang dihasilkan akan disimpan ke direktori yang ditentukan
    dan juga ditampilkan.

    Args:
        df (pd.DataFrame): DataFrame yang akan dianalisis.
        target_col (str): Nama kolom yang merupakan variabel target.

    Returns:
        None
    """
    if df is None:
        print("DataFrame tidak tersedia (None). Proses EDA dibatalkan.")
        return

    print("\n" + "=" * 20 + " MEMULAI EXPLORATORY DATA ANALYSIS (EDA) " + "=" * 20)

    # 1. Analisis Variabel Target
    plot_target_distribution(df, target_col)

    # 2. Identifikasi Fitur Numerik
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    print(
        f"\nFitur numerik yang terdeteksi ({len(numerical_features)} fitur): {numerical_features}"
    )
    categorical_features = df.select_dtypes(include="object").columns.tolist()
    if target_col in categorical_features:
        categorical_features.remove(target_col)
    print(
        f"Fitur kategorikal yang terdeteksi ({len(categorical_features)} fitur, selain target): {categorical_features}"
    )

    # 3. Distribusi Fitur Numerik
    if numerical_features:
        if (
            config.EDA_MAX_NUMERICAL_DIST_PLOTS is None
            or config.EDA_MAX_NUMERICAL_DIST_PLOTS != 0
        ):
            plot_numerical_feature_distributions(df, numerical_features)
        else:
            print(
                "\nPlotting distribusi fitur numerik dilewati sesuai konfigurasi (EDA_MAX_NUMERICAL_DIST_PLOTS = 0)."
            )
    else:
        print("\nTidak ada fitur numerik untuk divisualisasikan distribusinya.")

    # 4. Fitur Numerik vs. Target
    if numerical_features:
        plot_numerical_features_vs_target(df, numerical_features, target_col)
    else:
        print("\nTidak ada fitur numerik untuk diplot terhadap target.")

    # 5. Matriks Korelasi
    if numerical_features:
        plot_correlation_heatmap(df, numerical_features)
    else:
        print("\nTidak ada fitur numerik untuk membuat heatmap korelasi.")

    print("\n" + "=" * 20 + " EXPLORATORY DATA ANALYSIS (EDA) SELESAI " + "=" * 20)
