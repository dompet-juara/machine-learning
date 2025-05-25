# modules/training.py

"""
Modul Pelatihan Model Machine Learning.

Modul ini bertanggung jawab untuk proses pelatihan model, yang mencakup:
1. Inisialisasi Keras Tuner untuk pencarian hyperparameter otomatis.
2. Pelaksanaan proses tuning hyperparameter menggunakan data latih dan validasi.
3. Pengambilan model dan hyperparameter terbaik hasil tuning.
4. Pelatihan model final menggunakan hyperparameter terbaik dengan serangkaian
   callbacks canggih (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard).

Modul ini menggunakan TensorFlow, Keras Tuner, dan modul lokal `config`
untuk parameter konfigurasi serta `model_builder` untuk definisi arsitektur model.
"""

import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
from typing import Tuple, List, Any, Optional

from . import config
from . import model_builder


def get_tuner(input_shape: Tuple[int, ...], num_classes: int) -> kt.Tuner:
    """Menginisialisasi dan mengembalikan objek Keras Tuner (Hyperband).

    Fungsi ini menyiapkan Keras Tuner dengan algoritma Hyperband untuk
    melakukan pencarian hyperparameter. Tuner akan menggunakan fungsi
    `model_builder.build_hypermodel` untuk membangun model dengan berbagai
    konfigurasi hyperparameter.

    Pengaturan tuner (seperti objective, max_epochs, directory) diambil
    dari modul `config`.

    Args:
        input_shape (Tuple[int, ...]): Bentuk (shape) dari data input fitur.
            Contoh: `(jumlah_fitur,)` untuk data tabular.
        num_classes (int): Jumlah kelas unik pada variabel target.

    Returns:
        kt.Tuner: Objek Keras Tuner (misalnya, `keras_tuner.Hyperband`) yang
                  telah dikonfigurasi dan siap untuk proses `search()`.
    """
    print("\n--- Menginisialisasi Keras Tuner (Hyperband) ---")
    hypermodel_fn = lambda hp: model_builder.build_hypermodel(
        hp, input_shape=input_shape, num_classes=num_classes
    )

    tuner = kt.Hyperband(
        hypermodel_fn,
        objective=config.TUNER_OBJECTIVE,
        max_epochs=config.TUNER_MAX_EPOCHS_PER_TRIAL,
        factor=config.TUNER_FACTOR,
        directory=config.TUNER_DIR_BASE,
        project_name=config.TUNER_PROJECT_NAME,
        overwrite=True,
    )
    print(
        f"Keras Tuner (Hyperband) diinisialisasi. Objective: '{config.TUNER_OBJECTIVE}'."
    )
    print(f"Hasil tuning akan disimpan di: '{config.TUNER_DIR_PROJECT}'.")
    return tuner


def tune_hyperparameters(
    tuner: kt.Tuner,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[tf.keras.Model, kt.HyperParameters]:
    """Melakukan proses tuning hyperparameter menggunakan Keras Tuner.

    Fungsi ini menjalankan metode `search()` pada objek tuner yang diberikan,
    menggunakan data latih dan validasi. Callback EarlyStopping digunakan
    untuk menghentikan trial yang tidak menjanjikan lebih awal.

    Setelah pencarian selesai, hyperparameter terbaik dan model yang dibangun
    dengan hyperparameter tersebut akan diekstrak dan dikembalikan.

    Args:
        tuner (kt.Tuner): Objek Keras Tuner yang telah diinisialisasi.
        X_train (pd.DataFrame | np.ndarray): Fitur-fitur data latih.
        y_train (np.ndarray): Label target data latih (biasanya one-hot encoded).
        X_val (pd.DataFrame | np.ndarray): Fitur-fitur data validasi.
        y_val (np.ndarray): Label target data validasi (biasanya one-hot encoded).

    Returns:
        Tuple[tf.keras.Model, kt.HyperParameters]: Sebuah tuple berisi:
            - best_model_from_tuner (tf.keras.Model): Model Keras yang dibangun
              menggunakan hyperparameter terbaik (model ini belum dilatih secara penuh,
              hanya arsitekturnya yang sudah optimal).
            - best_hps (kt.HyperParameters): Objek HyperParameters yang berisi
              kombinasi hyperparameter terbaik yang ditemukan.
    """
    print("\n" + "=" * 20 + " MEMULAI TUNING HYPERPARAMETER " + "=" * 20)

    stop_early_tuner = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.TUNER_EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True,
    )

    print(
        f"Memulai pencarian hyperparameter dengan Keras Tuner (maks {config.TUNER_SEARCH_EPOCHS} epoch pencarian)..."
    )
    tuner.search(
        X_train,
        y_train,
        epochs=config.TUNER_SEARCH_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[stop_early_tuner],
        verbose=1,
    )

    print("\nEkstraksi hyperparameter terbaik...")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Hyperparameter terbaik yang ditemukan oleh Keras Tuner:")
    for param, value in best_hps.values.items():
        print(f"  - {param}: {value}")

    print("\nMembangun model dengan hyperparameter terbaik...")
    best_model_from_tuner = tuner.hypermodel.build(best_hps)
    print("Ringkasan arsitektur model terbaik dari tuner:")
    best_model_from_tuner.summary(line_length=100)

    print("\n" + "=" * 20 + " TUNING HYPERPARAMETER SELESAI " + "=" * 20)
    return best_model_from_tuner, best_hps


def get_advanced_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Mendefinisikan dan mengembalikan daftar callbacks Keras untuk pelatihan model final.

    Callbacks yang disertakan:
    - ModelCheckpoint: Menyimpan model terbaik berdasarkan metrik tertentu
      (misalnya, `val_accuracy`).
    - EarlyStopping: Menghentikan pelatihan jika tidak ada peningkatan pada
      metrik yang dipantau (misalnya, `val_loss`) setelah sejumlah epoch tertentu.
    - ReduceLROnPlateau: Mengurangi learning rate jika metrik yang dipantau
      tidak membaik.
    - TensorBoard: Menyimpan log untuk visualisasi di TensorBoard.

    Konfigurasi untuk setiap callback diambil dari modul `config`.

    Returns:
        List[tf.keras.callbacks.Callback]: Daftar objek callback Keras.
    """
    print("\n--- Menyiapkan Callbacks untuk Pelatihan Model Final ---")
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.BEST_MODEL_PATH,
            monitor=config.CALLBACK_MODEL_CHECKPOINT_MONITOR,
            save_best_only=True,
            verbose=1,
            mode="max"
            if "accuracy" in config.CALLBACK_MODEL_CHECKPOINT_MONITOR
            else "min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=config.CALLBACK_EARLY_STOPPING_MONITOR,
            patience=config.CALLBACK_EARLY_STOPPING_PATIENCE,
            verbose=1,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.CALLBACK_REDUCE_LR_MONITOR,
            factor=config.CALLBACK_REDUCE_LR_FACTOR,
            patience=config.CALLBACK_REDUCE_LR_PATIENCE,
            min_lr=config.CALLBACK_REDUCE_LR_MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config.LOG_DIR_FIT,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
        ),
    ]
    print(
        f"Callbacks yang digunakan: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard."
    )
    print(
        f"Model terbaik akan disimpan di: '{config.BEST_MODEL_PATH}' (berdasarkan '{config.CALLBACK_MODEL_CHECKPOINT_MONITOR}')."
    )
    print(f"Log TensorBoard akan disimpan di: '{config.LOG_DIR_FIT}'.")
    return callbacks_list


def train_final_model(
    model: tf.keras.Model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    best_hps: Optional[kt.HyperParameters] = None,
) -> tf.keras.callbacks.History:
    """Melatih model Keras final menggunakan hyperparameter terbaik dan callbacks.

    Fungsi ini mengambil model yang arsitekturnya sudah ditentukan (biasanya
    dari hasil tuning), data latih, data validasi, dan hyperparameter terbaik.
    Model akan dilatih selama jumlah epoch yang ditentukan dalam `config`,
    menggunakan batch size yang juga bisa diambil dari `best_hps` atau `config`.

    Args:
        model (tf.keras.Model): Model Keras yang akan dilatih. Sebaiknya model ini
            sudah dikompilasi dengan learning rate dari `best_hps` jika ada.
        X_train (pd.DataFrame | np.ndarray): Fitur-fitur data latih.
        y_train (np.ndarray): Label target data latih (biasanya one-hot encoded).
        X_val (pd.DataFrame | np.ndarray): Fitur-fitur data validasi.
        y_val (np.ndarray): Label target data validasi (biasanya one-hot encoded).
        best_hps (Optional[kt.HyperParameters], optional): Objek HyperParameters yang
            berisi hyperparameter terbaik. Digunakan untuk mengambil `batch_size`
            jika di-tune, atau learning rate jika model belum dikompilasi ulang.
            Defaultnya None.

    Returns:
        tf.keras.callbacks.History: Objek History yang berisi riwayat metrik
                                    pelatihan dan validasi.
    """
    print("\n" + "=" * 20 + " MEMULAI PELATIHAN MODEL FINAL " + "=" * 20)

    callbacks = get_advanced_callbacks()

    if best_hps and "batch_size" in best_hps.values:
        batch_size = best_hps.get("batch_size")
        print(f"Menggunakan batch_size dari hyperparameter terbaik: {batch_size}")
    else:
        batch_size = config.FINAL_MODEL_BATCH_SIZE
        print(f"Menggunakan batch_size default dari config: {batch_size}")

    if best_hps and "learning_rate" in best_hps.values:
        current_lr = model.optimizer.learning_rate.numpy()
        best_lr = best_hps.get("learning_rate")
        if abs(current_lr - best_lr) > 1e-7:
            print(
                f"Mengompilasi ulang model dengan learning rate terbaik: {best_lr} (sebelumnya: {current_lr:.2e})"
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
        else:
            print(f"Model sudah menggunakan learning rate optimal: {current_lr:.2e}")

    print(f"Memulai pelatihan model final selama {config.FINAL_MODEL_EPOCHS} epoch...")
    history = model.fit(
        X_train,
        y_train,
        epochs=config.FINAL_MODEL_EPOCHS,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    print(
        f"\nModel terbaik (berdasarkan '{config.CALLBACK_MODEL_CHECKPOINT_MONITOR}') "
        f"selama pelatihan telah disimpan di: {config.BEST_MODEL_PATH}"
    )
    print("\n" + "=" * 20 + " PELATIHAN MODEL FINAL SELESAI " + "=" * 20)
    return history
