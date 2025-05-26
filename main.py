# main.py

"""
Script utama untuk menjalankan pipeline analisis keuangan dan klasifikasi.
Pipeline ini mencakup pemuatan data, eksplorasi data, pra-pemrosesan, pelatihan model,
evaluasi, dan testing dengan data buatan manual.
"""

import os
import pickle

from modules import config
from modules import utils
from modules import data_loader
from modules import eda
from modules import preprocessing
from modules import training
from modules import evaluation


def run_analysis_pipeline():
    """Menjalankan seluruh pipeline analisis dan pemodelan."""
    print("Memulai pipeline analisis keuangan dan klasifikasi...")

    # 1. Setup Awal
    utils.set_seeds()
    utils.create_output_directories()

    # 2. Pemuatan dan Inspeksi Data
    df_financial = data_loader.load_data(config.DATA_FILEPATH)
    if df_financial is None:
        print("Gagal memuat data. Pipeline dihentikan.")
        return
    data_loader.initial_data_inspection(df_financial)

    # 3. Exploratory Data Analysis (EDA)
    eda.perform_eda(df_financial, config.TARGET_COLUMN)

    # 4. Preprocessing Data
    processed_data = preprocessing.preprocess_data(df_financial)
    if processed_data[0] is None:
        print("Gagal melakukan pra-pemrosesan data. Pipeline dihentikan.")
        return

    (
        X_train_s,
        X_test_s,
        y_train_oh,
        y_test_oh,
        _scaler_returned_from_preprocessing,
        _encoder_returned_from_preprocessing,
        class_names,
        num_classes,
        y_test_labels_enc,
    ) = processed_data

    if X_train_s is None:
        print("Data latih tidak berhasil di-preprocess. Pipeline dihentikan.")
        return

    feature_names_for_manual_test = X_train_s.columns.tolist()
    input_shape_nn = (X_train_s.shape[1],)

    # 5. Pembuatan Model NN dengan Keras Tuner
    tuner = training.get_tuner(input_shape=input_shape_nn, num_classes=num_classes)

    best_nn_model_from_tuner, best_hyperparams = training.tune_hyperparameters(
        tuner, X_train_s, y_train_oh, X_test_s, y_test_oh
    )

    # 6. Pelatihan Model Terbaik
    training_history = training.train_final_model(
        best_nn_model_from_tuner,
        X_train_s,
        y_train_oh,
        X_test_s,
        y_test_oh,
        best_hyperparams,
    )

    # 7. Evaluasi Model
    final_trained_model = evaluation.load_trained_model(config.BEST_MODEL_PATH)
    if final_trained_model:
        evaluation.evaluate_model(
            final_trained_model,
            X_test_s,
            y_test_oh,
            y_test_labels_enc,
            class_names,
            training_history,
        )

        # 8. Testing dengan Data Buatan Manual
        print("\n--- Memuat Scaler dan Label Encoder untuk Testing Manual ---")
        loaded_scaler = None
        loaded_label_encoder = None

        if os.path.exists(config.SCALER_FILEPATH):
            try:
                with open(config.SCALER_FILEPATH, "rb") as f:
                    loaded_scaler = pickle.load(f)
                print(f"Scaler berhasil dimuat dari: {config.SCALER_FILEPATH}")
            except FileNotFoundError:
                print(f"Error: File scaler tidak ditemukan di {config.SCALER_FILEPATH}")
            except pickle.UnpicklingError:
                print(
                    f"Error: Gagal melakukan unpickle pada file scaler di {config.SCALER_FILEPATH}."
                )
            except Exception as e:
                print(f"Error saat memuat scaler: {e}")
        else:
            print(
                f"Peringatan: File scaler tidak ditemukan di {config.SCALER_FILEPATH}. Testing manual mungkin tidak akurat."
            )

        if os.path.exists(config.LABEL_ENCODER_FILEPATH):
            try:
                with open(config.LABEL_ENCODER_FILEPATH, "rb") as f:
                    loaded_label_encoder = pickle.load(f)
                print(
                    f"Label encoder berhasil dimuat dari: {config.LABEL_ENCODER_FILEPATH}"
                )
            except FileNotFoundError:
                print(
                    f"Error: File label encoder tidak ditemukan di {config.LABEL_ENCODER_FILEPATH}"
                )
            except pickle.UnpicklingError:
                print(
                    f"Error: Gagal melakukan unpickle pada file label encoder di {config.LABEL_ENCODER_FILEPATH}."
                )
            except Exception as e:
                print(f"Error saat memuat label encoder: {e}")
        else:
            print(
                f"Peringatan: File label encoder tidak ditemukan di {config.LABEL_ENCODER_FILEPATH}. Testing manual mungkin tidak akurat."
            )

        if loaded_scaler and loaded_label_encoder:
            if class_names:
                evaluation.test_with_manual_data(
                    model_path=config.BEST_MODEL_PATH,
                    scaler=loaded_scaler,
                    label_encoder=loaded_label_encoder,
                    feature_names=feature_names_for_manual_test,
                    class_names=class_names,
                )
            else:
                print(
                    "Class names tidak tersedia, testing dengan data buatan dilewati."
                )
        else:
            print(
                "Scaler atau Label Encoder gagal dimuat dari file, testing dengan data buatan dilewati."
            )

    else:
        print(
            "Tidak dapat mengevaluasi atau melakukan tes dengan data buatan karena model gagal dimuat."
        )

    print("\nPipeline analisis keuangan dan klasifikasi selesai.")


if __name__ == "__main__":
    run_analysis_pipeline()
