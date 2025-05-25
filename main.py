# main.py

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
    
    (X_train_s, X_test_s, y_train_oh, y_test_oh,
     scaler_obj, encoder_obj, class_names, num_classes, y_test_labels_enc) = processed_data
    
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
        best_nn_model_from_tuner, X_train_s, y_train_oh, X_test_s, y_test_oh, best_hyperparams
    )

    # 7. Evaluasi Model
    final_trained_model = evaluation.load_trained_model(config.BEST_MODEL_PATH)
    if final_trained_model:
        evaluation.evaluate_model(
            final_trained_model, X_test_s, y_test_oh, y_test_labels_enc, class_names, training_history
        )
        
        # 8. Testing dengan Data Buatan Manual
        if scaler_obj and encoder_obj:
            evaluation.test_with_manual_data(
                model_path=config.BEST_MODEL_PATH,
                scaler=scaler_obj,
                label_encoder=encoder_obj,
                feature_names=feature_names_for_manual_test,
                class_names=class_names
            )
        else:
            print("Scaler atau Label Encoder tidak tersedia, testing dengan data buatan dilewati.")
            
    else:
        print("Tidak dapat mengevaluasi atau melakukan tes dengan data buatan karena model gagal dimuat.")

    print("\nPipeline analisis keuangan dan klasifikasi selesai.")

if __name__ == "__main__":
    run_analysis_pipeline()