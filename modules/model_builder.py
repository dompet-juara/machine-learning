# modules/model_builder.py

"""
Modul Pembangun Model untuk Keras Tuner.

Modul ini bertanggung jawab untuk mendefinisikan arsitektur model neural network
yang fleksibel dan dapat di-tuning menggunakan Keras Tuner. Fungsi utama,
`build_hypermodel`, memungkinkan pencarian otomatis untuk hyperparameter
optimal seperti jumlah lapisan, jumlah unit per lapisan, fungsi aktivasi,
tingkat dropout, dan laju pembelajaran (learning rate).

Tujuan utamanya adalah untuk memfasilitasi penemuan konfigurasi model
yang paling efektif untuk tugas klasifikasi tertentu, dalam konteks ini,
klasifikasi data keuangan.
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple


def build_hypermodel(
    hp: "keras_tuner.HyperParameters", input_shape: Tuple[int, ...], num_classes: int
) -> tf.keras.Model:
    """Membangun dan mengkompilasi model Keras Sequential yang dapat di-tuning.

    Fungsi ini mendefinisikan arsitektur model neural network di mana beberapa
    hyperparameter kunci (seperti jumlah lapisan, unit di setiap lapisan,
    fungsi aktivasi, laju dropout, dan laju pembelajaran optimizer)
    didefinisikan sebagai ruang pencarian menggunakan objek `HyperParameters`
    dari Keras Tuner.

    Arsitektur dasar model adalah sebagai berikut:
    1.  **Layer Input**: Menerima data fitur dengan `input_shape` yang ditentukan.
    2.  **Hidden Layers**: Sejumlah lapisan Dense (fully connected) yang dapat
        di-tuning (antara 1 hingga 3 lapisan). Setiap lapisan Dense diikuti oleh:
        a.  `BatchNormalization`: Untuk menstabilkan dan mempercepat pelatihan.
        b.  `Activation`: Fungsi aktivasi yang dapat di-tuning (misalnya, 'relu', 'tanh').
        c.  `Dropout`: Untuk regularisasi dan mencegah overfitting.
    3.  **Layer Output**: Sebuah lapisan Dense dengan fungsi aktivasi 'softmax',
        yang menghasilkan distribusi probabilitas atas `num_classes` kelas target.

    Model ini dikompilasi menggunakan optimizer Adam dengan laju pembelajaran
    (learning rate) yang juga dapat di-tuning, dan fungsi loss
    'categorical_crossentropy', yang cocok untuk klasifikasi multikelas
    dengan target one-hot encoded. Metrik yang dipantau adalah 'accuracy'.

    Args:
        hp (keras_tuner.HyperParameters): Objek `HyperParameters` dari Keras Tuner.
            Objek ini digunakan untuk mendefinisikan dan mengambil sampel nilai
            hyperparameter dari ruang pencarian yang telah ditentukan.
        input_shape (Tuple[int, ...]): Bentuk (shape) dari data input fitur.
            Untuk data tabular dengan N fitur, ini biasanya `(N,)`.
        num_classes (int): Jumlah kelas unik pada variabel target. Ini akan
            menentukan jumlah unit (neuron) pada lapisan output model.

    Returns:
        tf.keras.Model: Sebuah model Keras (`tf.keras.Model`) yang telah
                        dikompilasi dan siap untuk digunakan dalam proses tuning
                        oleh Keras Tuner atau untuk pelatihan langsung jika
                        hyperparameter sudah ditentukan.
    """
    model = tf.keras.Sequential(name="Financial_Classifier_Hypermodel")

    model.add(layers.Input(shape=input_shape, name="Input_Layer"))

    # --- Hidden Layers yang Dapat Di-tuning ---
    for i in range(hp.Int("num_hidden_layers", min_value=1, max_value=3, step=1)):
        layer_name_prefix = f"Hidden_Layer_{i+1}"

        # Layer Dense dengan jumlah unit yang dapat di-tuning
        model.add(
            layers.Dense(
                units=hp.Int(
                    f"units_{layer_name_prefix}", min_value=32, max_value=256, step=32
                ),
                name=f"Dense_{layer_name_prefix}",
            )
        )

        # Batch Normalization untuk stabilisasi dan percepatan training
        model.add(layers.BatchNormalization(name=f"BatchNorm_{layer_name_prefix}"))

        # Fungsi aktivasi yang dapat di-tuning
        model.add(
            layers.Activation(
                hp.Choice(
                    f"activation_{layer_name_prefix}",
                    values=["relu", "tanh", "elu", "selu", "swish", "leaky_relu"],
                ),
                name=f"Activation_{layer_name_prefix}",
            )
        )

        # Dropout untuk regularisasi, mencegah overfitting
        model.add(
            layers.Dropout(
                rate=hp.Float(
                    f"dropout_rate_{layer_name_prefix}",
                    min_value=0.0,
                    max_value=0.5,
                    step=0.05,
                ),
                name=f"Dropout_{layer_name_prefix}",
            )
        )

    # --- Layer Output ---
    model.add(layers.Dense(num_classes, activation="softmax", name="Output_Layer"))

    # --- Kompilasi Model ---
    learning_rate_hp = hp.Choice(
        "learning_rate", values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_hp),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
