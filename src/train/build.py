import numpy as np
import tensorflow as tf

from src.common.utils import normalize_to_float32_01


def build_cnn(
    conv_filters,
    seed,
    input_shape,
    dropout_conv,
    dense_units,
    dropout_dense,
    num_classes,
    learning_rate,
):
    f1, f2 = conv_filters
    init = tf.keras.initializers.GlorotUniform()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(input_shape),
            tf.keras.layers.Conv2D(
                f1, kernel_size=3, padding="same", kernel_initializer=init
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                f2, kernel_size=3, padding="same", kernel_initializer=init
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Dropout(dropout_conv),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_units, kernel_initializer=init),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(dropout_dense),
            tf.keras.layers.Dense(
                num_classes, activation="softmax", kernel_initializer=init
            ),
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate)

    model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def create_model_signature(model_name, input_shape, num_classes, learning_rate):
    return {
        "name": model_name,
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "optimizer": "adam",
        "learning_rate": learning_rate,
        "loss": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        "notes": "Compact, reproducible baseline for MNIST.",
    }


def build_tf_datasets(
    Xtr_u8: np.ndarray,
    ytr: np.ndarray,
    Xva_u8: np.ndarray,
    yva: np.ndarray,
    shuffle_buffer,
    seed,
    batch_size,
):
    Xtr = normalize_to_float32_01(Xtr_u8)
    Xva = normalize_to_float32_01(Xva_u8)

    # deterministic-ish shuffling
    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, ytr))
    ds_tr = ds_tr.shuffle(
        buffer_size=min(shuffle_buffer, len(ytr)),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    ds_tr = ds_tr.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    ds_va = tf.data.Dataset.from_tensor_slices((Xva, yva))
    ds_va = ds_va.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds_tr, ds_va
