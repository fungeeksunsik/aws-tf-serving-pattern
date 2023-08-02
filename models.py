import tensorflow as tf
from typing import Tuple


def define_model_mlp(input_shape: Tuple[int, int]) -> tf.keras.Sequential:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(
                data_format="channels_last",
                input_shape=input_shape,
                name="flatten_layer"
            ),
            tf.keras.layers.Dense(
                units=512,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="dense_layer_1"
            ),
            tf.keras.layers.Dense(
                units=256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="dense_layer_2"
            ),
            tf.keras.layers.Dense(
                units=128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="dense_layer_3"
            ),
            tf.keras.layers.Dense(
                units=64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="dense_layer_4"
            ),
            tf.keras.layers.Dense(
                units=10,
                activation="softmax",
                name="softmax_classifier"
            ),
        ],
        name="svhn_classifier_mlp"
    )
    return model


def define_model_cnn(input_shape: Tuple[int, int, int]) -> tf.keras.Sequential:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                input_shape=input_shape,
                name="convolution_1",
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="max_pool_1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Conv2D(
                filters=8,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                name="convolution_2",
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="max_pool_2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(
                data_format="channels_last",
                name="flatten_layer"
            ),
            tf.keras.layers.Dense(
                units=256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="dense_layer_1"
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                units=128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name="dense_layer_2"
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                units=10,
                activation="softmax",
                name="softmax_classifier"
            ),
        ],
        name="svhn_classifier_cnn"
    )
    return model


def compile_model(model: tf.keras.Sequential) -> tf.keras.Sequential:
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model
