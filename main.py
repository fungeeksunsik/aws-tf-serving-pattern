import logging
import pathlib
import sys
import numpy as np
import preprocess
import models
import tensorflow as tf

LOCAL_DIR = "/tmp/svhn"
TRAIN_DATA_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
TEST_DATA_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
TRAIN_DATA_NAME = "train_data.mat"
TEST_DATA_NAME = "test_data.mat"

MODEL_MLP_CHECKPOINT_POSTFIX = "model_mlp/ckpt"

formatter = logging.Formatter(
    fmt="%(asctime)s : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

local_dir = pathlib.Path(LOCAL_DIR)
local_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    logger.info("Download data from original source")
    preprocess.download_data(TRAIN_DATA_URL, TRAIN_DATA_NAME, local_dir)
    preprocess.download_data(TEST_DATA_URL, TEST_DATA_NAME, local_dir)

    logger.info("Load downloaded data")
    train_data, train_labels = preprocess.load_data(TRAIN_DATA_NAME, local_dir)

    logger.info("Preprocess train data")
    train_data = preprocess.reshape_data(train_data)
    train_data = preprocess.convert_to_grayscale(train_data)

    logger.info("Define callbacks to be used for model tracking")
    mlp_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{local_dir}/model_mlp/ckpt",
        save_weights_only=True,
        save_best_only=True,
        save_freq="epoch",
        monitor="val_accuracy",
        verbose=1,
    )
    mlp_records = tf.keras.callbacks.CSVLogger(
        filename=f"{local_dir}/model_mlp/training_log.csv"
    )
    cnn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{local_dir}/model_cnn/ckpt",
        save_weights_only=True,
        save_best_only=True,
        save_freq="epoch",
        monitor="val_accuracy",
        verbose=1,
    )
    cnn_records = tf.keras.callbacks.CSVLogger(
        filename=f"{local_dir}/model_cnn/training_log.csv"
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, mode="max"
    )

    logger.info("Train MLP classifier")
    model_mlp = models.define_model_mlp(train_data[0].shape)
    model_mlp = models.compile_model(model_mlp)
    model_mlp.fit(
        x=train_data,
        y=train_labels,
        batch_size=64,
        epochs=30,
        verbose=2,
        validation_split=0.15,
        callbacks=[early_stopping, mlp_checkpoint, mlp_records],
    )

    logger.info("Train CNN classifier")
    train_data = train_data[:, :, :, np.newaxis]
    model_cnn = models.define_model_cnn(train_data[0].shape)
    model_cnn = models.compile_model(model_cnn)
    model_cnn.fit(
        x=train_data,
        y=train_labels,
        batch_size=64,
        epochs=30,
        verbose=2,
        validation_split=0.15,
        callbacks=[early_stopping, cnn_checkpoint, cnn_records],
    )

    logger.info("Load best performing weight checkpoint and save model")
    model_mlp.load_weights(f"{local_dir}/model_mlp/ckpt")
    model_mlp.save(filepath=f"{local_dir}/model_mlp/model")
    model_cnn.load_weights(f"{local_dir}/model_cnn/ckpt")
    model_cnn.save(filepath=f"{local_dir}/model_cnn/model")
