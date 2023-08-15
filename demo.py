import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import preprocess
from typing import Tuple
from main import local_dir, TEST_DATA_NAME


@st.cache_resource
def load_pretrained_mlp_model() -> tf.keras.Model:
    model_dir = local_dir.joinpath("mlp")
    return tf.keras.models.load_model(str(model_dir.joinpath("model")))


@st.cache_resource
def load_pretrained_cnn_model() -> tf.keras.Model:
    model_dir = local_dir.joinpath("cnn")
    return tf.keras.models.load_model(str(model_dir.joinpath("model")))


@st.cache_data
def load_mlp_train_history() -> pd.DataFrame:
    model_dir = local_dir.joinpath("mlp")
    return pd.read_csv(model_dir.joinpath("training_log.csv"))


@st.cache_data
def load_cnn_train_history() -> pd.DataFrame:
    model_dir = local_dir.joinpath("cnn")
    return pd.read_csv(model_dir.joinpath("training_log.csv"))


@st.cache_data
def load_test_data() -> Tuple[np.array, np.array]:
    return preprocess.load_data(TEST_DATA_NAME, local_dir)


@st.cache_data
def get_mlp_model_summary_path(_model: tf.keras.Model) -> str:
    """
    Unlike string, tf.keras.Model is not hashable. Underscore is attached in front of `model` parameter to explicitly
    state that this object should not be hashed. This is totally fine since parameter 'model' is no longer required
    after this method saves the summary plot to given file path.
    :param _model: MLP model
    :return: file path to the summary plot
    """
    model_dir = local_dir.joinpath("mlp")
    summary_path = str(model_dir.joinpath("model_summary.png"))
    tf.keras.utils.plot_model(
        _model,
        to_file=summary_path,
        show_shapes=True,
        show_dtype=True,
        show_layer_activations=True,
        show_trainable=True,
    )
    return summary_path


@st.cache_data
def get_cnn_model_summary_path(_model: tf.keras.Model) -> str:
    model_dir = local_dir.joinpath("cnn")
    summary_path = str(model_dir.joinpath("model_summary.png"))
    tf.keras.utils.plot_model(
        _model,
        to_file=summary_path,
        show_shapes=True,
        show_dtype=True,
        show_layer_activations=True,
        show_trainable=True,
    )
    return summary_path


# load objects
mlp_model = load_pretrained_mlp_model()
cnn_model = load_pretrained_cnn_model()
test_data = load_test_data()
