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


# load objects
mlp_model = load_pretrained_mlp_model()
cnn_model = load_pretrained_cnn_model()
mlp_history = load_mlp_train_history()
cnn_history = load_cnn_train_history()
test_data = load_test_data()

st.dataframe(mlp_history)
st.dataframe(cnn_history)

# create loss comparison chart

# create accuracy comparison chart
