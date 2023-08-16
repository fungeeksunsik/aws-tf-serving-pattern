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
    images, labels = preprocess.load_data(TEST_DATA_NAME, local_dir)
    images = preprocess.reshape_data(images)
    labels = labels.flatten()
    return images, labels


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


tf.config.set_visible_devices([], 'GPU')  # prevent 'CHECK failed: target + size == res' error in M2 mac
mlp_model = load_pretrained_mlp_model()
cnn_model = load_pretrained_cnn_model()
images, labels = load_test_data()
st.title("Pretrained Image Classifiers Demo")

if st.sidebar.button("Refresh"):
    random_index = np.random.randint(0, images.shape[0])
    sample_image = images[random_index]
    text_column, sample_image_column = st.columns(spec=[0.85, 0.15])
    with text_column:
        """
        As displayed example, each image in SVHN dataset consists of 32x32 shaped RGB image with corresponding label 
        ranges from 0 to 9. Each of classifier will preprocess image, and pass the image into the model to calculate
        softmax values for each label. Decision rule is to pick lable whose predicted soft value is highest compared to 
        others.
        """
    with sample_image_column:
        st.image(sample_image, caption=f"label : {labels[random_index]}")
    grayscaled_image = preprocess.convert_to_grayscale(sample_image)

    st.header("MLP Model Prediction Result")
    grayscaled_image = grayscaled_image[np.newaxis, :, :]
    mlp_scores = mlp_model.predict(grayscaled_image, verbose=0)[0]
    mlp_prediction = np.argmax(mlp_scores)
    if mlp_prediction == labels[random_index]:
        st.success(f"Prediction result of MLP model is correct(label: {mlp_prediction})", icon="✅")
    else:
        st.warning(f"MLP model failed to give correct prediction(label: {mlp_prediction})", icon="⚠️")
    st.bar_chart(pd.DataFrame({"softmax scores(MLP)": mlp_scores}))

    st.header("CNN Model Prediction Result")
    grayscaled_image = grayscaled_image[:, :, :, np.newaxis]
    cnn_scores = cnn_model.predict(grayscaled_image, verbose=0)[0]
    cnn_prediction = np.argmax(cnn_scores)
    if cnn_prediction == labels[random_index]:
        st.success(f"Prediction result of CNN model is correct(label: {cnn_prediction})", icon="✅")
    else:
        st.warning(f"CNN model failed to give correct prediction(label: {cnn_prediction})", icon="⚠️")
    st.bar_chart(pd.DataFrame({"softmax scores(CNN)": cnn_scores}))
else:
    st.text("Click `Refresh` button on the sidebar")
