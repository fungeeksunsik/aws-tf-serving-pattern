import requests
import pathlib
import numpy as np
from scipy.io import loadmat
from typing import Tuple


def download_data(source_url: str, file_name: str, local_dir: pathlib.Path):
    """
    download data from original source(ufldl.stanford.edu)
    :param source_url: URL of data to be downloaded
    :param file_name: string to be used as downloaded file
    :param local_dir: local directory to save downloaded file
    :return: None
    """
    file_path = local_dir.joinpath(file_name)
    response = requests.get(source_url)
    with open(file_path, "wb") as file:
        file.write(response.content)


def load_data(file_name: str, local_dir: pathlib.Path) -> Tuple[np.array, np.array]:
    """
    Load data and apply two basic data preprocessing operations
     1. Normalize image pixel values into [0,1]
     2. Change label name '10' to '0' which in fact corresponds to image of zero
    :param file_name: filename of the dataset to be loaded
    :param local_dir: local directory which stores downloaded file
    :return: a pair of feed data and corresponding target label
    """
    file_path = local_dir.joinpath(file_name)
    dataset = loadmat(str(file_path))
    feed_data = dataset["X"] / 255.
    target_labels = dataset["y"]
    target_labels[target_labels == 10] = 0
    return feed_data, target_labels


def reshape_data(feed_data: np.array) -> np.array:
    """
    Reshape data so that batch_size dimension comes to first place
    :param feed_data: loaded feed data
    :return: reshaped feed data whose first dimension is place for batch_size
    """
    images = []
    for index in range(feed_data.shape[-1]):
        images.append(feed_data[:, :, :, index])
    return np.stack(images)


def convert_to_grayscale(feed_data: np.array) -> np.array:
    """
    Calculate average over color channel to convert input images into grayscale
    :param feed_data: reshaped feed data
    :return: images converted into grayscale
    """
    try:
        channel_index = feed_data.shape.index(3)
        return np.mean(feed_data, axis=channel_index)
    except ValueError:
        raise ValueError(
            "Cannot find color channel. It might have been converted to greyscale already."
        )
