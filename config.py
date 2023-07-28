from dataclasses import dataclass

LOCAL_DIR = "/tmp/svhn"


@dataclass
class PreprocessConfig:
    TRAIN_DATA_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    TEST_DATA_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    TRAIN_DATA_NAME = "train_data.mat"
    TEST_DATA_NAME = "test_data.mat"
