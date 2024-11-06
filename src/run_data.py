from sklearn import datasets
from ada_boost import AdaBoostDTC, AdaBoostGBC
import numpy as np


def even_odd(arr: np.ndarray[np.int64]) -> np.ndarray[np.int32]:
    result = np.zeros(len(arr), dtype=np.int32)
    for i in range(len(arr)):
        if arr[i] % 2 == 0:
            result[i] = 1
        else:
            result[i] = -1
    return result


def training_error(prediction, target) -> np.float32:
    return 0


if __name__ == "__main__":
    digits = datasets.load_digits()
    data: np.ndarray[np.ndarray[np.float64]] = digits["data"]
    target: np.ndarray[np.int64] = digits["target"]
    dtc = AdaBoostDTC()
    gbc = AdaBoostGBC()
    even_odd_ = even_odd(target)
    for i in range(10):
        i += 1
        dtc.train(data, even_odd_, i)
        gbc.train(data, even_odd_, i)
        dtc_training_error = training_error(dtc.predit(data), even_odd_)
        gbc_training_error = training_error(gbc.predit(data), even_odd_)
