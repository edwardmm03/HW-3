from sklearn import datasets
from ada_boost import AdaBoostDTC, AdaBoostGBC
import numpy as np
import matplotlib.pyplot as plt

TRAINING_DEPTH = 20


def even_odd(arr: np.ndarray[np.int64]) -> np.ndarray[np.int32]:
    result = np.zeros(len(arr), dtype=np.int32)
    for i in range(len(arr)):
        if arr[i] % 2 == 0:
            result[i] = 1
        else:
            result[i] = -1
    return result


def plot_errors(
    dtc_training_errors: np.ndarray[np.float32],
    gbc_training_errors: np.ndarray[np.float32],
    depths: np.ndarray[np.int32],
):
    plt.plot(depths, dtc_training_errors, label="DTC Classifier")
    plt.plot(depths, gbc_training_errors, label="GBC Classifier")
    plt.legend()
    plt.xlabel("Interations of Boosting")
    plt.ylabel("Error of Classifier")
    plt.show()


def training_error(
    prediction: np.ndarray[np.int32], target: np.ndarray[np.int32]
) -> np.float32:
    total_error = 0
    for i in range(len(prediction)):
        if prediction[i] != target[i]:
            total_error += 1
    return np.float32(total_error / len(prediction))


if __name__ == "__main__":
    digits = datasets.load_digits()
    data: np.ndarray[np.ndarray[np.float64]] = digits["data"]
    target: np.ndarray[np.int64] = digits["target"]
    dtc = AdaBoostDTC()
    gbc = AdaBoostGBC()
    even_odd_ = even_odd(target)
    depths: np.ndarray[np.int32] = np.array(range(1, TRAINING_DEPTH + 1))
    dtc_training_errors: np.ndarray[np.float32] = np.zeros(
        TRAINING_DEPTH, dtype=np.float32
    )
    gbc_training_errors: np.ndarray[np.float32] = np.zeros(
        TRAINING_DEPTH, dtype=np.float32
    )
    for i in depths:
        dtc.train(data, even_odd_, i)
        gbc.train(data, even_odd_, i)
        dtc_training_errors[i - 1] = training_error(
            dtc.predit(data), even_odd_
        )
        gbc_training_errors[i - 1] = training_error(
            gbc.predit(data), even_odd_
        )
    plot_errors(dtc_training_errors, gbc_training_errors, depths)
