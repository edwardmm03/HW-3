from sklearn import datasets
from digit_classifier import DigitClassifier
import numpy as np

if __name__ == "__main__":
    digits = datasets.load_digits()
    data: np.ndarray[np.ndarray[np.float64]] = digits["data"]
    target: np.ndarray[np.int64] = digits["target"]
    learner = DigitClassifier()
    learner.train(data, target, 5)
    # print(learner.predit(data))
