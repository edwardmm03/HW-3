from ada_boost import AdaBoost
from typing import Union
import numpy as np


class DigitClassifier:
    classifiers: Union[list[AdaBoost], None] = None

    def train(
        self,
        data: np.ndarray[np.ndarray[np.float64]],
        target: np.ndarray[np.int64],
        depth: int,
    ) -> None:
        classifiers = [None] * 10
        for i in range(len(classifiers)):
            classifiers[i] = AdaBoost()
            classifiers[i].train(data, self.target_digit(target, i), depth)

    def target_digit(
        self, target: np.ndarray[np.int64], digit: np.int64
    ) -> np.ndarray[np.float32]:
        result: np.ndarray[np.float32] = np.zeros(
            len(target), dtype=np.float32
        )
        for i in range(len(result)):
            if target[i] == digit:
                result[i] = 1
            else:
                result[i] = -1
        return result
