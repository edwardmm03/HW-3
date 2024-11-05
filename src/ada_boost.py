from sklearn.tree import DecisionTreeClassifier
from typing import Union
import numpy as np


class AdaBoost:
    learners: Union[list[DecisionTreeClassifier], None] = None
    alphas: Union[list[np.float32], None] = None

    def predit(
        self, data: np.ndarray[np.ndarray[np.float32]]
    ) -> Union[np.ndarray[np.int32], None]:
        if self.learners is None or self.alphas is None:
            return None
        result: np.ndarray[np.float32] = np.zeros(len(data), dtype=np.float32)
        for i in range(len(self.learners)):
            result = np.add(
                result,
                np.multiply(self.alphas[i], self.learners[i].predict(data)),
            )
        output: np.ndarray[np.int32] = np.zeros(len(result), dtype=np.int32)
        for i in range(len(output)):
            output[i] = self.sign(result[i])
        return output

    def train(
        self,
        data: np.ndarray[np.ndarray[np.float32]],
        target: np.ndarray[np.float32],
        depth: int,
    ) -> None:
        if len(data) != len(target):
            return None
        self.learners = [None] * depth
        self.alphas = [None] * depth
        distribution: np.ndarray[np.float32] = self.uniform_dist(len(data))
        for i in range(depth):
            self.learners[i] = DecisionTreeClassifier(
                random_state=1, max_depth=2
            )
            self.learners[i].fit(data, target, sample_weight=distribution)
            prediction: np.ndarray[np.float32] = self.learners[i].predict(data)
            errors: np.ndarray[np.float32] = self.errors(prediction, target)
            self.alphas[i] = self.alpha(distribution, errors)
            distribution: np.ndarray[np.float32] = self.update_distribution(
                distribution, errors, self.alphas[i]
            )

    def errors(
        self,
        prediction: np.ndarray[np.float32],
        target: np.ndarray[np.float32],
    ) -> Union[np.ndarray[np.float32], None]:
        if len(prediction) != len(target):
            return None
        errors: np.ndarray[np.float32] = np.zeros(
            len(prediction), dtype=np.float32
        )
        for i in range(len(errors)):
            errors[i] = prediction[i] * target[i]
        return errors

    def update_distribution(
        self,
        distribution: np.ndarray[np.float32],
        errors: np.ndarray[np.float32],
        alpha: np.float32,
    ) -> Union[np.ndarray[np.float32], None]:
        if len(distribution) != len(errors):
            return None
        result: np.ndarray[np.float32] = np.zeros(
            len(distribution), dtype=np.float32
        )
        for i in range(len(result)):
            result[i] = distribution[i] * np.exp(-1 * alpha * errors[i])
        return self.normalize(result)

    def alpha(
        self,
        distribution: np.ndarray[np.float32],
        errors: np.ndarray[np.float32],
    ) -> Union[np.float32, None]:
        if len(distribution) != len(errors):
            return None
        error: np.float32 = min(np.dot(distribution, errors), 1)
        return np.float32(0.5 * np.log2((1 + error) / (1 - error)))

    def sign(self, num: np.float32) -> np.int32:
        if num < 0:
            return np.int32(-1)
        return np.int32(1)

    def normalize(self, arr: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return np.multiply(arr, 1 / np.sum(arr))

    def uniform_dist(self, len: int) -> np.ndarray[np.float32]:
        return np.full(
            shape=len, fill_value=np.float32(1 / len), dtype=np.float32
        )
