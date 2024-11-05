from sklearn.tree import DecisionTreeClassifier
from typing import Union
import numpy as np


class AdaBoost:
    learners: Union[list[DecisionTreeClassifier], None] = None
    alphas: Union[list[np.float32], None] = None

    def predit(
        self, domain: np.ndarray[np.float32]
    ) -> Union[np.ndarray[np.int32], None]:
        if self.learners is None or self.alphas is None:
            return None
        result = np.zeros(len(domain), dtype=np.float32)
        for i in range(len(self.learners)):
            result = np.add(
                result,
                np.multiply(self.alphas[i], self.learners[i].predict(domain)),
            )
        output = np.zeros(len(result), dtype=np.int32)
        for i in range(len(output)):
            output[i] = self.sign(result[i])
        return output

    def train(
        self,
        domain: np.ndarray[np.float32],
        range: np.ndarray[np.float32],
        depth: int,
    ) -> Union[DecisionTreeClassifier, None]:
        if len(domain) != len(range):
            return None
        self.learners = [None] * depth
        self.alphas = [None] * depth
        distribution: np.ndarray[np.float32] = self.uniform_dist(len(domain))
        for i in range(depth):
            self.learners[i] = DecisionTreeClassifier(random_state=1)
            self.learners[i].fit(domain, range, sample_weight=distribution)
            prediction: np.ndarray[np.float32] = self.learners[i].predict(
                domain
            )
            errors = self.errors(prediction, range)
            self.alphas[i] = self.alpha(distribution, errors)
            distribution = self.update_distribution(
                distribution, errors, self.alphas[i]
            )

    def errors(
        self,
        prediction: np.ndarray[np.float32],
        range: np.ndarray[np.float32],
    ) -> Union[np.ndarray[np.float32], None]:
        if len(prediction) != len(range):
            return None
        errors = np.zeros(len(prediction), dtype=np.float32)
        for i in range(len(errors)):
            errors[i] = prediction[i] * range[i]
        return errors

    def update_distribution(
        self,
        distribution: np.ndarray[np.float32],
        errors: np.ndarray[np.float32],
        alpha: np.float32,
    ) -> Union[np.ndarray[np.float32], None]:
        if len(distribution) != len(errors):
            return None
        result = np.zeros(len(distribution), dtype=np.float32)
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
        error = np.dot(distribution, errors)
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


if __name__ == "__main__":
    pass
