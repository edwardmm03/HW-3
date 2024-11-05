from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import math


def ada_boost() -> None:
    pass


if __name__ == "__main__":

    clf: DecisionTreeClassifier = DecisionTreeClassifier(random_state=1)

    training_x = [[1], [2], [3], [4], 5]
    training_y: list[int] = [2, 4, 6, 8, 10]
    test_x: list[list[float]] = [[0.5], [6], [7], [8], [10]]
    test_y: list[int] = [1, 12, 14, 16, 20]

    m = len(training_x)
    distribution = []

    # setting the initial distribution
    for i in training_x:
        distribution.append(1 / m)

    # adaboost algorithm
    for x in range(1, 5):
        # ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (5,) + inhomogeneous part.
        # detected shape was (5,) + inhomogeneous part.
        clf.fit(training_x, training_y, sample_weight=distribution)
        yhat = clf.pred(test_x)
        accuracy = accuracy_score(test_y, yhat)
        error = 1 - accuracy
        # calculate alphat
        at = 0.5 * (math.log2((1 - error) / error))
        # calculate normalization factor
        zt = 2 * pow(error * (1 - error), 0.5)
        for i in m:
            # update the distribution
            distribution[i] = (
                pow(distribution[i], -at * test_y[i] * yhat[i])
            ) / zt

    yhat = clf.pred(test_x)
    accuracy = accuracy_score(test_y, yhat)
    error = 1 - accuracy
    print(error)
