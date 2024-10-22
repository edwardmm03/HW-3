from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import math

if __name__ == "__main__":

    clf = DecisionTreeClassifier(random_state=1)

    training_x = [[1],[2],[3],[4],5]
    training_y = [2,4,6,8,10]
    test_x = [[.5],[6],[7],[8],[10]]
    test_y = [1,12,14,16,20]

    m = len(training_x)
    distribution = []

    #setting the initial distribution
    for i in training_x:
        distribution.append(1/m)

    #adaboost algorithm
    for x in range(1,5):
        clf.fit(training_x,training_y, sample_weight=distribution) #ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (5,) + inhomogeneous part.
                                                                   #detected shape was (5,) + inhomogeneous part.
        yhat = clf.pred(test_x)
        accuracy = accuracy_score(test_y,yhat)
        error = 1-accuracy
        at = .5*(math.log2((1-error)/error)) #calculate alphat
        zt = 2*pow(error*(1-error), .5) #calculate normalization factor
        for i in m:
            #update the distribution
            distribution[i] = (pow(distribution[i], -at*test_y[i]*yhat[i]))/zt

    yhat = clf.pred(test_x)
    accuracy = accuracy_score(test_y,yhat)
    error = 1-accuracy
    print(error)
