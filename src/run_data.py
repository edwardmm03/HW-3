from sklearn import datasets
from ada_boost import AdaBoostDTC, AdaBoostGBC
import numpy as np
import matplotlib.pyplot as plt

def even_odd(arr: np.ndarray[np.int64]) -> np.ndarray[np.int32]:
    result = np.zeros(len(arr), dtype=np.int32)
    for i in range(len(arr)):
        if arr[i] % 2 ==0:
            result[i] = 1
        else:
            result[i] = -1
    return result

if __name__ == "__main__":
    digits = datasets.load_digits()
    data: np.ndarray[np.ndarray[np.float64]] = digits["data"]
    target: np.ndarray[np.int64] = digits["target"]
    
    #Ada Boost on a Desicion Tree Classifier
    learner = AdaBoostDTC()
    DTCErrors = learner.train(data, even_odd(target), 7)
    #print(DTCErrors) #debugging print
   
    #Ada Boost on a Gradient Boosting Classifier
    learner = AdaBoostGBC()
    GBCErrors = learner.train(data, even_odd(target),7)
    #print(GBCErrors)
    
    #plotting the errors of the learners during boosting
    xaxis = [0,1,2,3,4,5,6]
    plt.plot(xaxis,DTCErrors, label = "DTC Classifier")
    plt.plot(xaxis,GBCErrors, label = "GBC Classifier")
    plt.legend()
    plt.xlabel('Interations of Boosting')
    plt.ylabel('Error of Classifier')
    plt.show()
