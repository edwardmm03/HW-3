from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

'''
#loading the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=99)

#starting the model by creating an object for the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=1)

#training the model
clf.fit(X_train, y_train)

#making predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
'''

X= [[0],[1],[2]]
Y = [1,2,1]

clf = DecisionTreeClassifier(random_state=1)
clf.fit(X,Y,sample_weight=[1,2,3])
y_pred = clf.predict(X)
accuracy = accuracy_score(Y, y_pred)
print(f'Accuracy: {accuracy}')