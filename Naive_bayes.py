from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
iris = load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y)

gnb = GaussianNB()
gnb.fit(x_train,y_train)

##
y_pred = gnb.predict(x_test)

cnf = metrics.confusion_matrix(y_test,y_pred)
print cnf

accuracy = metrics.accuracy_score(y_test,y_pred)
print accuracy






