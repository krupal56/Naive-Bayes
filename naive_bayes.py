import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

data = pd.read_csv('train.csv')

data['Sex_new'] = np.where(data['Sex']=='male',0,1)

data['Embarked_new'] = np.where(data['Embarked']=='Q',0, np.where(data['Embarked']=='S',1, np.where(data['Embarked']=='C',2,3)))

data = data[['Survived','Pclass','Sex_new','Age','SibSp','Parch','Fare','Embarked_new']] 

data =data.dropna()

x = data[['Pclass','Sex_new','Age','SibSp','Parch','Fare','Embarked_new']]
y = data['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y)
#Guassian naive-bayes
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)

#confusion matrix
cnf = metrics.confusion_matrix(y_test,y_pred)  
print cnf

#accuracy 
accuracy = metrics.accuracy_score(y_test,y_pred)
print accuracy

