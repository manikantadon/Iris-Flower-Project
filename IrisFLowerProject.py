import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline


# if there is no lables for the datasets use the below names
# columns=['Id','Sepal length','Sepal width','Petal length','Petal width','Class labels']

# importing dataset
df=pd.read_csv('Iris.csv')#only for google colab.

#printing dataset
#df

#details of dataset
#df.describe()

#plotting graphs for the dataset for different classes
#sb.pairplot(df,hue='Species')

# taking the values separately
#data=df.values
#data

# separating the independent values(inputs)
x=data[:,1:5]
#x

#separating the dependent values (outcomes)
y=data[:,5]
#y

# splitting data sets into training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)#splitting into training sets and testing tests

# support vector machine predicting model
from sklearn.svm import SVC
model_svc=SVC()
model_svc.fit(x_train,y_train)
prediction1=model_svc.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction1)*100)
# print the results
#for i in range(len(prediction1)):
#       print(y_test[i],prediction1[i])


# logistic regression predicting model
from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(x_train,y_train)
prediction2=model_LR.predict(x_test)
print(accuracy_score(y_test,prediction2)*100)

# decision tree classifing model
from sklearn.tree import DecisionTreeClassifier
model_DTC=DecisionTreeClassifier()
model_DTC.fit(x_train,y_train)
prediction3=model_DTC.predict(x_test)
print(accuracy_score(y_test,prediction3)*100)

# classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction1))

# testing with new data sets (new testing data)
x_new=[[1,1,1,1],[0.4,0.5,0.3,1],[2.3,5.5,3.3,3.3],[5.7, 2.8, 4.1, 1.3]]
prediction=model_svc.predict(x_new)
print(prediction)

