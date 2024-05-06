# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: vishal s
RegisterNumber:  212223240184
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Result output:


![image](https://github.com/23013753/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145634121/c5f7df09-1e1e-4a78-9758-e10abb8f148a)


data.head():


![image](https://github.com/23013753/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145634121/5b53d2eb-4811-4abc-9079-fd05b9df765b)



data.info():


![image](https://github.com/23013753/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145634121/9391f910-07cf-45ac-8eb1-4aac316b1390)


data.isnull().sum():



![image](https://github.com/23013753/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145634121/5a7433a8-502f-46f1-8c68-453d1121f780)



Y_prediction value:


![image](https://github.com/23013753/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145634121/75729f8b-1760-46b8-89a2-7f8a8171c33b)


Accuracy value:


![image](https://github.com/23013753/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145634121/c5423de3-7e45-4eff-8deb-dded1b792b63)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
