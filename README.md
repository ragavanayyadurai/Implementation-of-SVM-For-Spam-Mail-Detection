# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
```
Program to implement the SVM For Spam Mail Detection.
Developed by: Ragavendran A
RegisterNumber:  212222230114

```
```python
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
```
```python
import pandas as pd
data= pd.read_csv("/content/spam.csv",encoding='Windows-1252')
```
```python
data.head()
```
```python
data.info()
```
```python
x=data["v1"].values
```
```python
y=data["v2"].values
```
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```
```python
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```python
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
```python
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/KothaiKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121215739/9957f67e-d5dd-4ab6-8696-9fe2ce24fe76)

![image](https://github.com/KothaiKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121215739/9b39773b-5160-4652-ad5b-2cc6181c3b68)

![image](https://github.com/KothaiKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121215739/914644f3-7348-44ac-8d01-3fb03ca3eb34)

![image](https://github.com/KothaiKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121215739/68e3f480-70d2-4d92-abb4-69f45fd2e8b0)

![image](https://github.com/KothaiKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121215739/bf0c77e5-f0f5-48dc-990e-2ac5f6e6d4a8)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
