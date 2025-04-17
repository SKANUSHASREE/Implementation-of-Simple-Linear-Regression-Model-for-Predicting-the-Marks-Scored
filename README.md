# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: S KANUSHA SREE
RegisterNumber:  212224040149
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()
```
```
df.tail()
```
```
x=df.iloc[:,:-1].values
x
```
```
y=df.iloc[:,1].values
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
```
```
y_test
```
```
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train, regressor.predict(x_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test, regressor.predict(x_test), color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ", rmse)
```
## Output:

![image](https://github.com/user-attachments/assets/127d1303-31d3-4af1-9bb7-40e082fc6b1f)

![image](https://github.com/user-attachments/assets/fd2a9022-d35e-4b44-b8b5-74602279d611)

![image](https://github.com/user-attachments/assets/67865046-b9ef-458d-a879-88ff60b64601)

![image](https://github.com/user-attachments/assets/b71bcce4-7494-45a3-9874-d3d5f9257683)
![image](https://github.com/user-attachments/assets/04e70b33-ac07-432a-b2f7-e74f60bd907e)
![image](https://github.com/user-attachments/assets/753ed80e-d9b6-486f-8ff3-9075956bb617)
![image](https://github.com/user-attachments/assets/dc1ae8fd-d8b0-444f-8be1-f4357c22b0d4)
![image](https://github.com/user-attachments/assets/b32749c0-eefe-4ea7-9b47-6e42bef37dbe)
![image](https://github.com/user-attachments/assets/f5e7756e-86d1-4c61-a166-80b4b3ce76e8)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
