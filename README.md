# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and separate input features and target output.

2.Convert data into numerical form and apply feature scaling.

3.Initialize the parameters (θ) and add a bias term to the input matrix.

4.Compute predictions and update parameters using gradient descent.

5.Predict the output for new input data and transform it back to original scale.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: PRIYAN V
RegisterNumber: 212224230211

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1, y, learning_rate=0.01, num_iters=100):
    x = np.c_[np.ones(len(x1)), x1]
    theta = np.zeros(x.shape[1]).reshape(-1,1)
    for i in range(num_iters):
        predictions = (x).dot(theta).reshape(-1,1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(x1)) * x.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
print("Name: SIBHIRAAJ R \nReg.no: 212224230268")
print(data.head())
x = (data.iloc[1:, :-2].values)
print(x)
x1=x.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled = scaler.fit_transform(x)
y1_scaled = scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
theta = linear_regression(x1_scaled, y1_scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="663" height="854" alt="image" src="https://github.com/user-attachments/assets/aea4c20e-0009-4446-895f-8fc9b7d822d6" />
<img width="602" height="858" alt="image" src="https://github.com/user-attachments/assets/501f2247-2250-445f-a817-8661c475db53" />
<img width="645" height="872" alt="image" src="https://github.com/user-attachments/assets/28fbed02-27f2-4035-848a-7b52eb3f3db5" />
<img width="677" height="869" alt="image" src="https://github.com/user-attachments/assets/3349b0d7-490f-46a9-8e40-8a26c6f6f244" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
