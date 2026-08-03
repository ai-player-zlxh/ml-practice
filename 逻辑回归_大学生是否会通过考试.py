#0考试不通过，1考试通过

import numpy as np
import matplotlib.pyplot as plt


def logistic_regerssion(x,y,w,b,learning_rate = 0.01,literation = 1000):
    m,n = x.shape
    for liter in range(literation):
       dj_dw = np.zeros(n)
       dj_db = 0
       for i in range(m):
           z = np.dot(w,x[i])+b
           f = 1/(1+np.exp(-z))
           error = f-y[i]
           for j in range(n):
               dj_dw[j] += error*x[i][j]
           dj_db += error
           #梯度下降
       w -= learning_rate*dj_dw/m
       b -= learning_rate*dj_db/m
    #计算交叉熵损失
       k = 0
       for i in range(m):

           z = np.dot(w,x[i])+b
           f_w_b = 1/(1+np.exp(-z))
           #
           eps = 1e-8
           k += -y[i]*np.log(f_w_b+eps)-(1-y[i])*np.log(1-f_w_b+eps)
           loss = k/m
       if liter%500 == 0:
           print(f"迭代 : {literation:4d}：损失值：{loss:4f}")
    return w, b, loss
#预测函数
def predict(x,w,b):
    m = x.shape[0]
    pred = np.zeros(m)
    for i in range(m):
       z = np.dot(w,x[i])+b
       f = 1/(1+np.exp(-z))
       pred[i] = 1 if f>0.5 else 0
    return pred

#study_hours	assignment_rate	attendance	mock_score	sleep_hours	tutoring
x = np.array([
             [2,60,70,55,5,0],
             [8,95,98,90,7,1],
             [4,70,80,65,6,0],
             [7,85,90,82,7,1],
             [3,50,60,45,5,0],
             [6,90,95,88,8,1]
])
y = np.array([0,1,0,1,0,1])
w = np.zeros(6)
b = 0
#特征缩放
mu = np.mean(x)
sigma = np.std(x)
x_scaled = (x-mu)/sigma

w_final, b_final, loss = logistic_regerssion(x_scaled,y,w,b,learning_rate = 0.01,literation = 5000)
print(w_final)
print(b_final)
y = predict(x_scaled,w_final,b_final)
print(y)