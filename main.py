import numpy as np
import random
import matplotlib.pyplot as plt

dianshu = 40
#生成训练点
x = []
y = []
for i in range(0,dianshu):
    x.append(i)
    y.append(-8 * i + 1000 + random.random() * 200 - 100)
x = np.array(x)
y = np.array(y)

#打印训练点
plt.plot(x,y,'ro')

#最小二乘法求k1、b1
x_mean = x.mean()
y_mean = y.mean()
k1 = (((x - x_mean) * (y - y_mean)).sum()) / (((x - x_mean) * (x - x_mean)).sum())
b1 = y_mean - k1 * x_mean

#打印最小二乘法直线
x_plot1 = [0,dianshu]
y_plot1 = [b1,dianshu * k1 + b1]
plt.plot(x_plot1,y_plot1,'g')

#梯度法求k2、b2
k2 = 0
b2 = 0
xuexixiaolv = 0.0001
#两种学习终止条件任选
'''
#以学习次数作为限制
xuexicishu = 100000
for i in range(0,xuexicishu):
    k2 = k2 - xuexixiaolv * ((((k2 * x + b2 - y) * x).sum()) / 2)
    b2 = b2 - xuexixiaolv * (((k2 * x + b2 - y).sum()) / 2)
'''
#以学习条件作为限制
cishu = 0
while (abs(xuexixiaolv * ((((k2 * x + b2 - y) * x).sum()) / 2)) > 0.000001) or (abs(xuexixiaolv * (((k2 * x + b2 - y).sum()) / 2)) > 0.000001):
    k2 = k2 - xuexixiaolv * ((((k2 * x + b2 - y) * x).sum()) / 2)
    b2 = b2 - xuexixiaolv * (((k2 * x + b2 - y).sum()) / 2)
    cishu = cishu + 1
print('学习',cishu,'次后达到精度目标')

#打印梯度法直线
x_plot2 = [0,dianshu]
y_plot2 = [b2,dianshu * k2 + b2]
plt.plot(x_plot2,y_plot2,'b')

print('最小二乘法的训练结果：\tk1=',k1,'b1=',b1)
print('梯度法的训练结果:\tk2=',k2,'b2=',b2)
#图像显示
plt.show()
