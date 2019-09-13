import matplotlib.pyplot as plt
import numpy as np
import sys

# from matplotlib.animation import FuncAnimation
#
#
# from PIL import Imagefrom matplotib.ticker import PercentFormatter
# from matplotlib import colors
#
#





# # plt.scatter([1,1,1,1], [1,4,2,2])
# plt.ylabel("")
# plt.show()
#
# t = np.random.randn(500)
# plt.hist(t,bins = 1000000)
# plt.xlabel("input")
# plt.axis([-10,10,1,1000000])
# plt.show()
# print(t)
def f( x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


x = np.linspace(0,10,50)
y =  np.linspace(0,10,50)

X,Y = np.meshgrid(x,y)

plt.contour(X,Y,f(X,Y), colors = 'black')
plt.show()
