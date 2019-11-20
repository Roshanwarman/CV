import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


t = np.linspace(-5, 5, num = 100)

def sigmoid(x):
    return 1/(1+ np.exp(-2*x))


# plt.plot(t, sigmoid(t))

fig, ax1 = plt.subplots()


derivative = np.gradient(sigmoid(t))
ax1.plot(t, derivative, color = "red")
ax1.set_ylabel('derivative')
# plt.plot(t, derivative)

ax2 = ax1.twinx()
second_derivative = np.gradient(derivative)

ax2.plot(t, second_derivative)
# plt.plot(t, second_derivative)
plt.show()
