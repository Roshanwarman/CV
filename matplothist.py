import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

#
# N_points = 100000
# n_bins = 20
#
# # Generate a normal distribution, center at x=0 and y=5
# x = np.random.randn(N_points)
# y = .4 * x + np.random.randn(100000) + 5
#
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#
# axs[0].hist(x, bins=n_bins)
# axs[1].hist(y, bins=n_bins)
# plt.show()
#
#
# #
# # import time
# # import sys
# #
# # for i in range(5):
# #     print(i, end=" "),
# #     sys.stdout.flush()
#
#
#
#
# # import sys
# # for x in range(20000):
# #     print ("HAPPY >> %s <<\r" % str(x),)
# #     # sys.stdout.flush()
#
#
# # for i in range (1000):
# #     for j in range(1000):
# #         print(i + j)


# print(3/10)

points = np.random.randn(100,2)

plt.scatter(points[:,0], points[:,1], s = 5)
plt.show()


from PIL import Image

# from scipy.spatial.distance import pdist,squareform
from matplotlib.animation import FuncAnimation

import matplotlib
from mpl_toolkits.mplot3d import Axes3D


new = plt.figure()




ax = new.add_subplot(221, projection = '3d')

ax1 = new.add_subplot(222, projection = '3d')
ax2 = new.add_subplot(223, projection = '3d')
ax3 = new.add_subplot(224, projection = '3d')

plt.show()
# A = pdist(points)
# B = squareform(A)
#
# plt.imshow(B)
