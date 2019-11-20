import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from scipy import misc

im = np.array(Image.open("Paolina.tiff"))
print(im.shape)

normalized_im = im/255.0
print(normalized_im)

# plt.imshow(normalized_im, cmap = 'gray')

S_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
S_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
# plt.show()


partial_imx = signal.convolve2d(normalized_im, S_x, mode = "same", boundary = "fill")
partial_imy = signal.convolve2d(normalized_im, S_y, 'same')
fig = plt.figure()
#
gs = fig.add_gridspec(1,4)
ax1 = fig.add_subplot(gs[:, 0])
plt.imshow(partial_imx, cmap = "gray")
# plt.imshow(partial_imx)
ax2 = fig.add_subplot(gs[:, 1])
plt.imshow(partial_imy, cmap = "gray")

ax3 = fig.add_subplot(gs[:, 2])
plt.imshow(np.subtract(partial_imx, partial_imy), cmap = 'gray')

ax4 = fig.add_subplot(gs[:, 3])
plt.imshow(np.subtract(partial_imx, partial_imx))
plt.colorbar()
plt.show()

# from scipy import ndimage.gaussian_filter
grad_magnitude = np.sqrt(np.square(partial_imx) + np.square(partial_imy))

threshold = np.mean(grad_magnitude)
plt.imshow(grad_magnitude, cmap = 'gray')
plt.show()
n, p = grad_magnitude.shape
print(grad_magnitude.shape)
horizontal_indices = np.arange(0, n )
vertical_indices = np.arange(0, p)

regularized_grad_magnitude = np.zeros((n,p))
for i in range(n):
    for j in range(p):
        if grad_magnitude[i][j] > 2*threshold:
            regularized_grad_magnitude[i][j] = 1
        else:
            regularized_grad_magnitude[i][j] = 0
# regularized_grad_magnitude = 1 if (grad_magnitude[horizontal_indices][vertical_indices] > 2*threshold) else 0
# plt.imshow(regularized_grad_magnitude, cmap = 'gray')
# plt.show()

m, n = im.shape; xs = np.arange(0, n); ys = np.arange(0, m);

xpeaks = (im[ys][:,xs-1] <= im[ys][:,xs]) & (im[ys][:,xs] > im[ys][:,xs+1])
ypeaks = (im[ys-1][:, xs] <= im[ys][:, xs]) & (im[ys][:, xs] > im[ys+1][:, xs])

peaked_magnitudes = np.sqrt(np.square(xpeaks) + np.square(ypeaks))

plt.imshow(peaked_magnitudes)
plt.show()
