import os
print(os.getcwd())
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)
print(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
def gaborfilter(theta, wavelength, phase, sigma, aspect, ksize=None):

	"""
	GB = GABORFILTER(THETA, WAVELENGTH, PHASE, SIGMA, ASPECT, KSIZE)
	creates a Gabor filter GB with orientation THETA (in radians),
	wavelength WAVELENGTH (in pixels), phase offset PHASE (in radians),
	envelope standard deviation SIGMA, aspect ratio ASPECT, and dimensions
	KSIZE x KSIZE. KSIZE is an optional parameter, and if omitted default
	dimensions are selected.
	 """

	if ksize is None:
	 	ksize = 8*sigma*aspect


	if type(ksize) == int or len(ksize) == 1:
	 	ksize = [ksize, ksize]


	xmax = np.floor(ksize[1]/2.)
	xmin = -xmax
	ymax = np.floor(ksize[0]/2.)
	ymin = -ymax

	xs, ys = np.meshgrid(np.arange(xmin,xmax+1), np.arange(ymax,ymin-1,-1))

	# rotation_matrix = np.array([[np.cos(theta), np.sin(theta) ], [-np.sin(theta), np.cos(theta)]])
	x = xs*np.cos(theta) + ys*np.sin(theta)
	y = -xs*np.sin(theta) + np.cos(theta)*ys

	gaussian = np.exp(-(x**2/aspect**2 + y**2)/(2*sigma**2))
	sinusoid = np.sin(2*np.pi/wavelength *y + phase)
	# Gabor = np.sin(((2*np.pi*new_ys)/4*sigma) + np.pi/2)*np.exp(((-new_xs**2)/(aspect**2) + new_ys**2)/2*sigma**2)

	Gabor = sinusoid * gaussian
	gabor_mean = np.mean(Gabor, dtype = 'float64')
	Gabor = Gabor - gabor_mean
	sum_of_squares = np.sqrt(np.sum(Gabor**2))
	Gabor = Gabor/sum_of_squares
	return Gabor
	# vector_matrix = np.array(list(zip(xs, ys)))
	# new_coords  = np.matmul(rotation_matrix, vector_matrix)
if __name__== "__main__":
	# fig = plt.figure()
	# gs = fig.add_gridspec(2,4)
	# for i in range(16):
	#
	# 	if i <=7:
	# 		test = gaborfilter(theta = i* np.pi/4, wavelength = 4, phase = 0, sigma  = 1, aspect = 2, ksize = 8)
	# 		plt.subplot(4, 4, i+1)
	# 		plt.imshow(test, cmap = "gray")
	# 	elif i<=11:
	# 		test = gaborfilter(theta = (i-8)* np.pi/4, wavelength = 4, phase = np.pi/2, sigma  = 1, aspect = 2, ksize = 8)
	# 		plt.subplot(4, 4, i+1)
	# 		plt.imshow(test, cmap = "gray")
	# 	elif i <=16:
	# 		test = gaborfilter(theta = (i-12)* np.pi/4, wavelength = 4, phase = 3*np.pi/2, sigma  = 1, aspect = 2, ksize = 8)
	# 		plt.subplot(4,4,i+1)
	# 		plt.imshow(test, cmap = 'gray')
	#
	# plt.show()
	#
	#
	# im = np.array(Image.open("Paolina.tiff"))
	# im = im.astype('float')/255
	# plt.imshow(im)
	# plt.show()
	# fig = plt.figure(figsize = (10,10))
	# for i in range(16):
	#
	# 	if i <=7:
	# 		test = gaborfilter(theta = i* np.pi/4, wavelength = 4, phase = 0, sigma  = 1, aspect = 2)
	# 		plt.subplot(4, 4, i+1)
	# 		plt.imshow(signal.convolve2d(im, test, mode = 'same'), cmap = 'gray')
	# 	elif i<=11:
	# 		test = gaborfilter(theta = (i-8)* np.pi/4, wavelength = 4, phase = np.pi/2, sigma  = 1, aspect = 2)
	# 		plt.subplot(4, 4, i+1)
	# 		plt.imshow(signal.convolve2d(im,test, mode = 'same'), cmap = 'gray')
	# 	elif i <=16:
	# 		test = gaborfilter(theta = (i-12)* np.pi/4, wavelength = 4, phase = 3*np.pi/2, sigma  = 1, aspect = 2)
	# 		plt.subplot(4,4,i+1)
	# 		plt.imshow(signal.convolve2d(im,test, mode = 'same'), cmap = 'gray')
	#
	# plt.show()
	# # for i in range(16):
		# scipy.signal.convolve2d(im, gaborfilter)


	circle = np.array(Image.open('circle.png'))
	print(circle.shape)
	circle = circle [:,:,2]

	print(circle.shape)

	plt.imshow(circle, cmap = 'gray')
	plt.show()
# part b

	# im = Image.open('Paolina.tiff')
