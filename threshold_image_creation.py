import numpy as np 
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

mtx=pickle.load(open("mtx.p", "rb"))
dist=pickle.load(open("dist.p", "rb"))

def undistort_image(img, mtx=mtx, dist=dist):
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist

def binary_image_creation(image, already_undistorted, s_thresh=(170,255), sx_thresh=(20,100)):
	if already_undistorted == False:
		undist = undistort_image(image)
		print("Lens distortion removed from image")
	else:
		undist = image
	img = np.copy(undist)

	#Convert to HLS colour space and isolate the L and S channels
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	print("Image converted form RGB to HLS colour space")

	#Gradient thresholding is done on L channel
	sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) #Take hte derivative in x (better for detecting vertical structures e.g. lane lines)
	abs_sobelx = np.absolute(sobelx) #Absolute x derivative accentuates lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	#Threshold x gradient and map onto a copy of image that will be presented in one layer of final mask
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	print("Gradient thresholding complete")

	#Threshold color channel is done on S channel and map onto a copy of image that will be presented in one layer of final mask
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	print("Colour thresholding complete")

	#Stack the different masks into different channels of an RGB image. This is just for visual effect.
	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) *255
	print("Thresholding results displayed in color_binary. Green shows gradient thresholding and blue shows colour thresholding.")

	#Stack the different masks into the same channel. This grayscale image is what we actually use
	overall_binary = cv2.cvtColor(color_binary, cv2.COLOR_RGB2GRAY)
	overall_binary[ overall_binary > 0] =255
	print("Grayscale mask created. Activated pixels show potential regions identified using EITHER gradient OR colour thresholding")

	return overall_binary, color_binary


image = cv2.imread('./test_images/test1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("test1 image loaded from disc")
thresh_gray, thresh_colour = binary_image_creation(image, already_undistorted = False)

#Plot original camera image (distorted) next to the calibrated (undistorted) image
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(thresh_colour)
ax2.set_title('Thresholding', fontsize = 50)
ax3.imshow(thresh_gray, cmap='gray')
ax3.set_title('Lane Detection', fontsize = 50)
plt.subplots_adjust(left=0., right = 1, top=0.9, bottom=0.)
plt.savefig('test1_thresh.png')
plt.close()
print('You can observe the lane detection in test1_thresh.png')
