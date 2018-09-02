#############GET CHESSBOARD CORNERS AND CALIBRATE/UNDISTORT CAMERA###################

#Recommended to use at least 20 images to obtain a reliable calibration
#Use glob API to read in all images of .jpg format

#We know the chessboard corners should appear rectangularly i.e. on a lattice.
#Currently there is a deviation due to lens distortion. We must correct for this 
#i.e. find affine matrix that maps current corners to desired (lattice point) corners

#Distortion correction may alter location of objects in image (more pronounced on larger chessboard)
#but itpreserves geometrical structures in the process. In fact, it makes the correct geometry manifest.

#The smaller chessboards show subtle changes - the lines are more straight if look closely

import numpy as np 
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


#Set number of inside corners for chessboard
nx=9
ny=6

#Get file names of calibration images
fnames = glob.glob('./camera_cal/calibration*.jpg')

#Arrays to store real object points as well as not real (distorted) image points
objpoints = []
imgpoints = []

#The real object points are the same for all images. We can prepare them as (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

#Loop through each image and calibrate camera
for i in range(len(fnames)):
	fname = fnames[i]
	img = cv2.imread(fname)

	#Convert to grayscale (cv2 read as BGR)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Find chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	print('Image '+str(i+1)+': chessboard corners identified')

	#If cornerns are found, do the following
	if ret == True:
		#append object points and images points to necessary array
		imgpoints.append(corners) # image locations of corners
		objpoints.append(objp) #actual locations of corners

		#draw and display the corners
		#cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
		#plt.imshow(img)
		#plt.savefig('./camera_cal/results/chess_cal_'+str(i)+'.png')
		#plt.close()
		
		# Get mtx (camera matrix) needed for this transform
		#Get dist (distortion coefficients) needed for this transform
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		pickle.dump(mtx, open("mtx.p", "wb"))
		pickle.dump(dist, open("dist.p", "wb"))

		#Use cv2.undistort to get destination (dst) image
		dst = cv2.undistort(img, mtx, dist, None, mtx)
		print('Image '+str(i+1)+': distortion removed')

		#Plot original camera image (distorted) next to the calibrated (undistorted) image
		f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
		f.tight_layout()
		ax1.imshow(img)
		ax1.set_title('Original Image', fontsize = 50)
		ax2.imshow(dst)
		ax2.set_title('Undistorted Image', fontsize = 50)
		plt.subplots_adjust(left=0., right = 1, top=0.9, bottom=0.)
		plt.savefig('./camera_cal/results/chess_cal_'+str(i+1)+'.png')
		plt.close()
		print('Image '+str(i+1)+': calibration image saved in camera_cal/results/')

