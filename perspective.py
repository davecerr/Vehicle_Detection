#############PERSPECTIVE TRANSFORM: BIRDS' EYE VIEW###################

#We will detect lane lines from normal (front facing) perspective but our ultimate goal here is to 
#detect the curvature of the lane as this can be used to decide on the steering angle.
#This requires us to fit a polynomial curve to the lane lines and this is easier from a birds' eye view.
#Once we have the polynomial equations, it's simple calculus to find curvature.

#Perspective is the phenomenon where objects appear smaller the farther away they are.
#Depending on your perspective, parallel lines will converge to different points.

#A perspective transform changes this by dragging points away from or towards the camera. 

#Why much easier to fit polynomial from birds' eye view? Well more just that it's really hard from front
#facing view: left line looks bending to right and right line looks bending to left. From birds' eye view, it 
#is clear the lines are parallel and it is clear the direction they are bending in. 
#Birds' eye view also allows easy comparison with maps which is good for establishing location.

#How does it work? Well we want to preserve straight lines whilst dragging points about the image. The perspective 
#transform itself will be an affine transformation (2x2 matrix). This has 4 elements so we'll need 4 simultaneous eqns.
#To construct these, we pick 4 points that we know lie on a rectangle in the original image and we then supply coords
#for 4 points in the warped image for where we want these to end up (will lie on standard rectangle i.e. vertical and horizontal
#edges). Then use cv2 functions to find the required matrix (solve the system of 4 simultaneous eqns for us).

#We will again work with chessboard images for this as they are easy to establish rectangular regions on. Once we find the 
#warp matrix, this can then be applied to any image passing through the lens - providing we can approximate a rectangular region on it
#This will require us to draw a kind of trapezoidal bounding box around the visible portion of the lane line (more on that later)


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

#ONLY NEED 4 POINTS FOR 2X2 WARP MATRIX SO USE OUTERMOST CORNERS

def corners_unwarp(img, nx=9, ny=6, mtx=mtx, dist=dist):
    undist = undistort_image(img)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M_warp = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M_warp, img_size)

        return warped, M_warp


#image = cv2.imread('./camera_cal/calibration20.jpg')
#warped, M_warp = corners_unwarp(image)
#pickle.dump(M_warp, open("M_warp.p", "wb"))
#print("Warp matrix has been pickle dumped.")


def birds_eye_new_matrix(image, already_undistorted, source_vert):
    assert len(source_vert) == 4
    print("Vertices of rectangular region have been loaded")
    if already_undistorted == False:
        undist = undistort_image(image)
        print("Lens distortion removed from image")
    else:
        undist = image
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    offset = 300
    #destination_vert = np.float32( [[offset, img_size[1]], [offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]] ])
    destination_vert = np.float32([[(img_size[0]/4),0], [(img_size[0]/4), img_size[1]], [(img_size[0]*3/4), img_size[1]], [(img_size[0]*3/4), 0]])
    M_warp = cv2.getPerspectiveTransform(source_vert, destination_vert)
    M_inv = cv2.getPerspectiveTransform(destination_vert, source_vert)
    pickle.dump(M_warp, open("M_warp.p", "wb"))
    pickle.dump(M_inv, open("M_inv.p", "wb"))
    print("Warp matrix and its inverse created and dumped to pickle file")
    warped = cv2.warpPerspective(undist, M_warp, img_size)
    cv2.polylines(warped, [destination_vert.astype(int).reshape((-1,1,2))], True, (255,0,0), 6)
    print("Perpsective changed to bird's eye view")
    return warped, M_warp



def birds_eye_existing_matrix(image, already_undistorted, source_vert):
    assert len(source_vert) == 4
    print("Vertices of rectangular region have been loaded")
    if already_undistorted == False:
        undist = undistort_image(image)
        print("Lens distortion removed from image")
    else:
        undist = image
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    offset = 300
    #destination_vert = np.float32( [[offset, img_size[1]], [offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]] ])
    destination_vert = np.float32([[(img_size[0]/4),0], [(img_size[0]/4), img_size[1]], [(img_size[0]*3/4), img_size[1]], [(img_size[0]*3/4), 0]])
    print(source_vert)
    print(destination_vert)
    M_warp = pickle.load(open("M_warp.p", "rb"))
    print("Warp matrix loaded from existing pickle file")
    warped = cv2.warpPerspective(undist, M_warp, img_size)
    cv2.polylines(warped, [destination_vert.astype(int).reshape((-1,1,2))], True, (255,0,0), 6)
    print("Perpsective changed to bird's eye view")
    return warped, M_warp





image = cv2.imread('./test_images/straight_lines1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
img_size = (gray.shape[1], gray.shape[0])
src_vert = np.float32([[(img_size[0]/2)-55, (img_size[1]/2)+100], [(img_size[0]/6)-10, img_size[1]], [(img_size[0]*5/6)+40, img_size[1]], [(img_size[0]/2)+60, (img_size[1]/2)+100] ])
print("test1 image loaded from disc")
#warped = birds_eye(image, already_undistorted = False, source_vert = np.float32([ [300, 675], [600, 460], [720, 460], [1085, 675] ]))[0]
warped = birds_eye_existing_matrix(image, already_undistorted = False, source_vert = src_vert)[0]
print("Image warp takes place before polygon drawn on original image thus avoiding blurred box in bird's eye image")
cv2.polylines(image, [src_vert.astype(int).reshape((-1,1,2))], True, (255,0,0), 6)
print("Lane line polygon drawn on original image after warped image completed")
#Plot original camera image (distorted) next to the calibrated (undistorted) image
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(warped)
ax2.set_title('Undistorted and Warped Image', fontsize = 50)
plt.subplots_adjust(left=0., right = 1, top=0.9, bottom=0.)
plt.savefig('test1_undistorted_and_warped.png')
plt.close()
print('You can observe the undistorted and warped image in test1_undistorted_and_warped.png')
