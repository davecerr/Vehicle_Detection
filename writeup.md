## Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* We also apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* Combine with Advanced Lane Detection

[//]: # (Image References)
[image0]: ./writeup_images/dataset.png
[image1]: ./writeup_images/HOG.png
[image2]: ./writeup_images/spatial_binning.png
[image3]: ./writeup_images/colour_histogram.png
[image4]: ./writeup_images/normalised_features.png
[image5]: ./writeup_images/sliding_windows.png
[image6]: ./writeup_images/distant_windows.png
[image7]: ./writeup_images/middle_windows.png
[image8]: ./writeup_images/close_windows.png
[image9]: ./writeup_images/multi_distance_search.png
[image10]: ./writeup_images/heatmap.png
[image11]: ./writeup_images/heatmap_threshold.png
[image12]: ./writeup_images/detection_with_heatmap.png
[video1]: output_project_video_lanes_and_vehicles2.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Dataset

We will train a Linear SVM classifier on a labelled dataset of car vs. not-car images. There are 8792 car images and 8968 non-car images. A sample is shown below:

![alt text][image0]

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cell 4 of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

#### 2. Spatial Features and Colour Histograms

I supplement the HOG features by converting each image to YCrCb space and summing the value of each colour channel for every pixel in the image. We can see the different results for car vs. not-car images below:

![alt text][image2]

Additionally, for each image, we produce a colour histogram. Across each colour channel, each pixel can take an integer value in the range [0,255] i.e. one of 256 possible values. We shall use 32 histogram bins to split these 256 possible values into 32 bins each with a range of 256/32 = 8 possible values. We can then produce a histogram for each colour channel as follows:

For e.g. the 'Y' channel, we count how many pixels in the image have a 'Y' value in the range [0,7], then how many pixels have a 'Y' value in the range [8,15] etc. This is done for the chroma 'Cr' and 'Cb' channels also.

At the end of this, we have three histograms, one for each colour channel. We concatenate them into a single feature vector. Note that if we plot this, it will now just be a single graph rather than three separate histograms. We can see the difference in colour histograms for car vs. non-car images below.

![alt text][image3]

The final feature vector for each image is then a concatenation of HOG features, spatial binning features and colour histogram features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features and color features.

Next, I made sure to normalise the feature vectors using an object of the RobustScaler class. This works well in dealing with outliers. It's especially important here if we are extracting three different types of features and concatenating them since they may be on different scales. Indeed, we can see the importance of this below:

![alt text][image4]

I then shuffled the dataset (to remove any ordering that may have been present) and subsequently split it in an 8:2 ratio into a training set for the SVM classifier to learn from and a test set that we can keep hidden until training is complete and which we can then use to measure performance on.

In cell 11 of the IPython Notebook, I train the SVC. We can see it achieves a 99.35% accuracy on the test set which is very encouraging.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code to run a sliding window search over the image is in cell 12 of the attached IPython Notebook. Essentially a 64x64 window is slid over the image and at each position, we perform feature extraction and run it through our SVM classifier to ascertain whether or not that window contains a car. 

We can be a little bit clever here and speed up the algorithm by restricting the search region to those y-values that are likely to contain a vehicle. For example, there is not point search for cars on the car bonnet or in the sky. Therefore, we restrict to a y-value range of y=400 to y=650. Running this, we obtain the following:

![alt text][image5]

This actually appears to do a pretty good job in detecting these two cars. However, in this frame both vehicles are at a "middle distance." Our algorithm might struggle if cars are far away or close up since in these cases they will appear much bigger or much smaller and therefore they won't fit perfectly inside our 64x64 search window.

To combat this, we shall perform the sliding window search using different sized windows. This is easy enough to implement since we can just adjust the scale of our searching boxes. We will use three such scales to make the algorithm sensitive to vehicles at different distances. In the three images below, I display the positions of all the search boxes that will be used at all three scales:

![alt text][image6]
![alt text][image7]
![alt text][image8]

Searching across all three scales produces the following result:

![alt text][image9]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap as follows:

![alt text][image10]

To filter out false positives, we can threshold this heatmap and demand that a region of pixels be covered by at least 2 bounding boxes before we consider it a valid vehicle location. 

`scipy.ndimage.measurements.label()` is used to identify individual active regions in the heatmap:

![alt text][image11]

These are then assumed to correspond to a vehicle and a bounding boxes is drawn around that corresponding region on the original image:

![alt text][image12]

### Using a Deque To Store Heatmap History

Running the above pipeline on a video led to very "jumpy" bounding boxes. This kind of uncertainty in vehicle position is unsafe for a self-driving car and one can imagine that it would be even worse the faster vehicles are travelling (since they cross more pixels every frame), adding an extra danger to high-speed self-driving.

To deal with this, we can create a deque to store a history of the previous 10 heatmaps. For each frame, we will then sum all 10 heatmaps and then demand that a region be active in at least 7 of them before we draw a bounding box around it.

The effect is to smooth the bounding boxes. Instead of drawing small bounding boxes that jump around a lot, we will get larger bounding boxes since the active regions are present over ten frames and therefore get "smeared" out as the car travels through our field of view. These larger bounding boxes follow the car much more accurately.

As an added bonus, we get an extra layer of protection against false positives since it would need to be present in 7 out of the last 10 frames to qualify for a bounding box. 

### Combine With Lane Detection

For fun, I decided it would be nice to combine this with the previous lane detection project. This was relatively simple as I just had to merge the two notebooks. The only tricky part was to get the output of both pipelines to display at the same time. I also added a colour heatmap on top of a grayscale version of the original iamge at the top to exhibit vehicle detection in action!

Here is the final video:

![Alt Text](https://media.giphy.com/media/fnygNfQpZHFPuyTRKG/giphy.gif)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I thought about restricting the range of x coordinates that we search over to avoid "false positives" in oncoming traffic. I decided against this for two reasons; firstly, it's not really a fals positive since it is important to know about oncoming traffic, and secondly, if we want to run this algorithm in left-hand drive countries then we cannot just kill half of the screen! Although, a simple "if" statement at the top of our code could be used to handle left- vs right-hand drive countries.

I still think my pipeline might struggle with vehicles moving much faster than ourselves. This is because it might move through our field of view so rapidly that the activated heatmap regions don't overlap and thus it wouldn't meet the threshold of 7 out of the last 10 frames. This is a potential problem since fast moving vehicles are particularly dangerous. We would need to play around with the thresholding technique to handle this. We would also need suitable video footage to practice with.

