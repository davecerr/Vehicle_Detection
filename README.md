# Vehicle Detection

![Alt Text](https://media.giphy.com/media/SJ9ySFtqGbQSifywwS/giphy.gif)

Vehicle Detection is performed using a Linear SVM classifier that slides over each frame at multiple scales to detect vehicles at different distances. The classifier itself is trained on a labelled dataset (car vs non-car) using feature vectors comprised of hisotgrams of oriented gradients (HOG), colour histograms and spatial binning. More details are available in the writeup.md file.

This is combined with a previous project on lane line detection to produce the above video output.
