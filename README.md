# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_processed.mp4

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I selected my parameters by optimizing my classifier test accuracy. My final parameters are in `config.py` under `classifier`. See section 3 for more details. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Trained a linear SVM classifier in `classifier.py` in the `Classifier` class. The class takes a dictionary of parameters upon initialization which allowed me to iterate through different combinations of parameters and select the highest test accuracy. I didn't use a validation set which is not good practice and the hold out set effectively bled into my training set by doing this parameter selection using test set. However the model seems to be generalizing well so I didn't pursue this any further.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used four set of sliding windows (details in `config.py` under `process['window_parameters']`) in a narrow vertical band (400-580 pixels) and with 75% overlap. After running some experiments these numbers gave me the least amount of false positives. However the large overlap generated many more likely boxes that I countered with a higher threshold. I chose the narrow vertical band to only detect the most likely area that cars may appear in (e.g. not in the sky). This choice will backfire if the car is traveling up or down steep hills.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Searched using YCrCb 3-channel HOG features with spatially binned color and histograms of color in the feature vector.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Link video
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Function `find_center_box` (line 258 in `detection.py`) calculates center, width, height, and likelihood score for each box that is identified (initialized at 1). The function `average_boxes` (line 330 in `detection.py`) uses these values and a history from previous frames to determine if a box should be retained given its score (which is accumulated over the history). 

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I left the detected boxes (red) in so we can see how the algorithm detects a lot of boxes and then only shows a subset (blue) based on the detection methodology described earlier. The false positive rate is pretty high which is not ideal. The training dataset is not very large and we could probably do a better job of model selection and parameter tuning. I mentioned earlier that I didn't do cross-validation and used the testing data in my parameter selection which is not a good practice. It stands to reason that the current model will not generalize to conditions outside this setting and lighting, different road conditions, and changing scenery would gravely impact the detection results. The filtering algorithm is helping to some extent but I thing the model will be easily confused (will probably detect pedestrian, animals, landmarks as vehicles).
