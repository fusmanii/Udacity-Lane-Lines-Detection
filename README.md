# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./solidYellowCurve.jpg "Filtered Image"
[image2]: ./solidYellowCurve2.jpg "Region of Interest"
[image3]: ./solidYellowCurve3.jpg "Canny Image"
[image4]: ./solidYellowCurve4.jpg "Line Drawn"

---

## Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

**My pipeline consisted of 5 steps.**

####a) Filter out all the colours except for white and yellow. 
RGB was used to filter out the white but for yellow HSV worked a lot better.
 
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

image = mpimg.imread('test_images/solidWhiteRight.jpg')

yellow = filter_colour_hsv(image, [20, 10, 10], [30, 250, 255])
white = filter_colour_rgb(image, [200, 200, 200], [255, 255, 255])
```
```python
def filter_colour_rgb(image, lower, upper):
  """
  Returns a filtered image with colours that fall into the provided range.
  upper and lower must be RGB values.
  """
  
  # create numpy arrays for openCV inRAnge function
  lower_limit = np.array(lower)
  upper_limit = np.array(upper)
  
  mask = cv2.inRange(image, lower_limit, upper_limit)
  return cv2.bitwise_and(image, image, mask=mask)

def filter_colour_hsv(image, lower, upper):
  """
  Returns a filtered image with colours that fall into the provided range.
  upper and lower must be HSV values.
  """
  
  hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  
  # create numpy arrays for openCV inRAnge function
  lower_limit = np.array(lower)
  upper_limit = np.array(upper)
  
  mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
  return cv2.bitwise_and(image, image, mask=mask)
```
This is what the image looks like after the filter:
![image1]
####b) Remove undesired regions of the image.
In order to improve performance, ```region_of_interest()``` was applied before canny edge detection.

```python
imshape = image.shape
vertices = np.array(
	[
		[(0,imshape[0]),
		(imshape[1]//2.2, imshape[0]//1.65),
		((2*imshape[1])//3.7, imshape[0]//1.65), 
		(imshape[1],imshape[0])]
	], dtype=np.int32)

region_yellow_white = region_of_interest(yellow_white, vertices)
```

```python
def region_of_interest(img, vertices):
  """
  Applies an image mask.
  
  Only keeps the region of the image defined by the polygon
  formed from `vertices`. The rest of the image is set to black.
  """
  #defining a blank mask to start with
  mask = np.zeros_like(img)   
  
  #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
  if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
  else:
    ignore_mask_color = 255
      
  #filling pixels inside the polygon defined by "vertices" with the fill color    
  cv2.fillPoly(mask, vertices, ignore_mask_color)
  
  #returning the image only where mask pixels are nonzero
  masked_image = cv2.bitwise_and(img, mask)
  return masked_image
```

####c) Apply gaussian blur to improve the quality of canny function.
The filtered lines are not very sharp, so before the canny edge detection the gaussian blur was applied to provide clear edges.

```python
kernel_size = 5
blur_yellow_white = gaussian_blur(region_yellow_white, kernel_size)
```

```python
def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
![Image2]

####d) Apply the canny edge detection algorithm.

```python
low_threshold = 20
high_threshold = 60

edge_yellow_white = canny(blur_yellow_white, low_threshold, high_threshold)
```

```python
def canny(img, low_threshold, high_threshold):
  """Applies the Canny transform"""
  return cv2.Canny(img, low_threshold, high_threshold)
```
![Image3]
####e) Finally, draw the lines on the image using ```HoughLinesP()``` function

```python
rho = 1
theta = np.pi/180
threshold = 20
min_line_length = 10
max_line_gap = 10

line_yellow_white = hough_lines(edge_yellow_white, rho, theta, threshold, min_line_length, max_line_gap)

final_yellow_white = weighted_img(line_yellow_white, image)
```

```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
  """
  `img` should be the output of a Canny transform.
      
  Returns an image with hough lines drawn.
  """
  lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
  line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
  draw_lines(line_img, lines, [255, 0, 0], 7)
  return line_img
```

![Image4]

**In order to draw a single line on the left and right lanes, I modified the draw_lines() function.**

####i) Separate the left and right lane lines that the  ```HoughLinesP()``` returned
Vertical lines and lines with high slope were ignored.

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
  """
  This function draws `lines` with `color` and `thickness`.    
  Lines are drawn on the image inplace (mutates the image).
  If you want to make the lines semi-transparent, think about combining
  this function with the weighted_img() function below
  """
  
  #variable to store the left and right lane lines
  left_line = [[],[]]
  right_line = [[],[]]
  for line in lines:
    for x1,y1,x2,y2 in line:
      if abs(y2 - y1) > 0:
          
        # (x2-x1)/(y2-y1) was used instead of ((y2-y1)/(x2-x1)) 
        # to make drawing lines easier with np.poly1d()
        slope = (x2-x1)/(y2-y1)
        if (slope > 1.0 and slope < 2.0) or (slope < -1.0 and slope > -2.0):
          if (not math.isnan(slope) and not np.isinf(slope) and slope > 0):
            left_line[0] += [x1, x2]
            left_line[1] += [y1, y2]
          elif (not np.isinf(slope) and not math.isnan(slope) and slope < 0):
            right_line[0] += [x1, x2]
            right_line[1] += [y1, y2]
```

Then, ```np.polyfit``` was used to get line of best fit for each of the lanes

```python
	# find line of best fit unsing numpy function polyfit
	left_fit = np.polyfit(left_line[1], left_line[0], 1)
	right_fit = np.polyfit(right_line[1], right_line[0], 1)
```
Using these lines the algorithm works very well even in the challenge video, however, the lines would move around the lanes very violently. To fix that I found the average of last five frames.

The global variables used for line averaging

```python
LEFT_SLOPE = []
LEFT_YINT = []
RIGHT_SLOPE = []
RIGHT_YINT = []
NUM_OF_FRAMES = 5
```
The rest of ```draw_lines()```

```python
  LEFT_SLOPE.append(left_fit[0])
  RIGHT_SLOPE.append(right_fit[0])
  
  if len(LEFT_SLOPE) > NUM_OF_FRAMES: LEFT_SLOPE.pop(0)
  if len(RIGHT_SLOPE) > NUM_OF_FRAMES: RIGHT_SLOPE.pop(0)
      
  LEFT_YINT.append(left_fit[1])
  RIGHT_YINT.append(right_fit[1])
  
  if len(LEFT_YINT) > NUM_OF_FRAMES: LEFT_YINT.pop(0)
  if len(RIGHT_YINT) > NUM_OF_FRAMES: RIGHT_YINT.pop(0)
      
  #updated fits
  left_fit = [sum(LEFT_SLOPE)/len(LEFT_SLOPE), sum(LEFT_YINT)/len(LEFT_YINT)]
  right_fit = [sum(RIGHT_SLOPE)/len(RIGHT_SLOPE), sum(RIGHT_YINT)/len(RIGHT_YINT)]
      
  #find intersection of the two lines
  y_intr = (left_fit[1] - right_fit[1])/(right_fit[0] - left_fit[0])
  shape = img.shape
  adj_y_intr = y_intr + shape[0]*0.03
      
  
  left_fun = np.poly1d([left_fit[0], left_fit[1]])
  right_fun = np.poly1d([right_fit[0], right_fit[1]])
  cv2.line(img, (int(left_fun(shape[0])), shape[0]), (int(left_fun(adj_y_intr)), int(adj_y_intr)), color, thickness)
  cv2.line(img, (int(right_fun(shape[0])), shape[0]), (int(right_fun(adj_y_intr)), int(adj_y_intr)), color, thickness)
```

### 2. Identify potential shortcomings with your current pipeline


a) Since the algorithm heavily relies on detecting yellow and white, it will not perform well in low lighting condition such as night, rain, cloudy, etc.

b) Since it finds straight lines, it will not perform well on videos with sharp turns. 


### 3. Suggest possible improvements to your pipeline

One improvement would be to use a higher degree polynomial fit in ```draw_lines()``` to capture the curvature of the road lanes.

Another potential improvement could be to create an adaptive range finder for white and yellow colour that could work on low light conditions as well.
