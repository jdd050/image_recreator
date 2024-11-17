# image_recreator
A Python app that takes an image and uses opencv to draw the image in Python's Turtle graphics library.

# Notes 
- On line 65, col 60-67
  - These parameters may be adjusted to modify edge detection quality accordingly.
- On line 107-108
  - For performance reasons, the coordinates of each contour are reduced by approximation using cv2's approxPolyDP
