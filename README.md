# Lane Departure Warning System

## Cranfield University - Connected and Autonomous Vehicle Engineering - Sensor, Perception and Visualisation Assignment

## Steps Involved
1. Lane detection / tracking
    * Use color transforms, gradients, etc., to create a thresholded binary image.
    * Apply a perspective transform to rectify binary image ("birds-eye view").
    * Detect / track lane pixels and fit to find the lane boundary.

2. Lane status anlysis
    * Determine the curvature of the lane
    * Compute the vehicle position with respect to center.

3. Lane augmentation
    * Warp the detected lane boundaries back onto the original image.
    * Print the road status into image


### Reference
1. [JunshengFu/driving-lane-departure-warning](https://github.com/JunshengFu/driving-lane-departure-warning)