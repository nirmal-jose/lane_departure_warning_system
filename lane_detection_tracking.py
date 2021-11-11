# Lane Detection and Tracking

import cv2
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import os
import logging
from constants import *

log_path = os.path.join(os.getcwd(), "logs")
if not (os.path.exists(log_path)):
    os.makedirs(log_path)
logging.basicConfig(
    filename=os.path.join(log_path, "lane_detection_tracking.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s: %(message)s"
)

op_path = os.getcwd()+'\output'
if not os.path.exists(op_path):
    os.makedirs(op_path)


class Lane():
    def __init__(self):
        # was the line detected in the last frame or not
        self.detected = False
        # x values for detected line pixels
        self.cur_fitx = None
        # y values for detected line pixels
        self.cur_fity = None
        # x values of the last N fits of the line
        self.prev_fitx = []
        # polynomial coefficients for the most recent fit
        self.current_poly = [np.array([False])]
        # best polynomial coefficients for the last iteration
        self.prev_poly = [np.array([False])]

    def average_pre_lanes(self):
        tmp = copy(self.prev_fitx)
        tmp.append(self.cur_fitx)
        self.mean_fitx = np.mean(tmp, axis=0)

    def append_fitx(self):
        if len(self.prev_fitx) == N:
            self.prev_fitx.pop(0)
        self.prev_fitx.append(self.mean_fitx)

    def process(self, ploty):
        self.cur_fity = ploty
        self.average_pre_lanes()
        self.append_fitx()
        self.prev_poly = self.current_poly


def threshold_col_channel(channel, thresh):

    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ threshold according to the direction of the gradient

    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def find_edges(img, s_thresh=s_thresh, sx_thresh=sx_thresh, dir_thresh=dir_thresh, visualization=False):

    img = np.copy(img)
    # Convert to HSV color space and threshold the s channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    s_binary = threshold_col_channel(s_channel, thresh=s_thresh)
    if(visualization):
        cv2.imshow("HSV color space", s_binary)
        cv2.waitKey(0)
        # save image
        img_file = os.path.join(op_path, "hsv_image.png")
        status = cv2.imwrite(img_file, 255*s_binary)
        logging.info(
            "HSV color space Image written to file-system : {}".format(status))

    # Sobel x
    sxbinary = abs_sobel_thresh(
        img, orient='x', sobel_kernel=3, thresh=sx_thresh)
    # mag_binary = mag_thresh(img, sobel_kernel=3, thresh=m_thresh)
    # # gradient direction
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)
    #
    # # output mask
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(((sxbinary == 1) & (dir_binary == 1)) |
                     ((s_binary == 1) & (dir_binary == 1)))] = 1

    # add more weights for the s channel
    c_bi = np.zeros_like(s_channel)
    c_bi[((sxbinary == 1) & (s_binary == 1))] = 2

    ave_binary = (combined_binary + c_bi)

    return ave_binary


def window_search(left_fit, right_fit, binary_warped, margin=100, visualization=False):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's easier to find line pixels with windows search
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if visualization:
        # Generate x and y values for plotting
        ploty = np.linspace(
            0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # And you're done! But let's visualize the result here as well
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = out_img.astype('uint8')
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds],
                nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds],
                nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width / input_scale))
        plt.ylim((frame_height / input_scale, 0))
        img_file = os.path.join(op_path, 'window_search.png')
        plt.savefig(img_file)
        plt.show()

    return left_fit, right_fit


def tracker(binary_sub, ploty, visualization=False):

    left_fit, right_fit = window_search(
        left_lane.prev_poly, right_lane.prev_poly, binary_sub, margin=100/input_scale, visualization=visualization)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 / input_scale):
        left_lane.detected = True
        right_lane.detected = True
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
        # global tt
        # tt = tt + 1
    else:
        left_lane.detected = False
        right_lane.detected = False
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        left_lane.cur_fitx = left_lane.prev_fitx[-1]
        right_lane.cur_fitx = right_lane.prev_fitx[-1]

# fit the lane line


def full_search(binary_warped, visualization=False):

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.astype('uint8')

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = np.floor(100/input_scale)
    # Set minimum number of pixels found to recenter window
    minpix = np.floor(50/input_scale)
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if visualization:
            cv2.rectangle(out_img, (int(win_xleft_low), int(win_y_low)), (int(
                win_xleft_high), int(win_y_high)), (0, 255, 0), 2)
            cv2.rectangle(out_img, (int(win_xright_low), int(win_y_low)), (int(
                win_xright_high), int(win_y_high)), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Visualization

    # Generate x and y values for plotting
    if visualization:
        ploty = np.linspace(
            0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds],
                nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds],
                nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.subplot(1,2,1)
        plt.imshow(out_img)
        # plt.imshow(binary_warped)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width / input_scale))
        plt.ylim((frame_height / input_scale, 0))
        img_file = os.path.join(op_path, 'window_search.png')
        plt.title("Window Search to extract left and right line pixel positions")
        plt.xlabel("Image Widht (pixels)")
        plt.ylabel("Image Height (pixels)")
        plt.savefig(img_file)
        plt.show()

    return left_fit, right_fit


def detector(binary_sub, ploty, visualization=False):

    left_fit, right_fit = full_search(binary_sub, visualization=visualization)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 / input_scale):
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
        left_lane.detected = True
        right_lane.detected = True
    else:
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        if len(left_lane.prev_fitx) > 0:
            left_lane.cur_fitx = left_lane.prev_fitx[-1]
            right_lane.cur_fitx = right_lane.prev_fitx[-1]
        else:
            left_lane.cur_fitx = left_fitx
            right_lane.cur_fitx = right_fitx
        left_lane.detected = False
        right_lane.detected = False


left_lane = Lane()
right_lane = Lane()


def lane_detection_tracking(frame, width=frame_width, height=frame_height, visualization=False):
    """
    Process frame
    frame: video frame to process

    1. Find Binary Image
    """

    global frame_width
    global frame_height

    frame_width = width
    frame_height = height

    # find the binary image of lane/edges
    img_binary = find_edges(frame, visualization=visualization)
    if(visualization):
        cv2.imshow("Binary Image", img_binary)
        cv2.waitKey(0)
        # save image
        img_file = os.path.join(op_path, "binary_image.png")
        status = cv2.imwrite(img_file, 255*img_binary)
        logging.info("Binary Image written to file-system : {}".format(status))

    # warp the image to bird view
    M = cv2.getPerspectiveTransform(src, dst)
    binary_warped = warper(img_binary, M)  # get binary image contains edges
    if(visualization):
        cv2.imshow("Binary Image - Bird View", binary_warped)
        cv2.waitKey(0)
        # save image
        img_file = os.path.join(op_path, "binary_image_bird_view.png")
        status = cv2.imwrite(img_file, 255*binary_warped)
        logging.info(
            "Binary Image - Bird View written to file-system : {}".format(status))

    # crop the binary image
    binary_sub = np.zeros_like(binary_warped)
    binary_sub[:, int(150/input_scale):int(-80/input_scale)
               ] = binary_warped[:, int(150/input_scale):int(-80/input_scale)]

    # start detector or tracker to find the lanes
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    if left_lane.detected:  # start tracker
        tracker(binary_sub, ploty, visualization)
    else:  # start detector
        detector(binary_sub, ploty, visualization)

    # average among the previous N frames to get the averaged lanes
    left_lane.process(ploty)
    right_lane.process(ploty)

    return ploty, left_lane, right_lane, binary_sub
