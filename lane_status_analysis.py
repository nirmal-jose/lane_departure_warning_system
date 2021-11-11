# Lane status analysis

import os
import logging
import numpy as np
from constants import *

log_path = os.path.join(os.getcwd(), "logs")
if not (os.path.exists(log_path)):
    os.makedirs(log_path)
logging.basicConfig(
    filename=os.path.join(log_path, "lane_status_analysis.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s: %(message)s"
)


def measure_lane_curvature(ploty, leftx, rightx, visualization=False):

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ym_per_pix = 30/(frame_height/input_scale)
    xm_per_pix = LANEWIDTH/(700/input_scale)  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix +
                            right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    if leftx[0] - leftx[-1] > 50/input_scale:
        curve_direction = 'Left curve'
    elif leftx[-1] - leftx[0] > 50/input_scale:
        curve_direction = 'Right curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad+right_curverad)/2.0, curve_direction


def off_center(left, mid, right):
    """

    :param left: left lane position
    :param mid:  car position
    :param right: right lane position
    :return: True or False, indicator of off center driving
    """
    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  # driving right off
        offset = a / width * LANEWIDTH - LANEWIDTH / 2.0
    else:       # driving left off
        offset = LANEWIDTH / 2.0 - b / width * LANEWIDTH

    return offset


def compute_car_offcenter(ploty, left_fitx, right_fitx, undist):

    # Create an image to draw the lines on
    height = undist.shape[0]
    width = undist.shape[1]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    bottom_l = left_fitx[height-1]
    bottom_r = right_fitx[0]

    offcenter = off_center(bottom_l, width/2.0, bottom_r)

    return offcenter, pts


def lane_status_analysis(frame, ploty, left_lane, right_lane):
    # measure the lane curvature
    curvature, curve_direction = measure_lane_curvature(
        ploty, left_lane.mean_fitx, right_lane.mean_fitx)
    logging.info("Curvature: {:.2f} m Curve Direction: {}".format(
        curvature, curve_direction))
    # compute the car's off-center in meters
    offcenter, pts = compute_car_offcenter(
        ploty, left_lane.mean_fitx, right_lane.mean_fitx, frame)
    logging.info("Off-Center: {:.4f} m".format(offcenter))

    return curvature, curve_direction, offcenter, pts
