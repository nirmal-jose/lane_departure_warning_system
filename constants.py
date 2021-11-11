# Constants

import numpy as np
import cv2

LANEWIDTH = 3.7  # highway lane width in US: 3.7 meters
input_scale = 1
output_frame_scale = 1
N = 4  # buffer previous N lines

# fullsize:1280x720
x = [194, 1117, 705, 575]
y = [719, 719, 461, 461]
X = [290, 990, 990, 290]
Y = [719, 719, 0, 0]

# Threshold for color and gradient thresholding
s_thresh, sx_thresh, dir_thresh, m_thresh, r_thresh = (
    120, 255), (20, 100), (0.7, 1.3), (30, 100), (200, 255)

src = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]], [
               x[2], y[2]], [x[3], y[3]]]) / input_scale)
dst = np.floor(np.float32([[X[0], Y[0]], [X[1], Y[1]], [
               X[2], Y[2]], [X[3], Y[3]]]) / input_scale)

frame_width = 1280
frame_height = 720

# Only for creating the final video visualization
X_b = [574, 706, 706, 574]
Y_b = [719, 719, 0, 0]
src_ = np.floor(np.float32([[x[0], y[0]], [x[1], y[1]], [
                x[2], y[2]], [x[3], y[3]]]) / (input_scale*2))
dst_ = np.floor(np.float32([[X_b[0], Y_b[0]], [X_b[1], Y_b[1]], [
                X_b[2], Y_b[2]], [X_b[3], Y_b[3]]]) / (input_scale*2))


def warper(img, M):

    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped
