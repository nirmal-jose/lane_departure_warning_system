# lane augmentation

import os
import logging
import cv2
from constants import *

log_path = os.path.join(os.getcwd(), "logs")
if not (os.path.exists(log_path)):
    os.makedirs(log_path)
logging.basicConfig(
    filename=os.path.join(log_path, "lane_augmentation.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s: %(message)s"
)

op_path = os.getcwd()+'\output'
if not os.path.exists(op_path):
    os.makedirs(op_path)


def create_output_frame(offcenter, pts, undist_ori, fps, curvature, curve_direction, binary_sub, threshold=0.6):
    """

    :param offcenter:
    :param pts:
    :param undist_ori:
    :param fps:
    :param threshold:
    :return:
    """

    undist_ori = cv2.resize(undist_ori, (0, 0), fx=1 /
                            output_frame_scale, fy=1/output_frame_scale)
    w = undist_ori.shape[1]
    h = undist_ori.shape[0]

    M_b = cv2.getPerspectiveTransform(src_, dst_)
    undist_birdview = warper(cv2.resize(
        undist_ori, (0, 0), fx=1/2, fy=1/2), M_b)

    color_warp = np.zeros_like(undist_ori).astype(np.uint8)

    # create a frame to hold every image
    whole_frame = np.zeros((int(h*2.5), int(w*2.34), 3), dtype=np.uint8)

    if abs(offcenter) > threshold:  # car is offcenter more than 0.6 m
        # Draw Red lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))  # red
    else:  # Draw Green lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))  # green

    M_inv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (int(
        frame_width/input_scale), int(frame_height/input_scale)))

    # Combine the result with the original image    # result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    newwarp_ = cv2.resize(newwarp, None, fx=input_scale/output_frame_scale,
                          fy=input_scale/output_frame_scale, interpolation=cv2.INTER_LINEAR)

    output = cv2.addWeighted(undist_ori, 1, newwarp_, 0.3, 0)

    ############## generate the combined output frame only for visualization purpose ################
    whole_frame[40:40+h, 20:20+w, :] = undist_ori
    whole_frame[40:40+h, 60+w:60+2*w, :] = output
    whole_frame[220+h//2:220+2*h//2, 20:20+w//2, :] = undist_birdview
    whole_frame[220+h//2:220+2*h//2, 40+w//2:40+w,
                0] = cv2.resize((binary_sub*255).astype(np.uint8), (0, 0), fx=1/2, fy=1/2)
    whole_frame[220+h//2:220+2*h//2, 40+w//2:40+w,
                1] = cv2.resize((binary_sub*255).astype(np.uint8), (0, 0), fx=1/2, fy=1/2)
    whole_frame[220+h//2:220+2*h//2, 40+w//2:40+w,
                2] = cv2.resize((binary_sub*255).astype(np.uint8), (0, 0), fx=1/2, fy=1/2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if offcenter >= 0:
        offset = offcenter
        direction = 'Right'
    elif offcenter < 0:
        offset = -offcenter
        direction = 'Left'

    info_road = "Road Status"
    info_lane = "Lane info: {0}".format(curve_direction)
    info_cur = "Curvature {:6.1f} m".format(curvature)
    info_offset = "Off center: {0} {1:3.1f}m".format(direction, offset)
    info_framerate = "{0:4.1f} fps".format(fps)
    info_warning = "Warning: offcenter > 0.6m (use higher threshold in real life)"

    info = "{}, {}, {}".format(info_lane, info_cur, info_offset)

    cv2.putText(whole_frame, "Departure Warning System with a Monocular Camera",
                (23, 25), font, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Origin", (22, 70), font,
                0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Augmented", (40+w+25, 70),
                font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Bird's View", (22+30, 70+35+h),
                font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Lanes", (22+225, 70+35+h),
                font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, info_road, (40+w+50, 70+35+h),
                font, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, info_warning, (35+w, 60+h),
                font, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, info_lane, (40+w+50, 70+35+40+h),
                font, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, info_cur, (40+w+50, 70+35+80+h),
                font, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, info_offset, (40+w+50, 70+35+120+h),
                font, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, info_framerate, (40+w+250, 70),
                font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(output, info, (50, 50), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    #cv2.putText(output, info_warning, (20,20), font, 0.4, (255,255,0), 1,cv2.LINE_AA)
    #cv2.putText(output, info_lane, (30,30), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    #cv2.putText(output, info_cur, (40,40), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    #cv2.putText(output, info_offset, (50,50), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    return whole_frame, output


def lane_augmentation(offcenter, pts, img, fps, curvature, curve_direction, binary_sub):
    # combine all images into final video output (only for visualization purpose)
    output, wrap = create_output_frame(
        offcenter, pts, img, fps, curvature, curve_direction, binary_sub)

    return wrap
