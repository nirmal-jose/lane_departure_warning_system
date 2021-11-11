# Lane Departure Warning System - ADAS

from timeit import default_timer as timer
from lane_augmentation import lane_augmentation
from lane_status_analysis import lane_status_analysis
from lane_detection_tracking import lane_detection_tracking
import cv2
import sys
import os
import logging
log_path = os.path.join(os.getcwd(), "logs")
if not (os.path.exists(log_path)):
    os.makedirs(log_path)
logging.basicConfig(
    filename=os.path.join(log_path, "lane_departure_warning.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s: %(message)s"
)

op_path = os.getcwd()+'\output'
if not os.path.exists(op_path):
    os.makedirs(op_path)


def lane_departure_warning_frame(ip_file, op_file):
    """
    Load image file from ip_file
    """

    logging.info("Reading Image File")
    img = cv2.imread(ip_file)
    height, width, channels = img.shape
    logging.info("Image Width:{} Height:{}".format(
        width, height))

    start = timer()
    ploty, left_lane, right_lane, binary_sub = lane_detection_tracking(
        img, width=width, height=height, visualization=True)
    curvature, curve_direction, offcenter, pts = lane_status_analysis(
        img, ploty, left_lane, right_lane)
    end = timer()
    fps = 1.0 / (end - start)
    newwrap = lane_augmentation(
        offcenter, pts, img, fps, curvature, curve_direction, binary_sub)
    cv2.imshow("Lane Departure Warning System", newwrap)
    cv2.waitKey(0)
    # save image
    status = cv2.imwrite(op_file+".png", newwrap)
    logging.info(
        "Lane Departure Warning System Image written to file-system : {}".format(status))


def lane_departure_warning_video(ip_file, op_file):
    """
    Load video file from ip_file and write output to op_file
    """

    logging.info("Reading Video File")
    video = cv2.VideoCapture(ip_file)

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    aspect_ratio = width/height
    logging.info("Video Width:{} Height:{} FPS:{} Aspect Ratio:{:.2f}".format(
        width, height, fps, aspect_ratio))
    frame_counter = 0

    window = "Lane Departure Warning System"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    if not video.isOpened():
        logging.info("Unable to open video file")

    else:
        logging.info("Reading frames..")

        video_fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_limit = video_fps
        # Video Encoder, give width and height of frame not that of video file.
        op_video = cv2.VideoWriter(op_file+".avi",
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   video_fps, (int(width), int(height)))
        logging.info("Encoding frames to, {}.avi".format(op_file))

        while(True):
            # input(type(video))
            ok, frame = video.read()
            if ok:
                frame_counter += 1
                # feed to detector and return processed frame
                start = timer()
                ploty, left_lane, right_lane, binary_sub = lane_detection_tracking(
                    frame, width=width, height=height)
                curvature, curve_direction, offcenter, pts = lane_status_analysis(
                    frame, ploty, left_lane, right_lane)
                end = timer()
                fps = 1.0 / (end - start)
                newwrap = lane_augmentation(
                    offcenter, pts, frame, fps, curvature, curve_direction, binary_sub)
                op_video.write(newwrap)
                cv2.imshow(window, newwrap)
                # cv2.waitKey(25), for auto-play, press q to quit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        video.release()
        op_video.release()
        cv2.destroyAllWindows()

    video.release()


if __name__ == "__main__":
    logging.info("Lane Departure Warning System")

    demo = 1  # 1: image, 2 video

    if demo == 1:
        ip_file = os.path.join(os.getcwd(), "input", "straight_lines2.jpg")
        op_file = os.path.join(op_path, os.path.splitext(
            os.path.basename(ip_file))[0])
        lane_departure_warning_frame(ip_file, op_file)
    else:
        ip_file = os.path.join(os.getcwd(), "input", "project_video.mp4")
        if not os.path.exists(ip_file):
            logging.info("Input File doesn't exists..Aborting!!!")
            sys.exit()

        op_file = os.path.join(op_path, ip_file.split('\\')[-1].split('.')[0])

        lane_departure_warning_video(ip_file, op_file)
