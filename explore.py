import numpy as np
import pandas as pd
import cv2


def show_video(video):
    while(video.isOpened()):
        ret, frame = video.read()
        if ret != False:
            cv2.imshow('frame', frame)
        else:
            break
        if (cv2.waitKey(1) == ord('q')):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    video = cv2.VideoCapture('labeled/0.hevc')

    # --- show video
    # show_video(video)