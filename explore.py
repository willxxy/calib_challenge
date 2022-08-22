import numpy as np
import pandas as pd
import cv2
import torch


def show_video(path_to_video):
    # --- shows video
    video = cv2.VideoCapture(path_to_video)
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


def videos_array(path_to_labeled_vides, path_to_unlabeled_videos):
    # --- returns 4d array of all videos from labeled and unlabeled
    






if __name__ == '__main__':
    # --- example labeled video
    # labeled_video = 'labeled/0.hevc'
    # show_video(labeled_video)

    # --- example unlabeled video
    # unlabeled_video = 'unlabeled/5.hevc'
    # show_video(unlabeled_video)
