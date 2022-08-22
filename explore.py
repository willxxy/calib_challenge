import numpy as np
import pandas as pd
import cv2
import torch


def show_labeled_video(video):
    # --- shows labeled video

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


def show_unlabeled_video(video):
    # --- shows unlabeled video

    while(video.isOpened()):
        ret, frame = video.read()
        print(frame)
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
    # labeled_video = cv2.VideoCapture('labeled/0.hevc')

    # --- example unlabeled video
    # unlabeled_video = cv2.VideoCapture('unlabeled/5.hevc')

    # --- show labeled video
    # show_labeled_video(labeled_video)

    # --- show unlabeled video
    # show_unlabeled_video(unlabeled_video)
