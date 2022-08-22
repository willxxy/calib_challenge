import numpy as np
import pandas as pd
import cv2
import torch
import os


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


def videos_array(path_to_labeled_videos, path_to_unlabeled_videos):
    # --- returns 4d array of all videos from labeled and unlabeled
    labeled_list = []
    for file in sorted(os.listdir(path_to_labeled_videos)):
        if '.hevc' in file:
            labeled_video = cv2.VideoCapture(f'{path_to_labeled_videos}/{file}')
            _, labeled_frame = labeled_video.read()
            labeled_list.append(labeled_frame)

    unlabeled_list = []
    for file in sorted(os.listdir(path_to_unlabeled_videos)):
        if '.hevc' in file:
            unlabeled_video = cv2.VideoCapture(f'{path_to_unlabeled_videos}/{file}')
            _, unlabeled_frame = unlabeled_video.read()
            unlabeled_list.append(unlabeled_frame)

    labeled_array = np.array(labeled_list)
    print('Shape of labeled arrays: ', labeled_array.shape)
    unlabeled_array = np.array(unlabeled_list)
    print('Shape of unlabeled arrays: ', unlabeled_array.shape)
    

    return labeled_array, unlabeled_array






if __name__ == '__main__':
    # --- example labeled video
    # labeled_video = 'labeled/0.hevc'
    # show_video(labeled_video)

    # --- example unlabeled video
    # unlabeled_video = 'unlabeled/5.hevc'
    # show_video(unlabeled_video)

    # --- stacked labeled and unlabeled arrays of all .hevc files 
    labeled_array, unlabeled_array = videos_array('labeled', 'unlabeled')


