from tkinter import N
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
    # print('Shape of labeled arrays: ', labeled_array.shape)
    unlabeled_array = np.array(unlabeled_list)
    # print('Shape of unlabeled arrays: ', unlabeled_array.shape)
    return labeled_array, unlabeled_array


def load_txt(path_to_txt):
    pitch_yaw = []
    pitch = []
    yaw = []
    for file in sorted(os.listdir(path_to_txt)):
        if '.txt' in file:
            txt = np.loadtxt(f'{path_to_txt}/{file}')

            txt = np.nan_to_num(txt)
            txt[:,0] = np.nan_to_num(txt[:,0])
            txt[:,1] = np.nan_to_num(txt[:, 1])

            pitch_yaw.append(txt)
            pitch.append(txt[:, 0])
            yaw.append(txt[:, 1])

    pitch_yaw = np.array(pitch_yaw, dtype = 'object')
    pitch = np.array(pitch, dtype = 'object')
    yaw = np.array(yaw, dtype = 'object')
    print('Shape of pitch_yaw arrays: ', pitch_yaw.shape)
    print('Shape of pitch arrays: ', pitch.shape)
    print('Shape of yaw arrays: ', yaw.shape)

    return pitch_yaw, pitch, yaw


if __name__ == '__main__':
    # --- example labeled video
    # labeled_video = 'labeled/0.hevc'
    # show_video(labeled_video)

    # --- example unlabeled video
    # unlabeled_video = 'unlabeled/5.hevc'
    # show_video(unlabeled_video)

    # --- stacked labeled and unlabeled arrays of all .hevc files 
    labeled_array, unlabeled_array = videos_array('labeled', 'unlabeled')

    # --- load in pitch and yaw angles (radians) from txt files 
    pitch_yaw, pitch, yaw = load_txt('labeled')
    



