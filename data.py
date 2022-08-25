from tkinter import N
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import os


def show_video(path_to_video):
    # --- shows video
    video = cv2.VideoCapture(path_to_video)
    while(video.isOpened()):
        ret, frame = video.read()
        # print(frame)
        # break 
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
        labeled_video = cv2.VideoCapture(f'{path_to_labeled_videos}/{file}')
        while labeled_video.isOpened():
            ret, labeled_frame = labeled_video.read()
            if ret != False:
                labeled_list.append(labeled_frame)
            else:
                break
        labeled_video.release()

    unlabeled_list = []
    for file in sorted(os.listdir(path_to_unlabeled_videos)):
        
        unlabeled_video = cv2.VideoCapture(f'{path_to_unlabeled_videos}/{file}')
        while unlabeled_video.isOpened(): 
            ret, unlabeled_frame = unlabeled_video.read()
            if ret != False:
                unlabeled_list.append(unlabeled_frame)
            else:
                break
        unlabeled_video.release()
    labeled_array = np.array(labeled_list)
    print('Shape of labeled arrays: ', labeled_array.shape)
    unlabeled_array = np.array(unlabeled_list)
    print('Shape of unlabeled arrays: ', unlabeled_array.shape)
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
    # print('Shape of pitch_yaw arrays: ', pitch_yaw.shape)
    # print('Shape of pitch arrays: ', pitch.shape)
    # print('Shape of yaw arrays: ', yaw.shape)

    return pitch_yaw, pitch, yaw


class torch_dataset(Dataset):
    def __init__(self, labeled_array, unlabeled_array, pitch_yaw, pitch, yaw):
        self.labeled_array = labeled_array
        self.unlabeled_array = unlabeled_array
        self.pitch_yaw = pitch_yaw
        self.pitch = pitch
        self.yaw = yaw

        @property
        def n_insts(self):
            return len(self.pitch_yaw)

        @property
        def labeled_len(self):
            return self.labeled_array.shape
        
        def unlabeled_len(self):
            return self.labeled_array.shape

        def __len__(self):
            return self.n_insts
        
        # def __getitem__(self, item):
            # labeled = 

        

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
    pitch_yaw, pitch, yaw = load_txt('labels')
    
    # print(pitch_yaw[0].shape)
    # print(labeled_array.shape)



