# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:29:25 2022

@author: asligar
"""

import os
import numpy as np
import pandas as pd
import scipy.io as scipyio
import matplotlib.pyplot as plt
import utm
from tqdm import tqdm
from datetime import datetime
import scipy.interpolate
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
class DeepSense_6G:
    def __init__(self,csv_file = 'scenario1.csv',scenario_folder = './'):
        self.csv_file = csv_file
        self.scenario_folder = scenario_folder

    def load(self,seq_index=1):
        print(f'Loading DeepSense Data, sequence number {seq_index}')
        full_path = os.path.join(self.scenario_folder, self.csv_file)
        dataframe = pd.read_csv(full_path)

        
        N_BEAMS = 64

        all_seq_index = dataframe['seq_index'].values
        values = np.where(all_seq_index==seq_index)[0]
        n_start = values[0]
        n_stop = values[-1]+1
        n_samples = n_stop-n_start
        pwr_rel_paths = dataframe['unit1_pwr_60ghz'].values



        all_times = dataframe['time_stamp[UTC]']
        # all_times = all_times[n_start:n_stop]
        all_times = all_times.to_numpy()

        time_stamps = np.zeros((n_samples))
        #I don't know how to get time stamps in seconds from pandas dataframe, so doing this another way
        for idx, n in enumerate(range(n_start,n_stop)):
            time = all_times[n]
            time = time.replace('\'','')
            dt_obj = datetime.strptime(time,'[%H-%M-%S-%f]').strftime('%S.%f')
            dt_obj_m = datetime.strptime(time,'[%H-%M-%S-%f]').strftime('%M')
            dt_obj_h = datetime.strptime(time,'[%H-%M-%S-%f]').strftime('%H')
            d_in_s = float(dt_obj_h)*360+float(dt_obj_m)*60+float(dt_obj)
            time_stamps[idx] = d_in_s

        all_img_rel_paths = dataframe['unit1_rgb'].values
        img_rel_paths_seq = all_img_rel_paths[n_start:n_stop]
        full_path_imgs = []
        for each in img_rel_paths_seq:
            full_path_imgs.append(os.path.join(self.scenario_folder, each))

        #zero shift
        time_stamps = time_stamps-time_stamps[0]
        time_stamps = time_stamps
        num_time_stamps = len(time_stamps)

        time_stop = time_stamps[-1]
        time_stamps_resampled = np.linspace(0,time_stop,num=num_time_stamps)        
        
        pwrs_array = np.zeros((n_samples, N_BEAMS))
        
        for n, sample_idx in enumerate(tqdm(range(n_start,n_stop))):
            pwr_abs_path = os.path.join(self.scenario_folder, pwr_rel_paths[sample_idx])
            pwrs_array[n] = np.loadtxt(pwr_abs_path)
    


        # BS position (taking the first is enough if we know it is static)
        bs_pos_latlon = np.loadtxt(os.path.join(self.scenario_folder, dataframe['unit1_loc'].values[0]))
        bs_pos_utm = utm.from_latlon(bs_pos_latlon[0],bs_pos_latlon[1])
        bs_pos_abs = [0,0]
        # UE positions
        pos_rel_paths = dataframe['unit2_loc'].values
        pos_array_latlon = np.zeros((n_samples, 2)) # 2 = Latitude and Longitude
        pos_array_utm = np.zeros((n_samples, 2)) # 2 = Latitude and Longitude
        pos_array_abs = np.zeros((n_samples, 3)) # 2 = Latitude and Longitude
        
        # Load each individual txt file
        for n, sample_idx in enumerate(range(n_start,n_stop)):
            pos_abs_path = os.path.join(self.scenario_folder, pos_rel_paths[sample_idx])
            pos_array_latlon[n] = np.loadtxt(pos_abs_path)
            utm_temp = utm.from_latlon(pos_array_latlon[n,0],pos_array_latlon[n,1])
            pos_array_utm[n] = [utm_temp[0],utm_temp[1]]
            pos_array_abs[n] = [pos_array_utm[n,0]-bs_pos_utm[0],pos_array_utm[n,1]-bs_pos_utm[1],0]
            
        pos_array_interpFunc = scipy.interpolate.interp1d(time_stamps,pos_array_abs,axis=0,assume_sorted=True)
        self.pos_array_interpFunc = pos_array_interpFunc
        pos_array_abs_resampled = pos_array_interpFunc(time_stamps_resampled)
        
        diff_pos_array=np.diff(pos_array_abs_resampled,axis=0,append=0)
        diff_pos_array[-1] = diff_pos_array[2]
        yaw_array=np.arctan2(diff_pos_array[:,1],diff_pos_array[:,0])
        yaw_array = savgol_filter(yaw_array,5, 2)
        rot_obj = Rotation.from_euler('Z',yaw_array)
        rot_array = rot_obj.as_matrix()

        rot_array_interpFunc = scipy.interpolate.interp1d(time_stamps,rot_array,axis=0,assume_sorted=True)
        self.rot_array_interpFunc= rot_array_interpFunc

        self.pos_array_original = pos_array_abs
        self.rot_array_original = rot_array
        self.yaw_array = yaw_array
        self.power_vs_index = pwrs_array
        self.rot_array = rot_array
        self.pos_array = pos_array_abs_resampled
        self.time_stamps = time_stamps_resampled
        self.time_stamps_original = time_stamps
        self.num_time_stamps = len(time_stamps)
        self.frame_image = full_path_imgs

    def get_position_original(self,idx):
        return self.pos_array_original[idx]
    def get_rotation_original(self,idx):
        return self.rot_array_original[idx]

    def get_position(self,time,x_offset=0,y_offset=0):
        pos = self.pos_array_interpFunc(time)
        pos[0] += x_offset
        pos[1] += y_offset
        return pos
    def get_rotation(self,time):
        return self.rot_array_interpFunc(time)