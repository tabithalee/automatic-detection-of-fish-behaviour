import csv
import os

import scratch
'''
class VideoInfo:
    def __init__(self, name, startle_frame, roi):
        self.name = name
        self.startle_frame = startle_frame
        self.roi = roi
'''
import numpy as np
from npz_processing import find_abs_array_max, find_peak_duration

reader = {}
relativePath = '/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/'
sheetPath = relativePath + 'sheets/annotated_data_3x3_2.csv'

csv.register_dialect('myDialect', delimiter='|', skipinitialspace=True)

dirName = '3_data_process'

if not os.path.exists(''.join((relativePath, dirName))):
    os.mkdir(''.join((relativePath, dirName)))

allWindows = False

skew_maxes_index = np.zeros(27)
kurt_maxes_index = np.zeros(27)
skew_d_maxes_index = np.zeros(27)
kurt_d_maxes_index = np.zeros(27)
max_maxes_index = np.zeros(27)

skew_dur = np.zeros(27)
kurt_dur = np.zeros(27)
skew_d_dur = np.zeros(27)
kurt_d_dur = np.zeros(27)
max_dur = np.zeros(27)

my_index = 0
sample_interval = 5

with open(sheetPath, 'r') as csvfile:
    reader = csv.DictReader(csvfile, dialect='myDialect')
    for row in reader:
        print(row)
        myWindows = range(1, 10) if allWindows is True else [int(item) for item in row['Tracked Regions'].split(',')]

        npzfile = np.load(''.join(('/home/tabitha/Desktop/5_data_process/5_data_process', row['Video Title'], '.npz')))

        skew_maxes_index[my_index] = find_abs_array_max(npzfile['skew_list'])
        kurt_maxes_index[my_index] = find_abs_array_max(npzfile['kurtosis_list'])
        skew_d_maxes_index[my_index] = find_abs_array_max(npzfile['d1'])
        kurt_d_maxes_index[my_index] = find_abs_array_max(npzfile['d2'])
        max_maxes_index[my_index] = find_abs_array_max(npzfile['max_list'])

        skew_dur[my_index] = find_peak_duration(npzfile['skew_list'], sample_interval)
        kurt_dur[my_index] = find_peak_duration(npzfile['kurtosis_list'], sample_interval)
        skew_d_dur[my_index] = find_peak_duration(npzfile['d1'], sample_interval)
        kurt_d_dur[my_index] = find_peak_duration(npzfile['d2'], sample_interval)
        max_dur[my_index] = find_peak_duration(npzfile['max_list'], sample_interval)

        my_index += 1

        '''
        scratch.main(row['Video Title'], row['Startle Frame'], row['Extra Startle'],
                     myWindows, True, dirName)
        '''
    np.savez(''.join(('/home/tabitha/Desktop/', 'five_data')), sm=skew_maxes_index,
             km=kurt_maxes_index, sdm=skew_d_maxes_index, kdm=kurt_d_maxes_index, mm=max_maxes_index,
             sd=skew_dur, kd=kurt_dur, sdd=skew_d_dur, kdd=kurt_d_dur, md=max_dur)

csvfile.close()
