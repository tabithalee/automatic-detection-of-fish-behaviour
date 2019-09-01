import csv
import os

import main

import numpy as np
from npz_processing import find_abs_array_max, find_peak_duration

from configparser import ConfigParser

parser = ConfigParser()
parser.read('dev.ini')

reader = {}

# user settings
repoPath = parser.get('user_settings', 'repoPath')

# main settings
dirName = parser.get('main_settings', 'dirName')
allWindows = parser.get('main_settings', 'allWindows')
sampleInterval = parser.getint('main_settings', 'sampleInterval')
my_csvFile = parser.get('main_settings', 'csvFile')
numGoodVids = parser.getint('main_settings', 'numGoodVids')
numDivX = parser.getint('main_settings', 'numDivX')
numDivY = parser.getint('main_settings', 'numDivY')
numHistBins = parser.getint('main_settings', 'numHistBins')
saveHistograms = parser.getboolean('main_settings', 'saveHistograms')
saveData = parser.getboolean('main_settings', 'saveData')
fps = parser.getint('main_settings', 'fps')

relativePath = ''.join((repoPath, '/savedHistograms/'))
sheetPath = ''.join((relativePath, 'sheets/', my_csvFile))

csv.register_dialect('myDialect', delimiter='|', skipinitialspace=True)

if not os.path.exists(''.join((relativePath, dirName))):
    os.mkdir(''.join((relativePath, dirName)))

if not os.path.exists(''.join((relativePath, '/npzfiles/', dirName))):
    os.mkdir(''.join((relativePath, '/npzfiles/', dirName)))

# initialization of arrays to store statistical information
my_index = 0

skew_maxes_index = np.zeros(numGoodVids)
kurt_maxes_index = np.zeros(numGoodVids)
skew_d_maxes_index = np.zeros(numGoodVids)
kurt_d_maxes_index = np.zeros(numGoodVids)
max_maxes_index = np.zeros(numGoodVids)

skew_dur = np.zeros(numGoodVids)
kurt_dur = np.zeros(numGoodVids)
skew_d_dur = np.zeros(numGoodVids)
kurt_d_dur = np.zeros(numGoodVids)
max_dur = np.zeros(numGoodVids)


with open(sheetPath, 'r') as csvfile:
    reader = csv.DictReader(csvfile, dialect='myDialect')
    for row in reader:
        print(row)
        myWindows = range(1, (numDivX * numDivY + 1)) if allWindows is True \
            else [int(item) for item in row['Tracked Regions'].split(',')]

        # run the algorithm
        main.main(row['Video Title'], row['Startle Frame'], row['Extra Startle'],
                  myWindows, True, dirName, numDivX, numDivY, sampleInterval, saveHistograms, saveData, repoPath, fps)

        # process the resulting npzfiles
        npzfile = np.load(''.join((relativePath, 'npzfiles/', dirName, '/', row['Video Title'], '.npz')))

        skew_maxes_index[my_index] = find_abs_array_max(npzfile['skew_list'])
        kurt_maxes_index[my_index] = find_abs_array_max(npzfile['kurtosis_list'])
        skew_d_maxes_index[my_index] = find_abs_array_max(npzfile['d1'])
        kurt_d_maxes_index[my_index] = find_abs_array_max(npzfile['d2'])
        max_maxes_index[my_index] = find_abs_array_max(npzfile['max_list'])

        skew_dur[my_index] = find_peak_duration(npzfile['skew_list'], sampleInterval)
        kurt_dur[my_index] = find_peak_duration(npzfile['kurtosis_list'], sampleInterval)
        skew_d_dur[my_index] = find_peak_duration(npzfile['d1'], sampleInterval)
        kurt_d_dur[my_index] = find_peak_duration(npzfile['d2'], sampleInterval)
        max_dur[my_index] = find_peak_duration(npzfile['max_list'], sampleInterval)

        my_index += 1

    np.savez(''.join((relativePath, 'npzfiles/', dirName)), sm=skew_maxes_index,
             km=kurt_maxes_index, sdm=skew_d_maxes_index, kdm=kurt_d_maxes_index, mm=max_maxes_index,
             sd=skew_dur, kd=kurt_dur, sdd=skew_d_dur, kdd=kurt_d_dur, md=max_dur)

csvfile.close()
