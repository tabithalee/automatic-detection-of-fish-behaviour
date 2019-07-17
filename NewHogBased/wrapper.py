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

reader = {}
relativePath = '/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/'
sheetPath = relativePath + 'sheets/annotated_data.csv'

csv.register_dialect('myDialect', delimiter='|', skipinitialspace=True)

dirName = 'Smooth2x2'

if not os.path.exists(''.join((relativePath, dirName))):
    os.mkdir(''.join((relativePath, dirName)))

with open(sheetPath, 'r') as csvfile:
    reader = csv.DictReader(csvfile, dialect='myDialect')
    for row in reader:
        print(row)
        scratch.main(row['Video Title'], row['Startle Frame'], row['Extra Startle'],
                     [int(item) for item in row['Tracked Regions'].split(',')],
                     True, dirName)

csvfile.close()