from configparser import ConfigParser

config = ConfigParser()

config['user_settings'] = {
    'repoPath': '/home/tabi/Desktop/automatic-detection-of-fish-behaviour'
}

config['segmentation_settings'] = {
    'videoDirPath': 'good_vids/sablefish',
    'video': 'BC_POD1_PTILTVIDEO_20110519T091755.000Z_1.ogg',
    'prevFrame': 'False',
    'minArea': '750',
    'threshVal': '10',
    'dilationIterations': '3',
    'gaussianSize': '61',
    'bufferLength': '72',
    'lineThick': '2',
    'saveFrames': 'False',
    'lk_mode': 'False',
    'seg_mode': 'True'
}

config['main_settings'] = {
    'dirName': '3_data_process',
    'allWindows': 'False',
    'sampleInterval': '5',
    'csvFile': 'annotated_data_3x3_2.csv',
    'numGoodVids': '27',
    'numDivX': '3',
    'numDivY': '3',
    'numHistBins': '16',
    'saveHistograms': 'True',
    'saveData': 'True',
    'fps': '15'
}

config['single_main_settings'] = {
    'videoTitle': 'BC_POD1_PTILTVIDEO_20110519T091755.000Z_1.ogg',
    'startleFrame': '22',
    'extraStartle': 'False',
    'saveToFolder': 'False',
    'dirName': 'single_video',
    'sampleInterval': '5',
    'numDivX': '3',
    'numDivY': '3',
    'numHistBins': '16',
    'saveHistograms': 'True',
    'saveData': 'True',
    'fps': '15',
    'roi': '1,2,3'
}

with open('./dev.ini', 'w') as f:
    config.write(f)
