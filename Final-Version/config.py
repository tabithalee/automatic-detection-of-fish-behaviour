from configparser import ConfigParser

config = ConfigParser()

config['user_settings'] = {
    'desktopPath': '/home/tabi/Desktop'
}

config['segmentation_settings'] = {
    'videoDirPath': 'automatic-detection-of-fish-behaviour/good_vids/sablefish',
    'video': 'BC_POD1_PTILTVIDEO_20110519T091755.000Z_1.ogg',
    'prevFrame': 'False',
    'minArea': '750',
    'threshVal': '10',
    'dilationIterations': '3',
    'gaussianSize': '61',
    'bufferLength': '72',
    'lineThick': '2',
    'saveFrames': 'True'
}

with open('./dev.ini', 'w') as f:
    config.write(f)