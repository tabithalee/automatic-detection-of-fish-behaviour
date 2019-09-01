# Automatic Detection Of Fish Behaviour

A summer NSERC project at the University of Victoria focussing on detecting startle-type fish behaviour with opencv and python. More information about the project can be found in the NSERC USRA 2019 pdf file.

## Getting Started
The following steps will help get a running copy on your local machine for testing and development.

### Prerequisites
You will need to have at least Python 3.5.2 installed on your computer as well as the python OpenCV 2.4.1.0 framework. Ensure when running that you have the correct version of Python and OpenCV, as previous versions are not completely compatible with the code. Some other libraries needed are used such as numpy and matplotlib, but you can install them as necessary.

Once the pre-requisites are installed, you should be able to run the code in Final- Version. This is where all the code mentioned in the paper is. All other folders are development code and therefore not clean.

## Deployment
All code has been run from PyCharm Community Edition IDE and is suggested that code be run from there.

The two approaches mentioned in the paper can be run separately. What follows are instructions for running the trajectory-based approaches (Lucas-Kanade and segmentation) and for running the main, histogram-based approach.

Both approaches work from the settings defined in config.py. This creates a dev.ini file which is parsed into code for running. The user should edit make sure the path to the repo is defined correctly as necessary. This is under the 'user_settings' section of config.py. The config.py file has to be run every time a setting is changed so that the dev.ini can be updated.

### Trajectory-based approach
All settings for this approach can be specified in the section 'segmentation_settings' section of config.py. The main settings to set are 'saveFrames', 'lk_mode', and 'seg_mode'.
* 'saveFrames' - if saveFrames is true, then each frame will be saved for your perusal later in the main directory
* 'lk_mode' - if True, then lucas kanade algorithm will be run.
* 'seg_mode' - if True, then the segmentation algorithm will be run. Both this setting and 'lk_mode' can be True.

The file lk_test.py is only a slightly modified version of the file of the same name provided in the OpenCV python samples.

Once the config.py file has been run, trajectories.py can be run for the trajectory-based approaches. 

### Histogram-based approach
For this approach, you can choose to run the code for all the videos in the set or just a single video

#### Single Video
For a single video the settings for this approach are under the section 'single_main_settings' in the config.py file. The settings are explained below:
* videoTitle - the file name of the video (assuming that it is in the good_vids/sablefish directory)
* startleFrame - the frame at which startle should be drawn on the skewness and kurtosis plots
* extraStartle - the frame at which an extra startle should be drawn. If no extra startle, use -1.
* saveToFolder - If true, create a directory for each video and save each figure to the directory sepcified by dirName. If false figures will not be saved
* dirName - The folder name of the directory to save histograms under (will be in savedHistograms directory)
* sampleInterval - the sampling interval
* numDivX - number of horizontal divisions of each frame
* numDivY - number of vertical divisions of each frame
* numHistBins - number of bins for the histograms
* saveHistograms - If true, save the histogram figures (not the same as the optical flow figures)
* saveData - If true, save the raw skew, kurtosis, skew derivative, kurtosis derivative, and max arrays of each video into a .npz file. This can be used for analyzing later
* fps - the number of frames per second of the video
* roi - the regions which you wish to track in each frame of the video. They start from 1 and go left to right, row by row. For example, the regions of a 3x3 are labelled as follows

'---'---'---'

| 1 | 2 | 3 |

'---'---'---'

| 4 | 5 | 6 |

'---'---'---'

| 7 | 8 | 9 |

'---'---'---'

If we only wanted to look at the top row, we would specify the roi as a list '1,2,3', separated only by commas (no spaces). 
After the config.py is run, we can run main.py for processing just one video.

#### Multiple Videos
The settings for running the code for multiple videos can be modified in the 'main_settings' section of the config.py file. The settings are similar to the single video case, with a few extra settings below:
* allWindows - a way to keep track of the entire frame at once, instead of having to specify for each video in the set
* csvFile - all the videos and their data are stored in the csv file specified. (We are assuming that the csvfile is in the directory savedHistograms/sheets. If creating a new csv file, the csvfile should be delimited by a '|' instead of the usual ','.
* numGoodVids - the number of videos in the video set.

After the config.py file is run, you can run wrapper.py. If you specified saveData to be true, you can find the .npz files in the directory savedHistograms/npzfiles. You can then load these arrays for further analyzing the skewness, kurtosis, and their first derivatives of your dataset.

### Other Files
Two other files have been included, but are currently unused in our current implementations. Both contain only python methods, so they will have to be written into their own code. These two files are 
* hog_functions.py
* mbh.py

## Authors
* Tabitha Lee
* Supervisor: Alexandra Branzan Albu



