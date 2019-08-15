from of_kalman import process_video_frame

# testing process_video_frame

videoTitle = 'BC_POD1_PTILTVIDEO_20110519T091755.000Z_1'
videoPath = ''.join(('/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/good_vids/training/', videoTitle,
                       '.ogg'))

num_divisions_W = 10
num_divisions_H = 10

hsv2_list, num_times_iterate = process_video_frame(videoPath, num_divisions_W, num_divisions_H)

sum_random_stall_thing = 2