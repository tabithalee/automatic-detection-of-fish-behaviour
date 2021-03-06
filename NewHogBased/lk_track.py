#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import video
from common import anorm2, draw_str
from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.prevTracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0

        # added the following attributes for histogram plotting
        self.numBins = 16
        self.histRange = np.arange(0, 2 * np.pi + (2*np.pi/self.numBins), 2*np.pi/self.numBins)
        self.frameHistogram = np.zeros((self.numBins,))
        self.summedHistogram = np.zeros((self.numBins,))

    def run(self):
        savedPlotCount = 0
        while True:
            _ret, frame = self.cam.read()
            if _ret:
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame_gray = cv.bitwise_not(frame_gray)
                vis = frame_gray.copy()

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.prevTracks = self.tracks
                    self.tracks = new_tracks

                    print(len(self.prevTracks), len(self.tracks))
                    # ensuring the number of tracks are the same size...
                    if len(self.prevTracks) > len(self.tracks):
                        self.prevTracks = self.prevTracks[:len(self.tracks)]
                        print('corrected', len(self.prevTracks), len(self.tracks))

                    if len(self.tracks) > len(self.prevTracks):
                        self.tracks = self.tracks[:len(self.prevTracks)]
                        print('corrected', len(self.tracks), len(self.prevTracks))

                    for track in range(len(self.tracks)-1):
                        #print(track)
                        my_vector = np.array(self.tracks[track]) - np.array(self.prevTracks[track])

                        # add the tracks to a histogram here
                        mag, ang = cv.cartToPolar(my_vector[..., 0], my_vector[..., 1])
                        self.frameHistogram, _bins = np.histogram(ang, bins=self.histRange, weights=mag, density=False)
                        self.summedHistogram += self.frameHistogram

                    cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv.circle(mask, (x, y), 5, 0, -1)
                    p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])
                        self.prevTracks = self.tracks

                plt.subplot(2, 1, 1)
                plt.bar(self.histRange[:-1], self.summedHistogram, align='edge', width=2 *np.pi / self.numBins)

                self.frame_idx += 1
                self.prev_gray = frame_gray

                plt.subplot(2, 1, 2)
                plt.imshow(vis)
                plt.pause(0.001)

                figNameString = '/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/' \
                                + '{0:08}'.format(savedPlotCount) + '.png'
                plt.savefig(figNameString)
                plt.clf()

                self.summedHistogram = np.zeros((self.numBins,))
                savedPlotCount +=1



            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
