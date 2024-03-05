#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:29:52 2023

@author: hadleys
"""
import numpy as np
import matplotlib.pyplot as plt

#%% defining scroll through function
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = int(self.slices // 2)  #changed np.int to np.int16 because of error, 12/7/23

        self.im = ax.imshow(self.X[:, :, self.ind], cmap="gray")
        #ax.set_title(f'slice {self.ind}')
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :,self.ind])
        self.ax.set_title('slice %s' % self.ind)
        self.ax.axis('off')
        self.im.axes.figure.canvas.draw()


figList,axList=[],[]
def plotSlices(image, view):
    if view == 1:
        image = np.transpose(image, (2, 1, 0))
    elif view == 2:
        image = np.transpose(image, (0, 2, 1))
    fig, ax = plt.subplots(1, 1)
    figList.append(fig)
    axList.append(ax)
    tracker = IndexTracker(axList[-1], image)
    figList[-1].canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show(figList[-1],block=False)
    
