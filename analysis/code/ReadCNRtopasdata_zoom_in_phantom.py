#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper3: read topas data for CNR phantom
Created on Mon Aug 21 14:16:46 2023

@author: hadleys
This is brief code for forming image of zoomed in phantom
"""
#%% initialize
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as spio 
import scipy.ndimage as scipyim
import math
from numba import jit
import matplotlib.patches as patches
import cv2 
from plotSlices import plotSlices

#%% format figures
plt.rcParams.update({
    # figure
    "figure.dpi": 600,
    # text
    "font.size":10,
    "font.family": "serif",
    "font.serif": ['Computer Modern Roman'],
    "text.usetex": True,
    # axes
    "axes.titlesize":10,
    "axes.labelsize":8,
    "axes.linewidth": 1,
    # ticks
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    # grid
    "axes.grid" : False,
     "grid.color": "lightgray",
     "grid.linestyle": ":",
     # legend
     "legend.fontsize":8,
     # lines
     #"lines.markersize":5,
     "lines.linewidth":1,
     })

#%% LOAD IN data FOR GEN 4 PHANTOM ONLY

numEBins = 10                     #num of energy bins
detcolum = 46                      #number of detector pixels in a one column
detrow = 120 #80 
spatialDims = [detcolum, detrow]         #num of detector spatial pixels (y,z) 
testindex = 4                   #need to check that this is the correct index for fluorescent energy
i = 441*detrow                             #number of object pixels

#hexeganal detectors are designated by their cardinal direction relative to a beam's eye view
os.chdir('../../simulations/XFET/DDC_phantom/zoom_in_0.05%gold_superficial_sphere/')
NWdata = np.empty((i, detcolum)) #detector 1, and so on
SWdata = np.empty((i, detcolum)) #detector 2
Ndata = np.empty((i, detcolum)) #detector 3
NEdata = np.empty((i, detcolum)) #detector 4
SEdata = np.empty((i, detcolum)) #detector 5
Sdata = np.empty((i, detcolum)) #detector 6

NWdata_m = np.empty((i, detcolum)) #detector 1, and so on
SWdata_m = np.empty((i, detcolum)) #detector 2
Ndata_m = np.empty((i, detcolum)) #detector 3
NEdata_m = np.empty((i, detcolum)) #detector 4
SEdata_m = np.empty((i, detcolum)) #detector 5
Sdata_m = np.empty((i, detcolum)) #detector 6

NWdata_p = np.empty((i, detcolum)) #detector 1, and so on
SWdata_p = np.empty((i, detcolum)) #detector 2
Ndata_p = np.empty((i, detcolum)) #detector 3
NEdata_p = np.empty((i, detcolum)) #detector 4
SEdata_p = np.empty((i, detcolum)) #detector 5
Sdata_p = np.empty((i, detcolum)) #detector 6

for filename in os.listdir(os.getcwd()):
    if filename[-1] == 'v':   #pick out csvs only
        binnedTracks = np.loadtxt(filename,delimiter=',',usecols=np.linspace(4,numEBins+4,num=numEBins,dtype=int))
        
        #fluoro energy 
        tempdata = binnedTracks[:,testindex] #need to pull out right energy bin first!
        tempdata =np.transpose(np.flip(np.reshape(tempdata, [spatialDims[0],spatialDims[1]]),axis = 1))
        
        #one lower energy bin
        tempdatam = binnedTracks[:,testindex-1] #need to pull out right energy bin first!
        tempdatam =np.transpose(np.flip(np.reshape(tempdatam, [spatialDims[0],spatialDims[1]]),axis = 1))
        
        #one higher enrgy bin
        tempdatap = binnedTracks[:,testindex+1] #need to pull out right energy bin first!
        tempdatap =np.transpose(np.flip(np.reshape(tempdatap, [spatialDims[0],spatialDims[1]]),axis = 1))
        
        idxstart = int(filename[-8:-4])*detrow
        
        if filename[16] == '2':
            SWdata[idxstart:idxstart+detrow,:] = tempdata
            SWdata_p[idxstart:idxstart+detrow,:] = tempdatap
            SWdata_m[idxstart:idxstart+detrow,:] = tempdatam

        if filename[16] == '3':
            Ndata[idxstart:idxstart+detrow,:] = tempdata
            Ndata_p[idxstart:idxstart+detrow,:] = tempdatap
            Ndata_m[idxstart:idxstart+detrow,:] = tempdatam
            
        if filename[16] == '4':
            NEdata[idxstart:idxstart+detrow,:] = tempdata  
            NEdata_p[idxstart:idxstart+detrow,:] = tempdatap     
            NEdata_m[idxstart:idxstart+detrow,:] = tempdatam    

        if filename[16] == '5':
            SEdata[idxstart:idxstart+detrow,:] = tempdata 
            SEdata_p[idxstart:idxstart+detrow,:] = tempdatap   
            SEdata_m[idxstart:idxstart+detrow,:] = tempdatam 

        if filename[16] == '6':
            Sdata[idxstart:idxstart+detrow,:] = tempdata  
            Sdata_p[idxstart:idxstart+detrow,:] = tempdatap 
            Sdata_m[idxstart:idxstart+detrow,:] = tempdatam

        elif filename[12] == '_':
            NWdata[idxstart:idxstart+detrow,:] = tempdata
            NWdata_p[idxstart:idxstart+detrow,:] = tempdatap
            NWdata_m[idxstart:idxstart+detrow,:] = tempdatam



#%% FORM OBJECT IMAGE

xnum = 21
ynum = 21
znum = 120

#sum of counts for each energy bin
meandata = (np.sum(SWdata, 1) + np.sum(NWdata, 1) + np.sum(Ndata, 1) + np.sum(NEdata, 1) + np.sum(SEdata, 1) + np.sum(Sdata, 1))
meandata_m = (np.sum(SWdata_m, 1) + np.sum(NWdata_m, 1) + np.sum(Ndata_m, 1) + np.sum(NEdata_m, 1) + np.sum(SEdata_m, 1) + np.sum(Sdata_m, 1))
meandata_p = (np.sum(SWdata_p, 1) + np.sum(NWdata_p, 1) + np.sum(Ndata_p, 1) + np.sum(NEdata_p, 1) + np.sum(SEdata_p, 1) + np.sum(Sdata_p, 1))

#meanscatter image
meanscatter = (meandata_m + meandata_p)/2

#Scatter reshaped to image space
meanview_s = np.flip(np.flip(np.reshape(meanscatter,(ynum, xnum ,znum)), axis = 2), axis = 1)  #switched from flipping 0,1 to 2,1
plotSlices(meanview_s, 0)
#smooth scatter
smoothed = scipyim.gaussian_filter(meanview_s, sigma = 1)
plotSlices(smoothed, 0)
#rehsape back to data space
meanscatter_smoothed = np.reshape(np.flip(np.flip(smoothed, 1), 2), (xnum*ynum*znum)) #switched flipping from 1,0 to 1,2. Might need to check

#subtract mean data and smoothed scatter
meandatafinal = np.subtract(meandata, meanscatter_smoothed)
#correct for any negative counts
meandatafinal[meandatafinal < 0] = 0

#final meanview is reshaped back to object space
meanview =np.flip(np.reshape(meandata,(ynum, xnum ,znum)), axis = 2)


%matplotlib qt
plotSlices(meanview, 0)     
#save the image
spio.savemat('../inputs/CNR phantom data/gen4/results/zoom_in_object_w_scatter_subtracted.mat',{'meanview':meanview})


