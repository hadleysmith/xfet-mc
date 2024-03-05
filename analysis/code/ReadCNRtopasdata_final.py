#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper3: read topas data for CNR phantom
Created on Mon Aug 21 14:16:46 2023

@author: hadleys
This is the main code for reading in DDC phantom data and computing CNRs
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from get_CNR import find_DDC_CNR_from_circleROI
from get_CNR import find_DDC_CNR_from_circleROI_XFET
from convert_coordinates import *


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
testindex = 4                   #need to check that this is the correct index for fluorescent energy. previously 69
i = 441*detrow #4096*detrow #8000                            #number of object pixels

#hexegan detectors are designated by their cardinal direction relative to a beam's eye view

#choose path for run:
#os.chdir('../../simulations/XFET/DDC_phantom/zoom_in_0.05%gold_superficial_sphere/')
os.chdir('../../simulations/XFET/DDC_phantom/depth_1+2/run1')


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


#%% save arrays----

spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/SWdata.mat',{'SWdata':SWdata})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/NWdata.mat',{'NWdata':NWdata})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/Ndata.mat',{'Ndata': Ndata})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/NEdata.mat',{'NEdata':NEdata})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/SEdata.mat',{'SEdata':SEdata})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/Sdata.mat',{'Sdata': Sdata})


spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/SWdata_p.mat',{'SWdata_p':SWdata_p})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/NWdata_p.mat',{'NWdata_p':NWdata_p})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/Ndata_p.mat',{'Ndata_p': Ndata_p})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/NEdata_p.mat',{'NEdata_p':NEdata_p})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/SEdata_p.mat',{'SEdata_p':SEdata_p})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/Sdata_p.mat',{'Sdata_p': Sdata_p})


spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/SWdata_m.mat',{'SWdata_m':SWdata_m})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/NWdata_m.mat',{'NWdata_m':NWdata_m})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/Ndata_m.mat',{'Ndata_m': Ndata_m})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/NEdata_m.mat',{'NEdata_m':NEdata_m})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/SEdata_m.mat',{'SEdata_m':SEdata_m})
spio.savemat('../inputs/CNR phantom data/gen4/0.5mm_res/Sdata_m.mat',{'Sdata_m': Sdata_m})

#%% load in saved arrays
SWdata = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/SWdata.mat')['SWdata']
NWdata = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/NWdata.mat')['NWdata']
Ndata = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/Ndata.mat')['Ndata']
NEdata = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/NEdata.mat')['NEdata']
SEdata = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/SEdata.mat')['SEdata']
Sdata = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/Sdata.mat')['Sdata']

SWdata_p = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/SWdata_p.mat')['SWdata_p']
NWdata_p = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/NWdata_p.mat')['NWdata_p']
Ndata_p = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/Ndata_p.mat')['Ndata_p']
NEdata_p = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/NEdata_p.mat')['NEdata_p']
SEdata_p = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/SEdata_p.mat')['SEdata_p']
Sdata_p = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/Sdata_p.mat')['Sdata_p']

SWdata_m = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/SWdata_m.mat')['SWdata_m']
NWdata_m = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/NWdata_m.mat')['NWdata_m']
Ndata_m = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/Ndata_m.mat')['Ndata_m']
NEdata_m = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/NEdata_m.mat')['NEdata_m']
SEdata_m = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/SEdata_m.mat')['SEdata_m']
Sdata_m = spio.loadmat('../inputs/CNR phantom data/gen4/0.5mm_res/Sdata_m.mat')['Sdata_m']

#%% FORM OBJECT IMAGE

xnum = 64
ynum = 64
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
meanview =np.flip(np.reshape(meandatafinal,(ynum, xnum ,znum)), axis = 2)


%matplotlib qt
plotSlices(meanview, 0)     
#save the image
spio.savemat('../inputs/CNR phantom data/gen4/results/object_w_scatter_subtracted_run3.mat',{'meanview':meanview})



#%% load in CT images

#load in CT images
#EID
CT_EID_64_run1 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x1_waterBHC_HU_64_float32.bin", dtype=np.float32)
CT_EID_64_run2 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x2_waterBHC_HU_64_float32.bin", dtype=np.float32)
CT_EID_64_run3 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x3_waterBHC_HU_64_float32.bin", dtype=np.float32)
CT_EID_64_run4 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x4_waterBHC_HU_64_float32.bin", dtype=np.float32)
CT_EID_64_run5 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x5_waterBHC_HU_64_float32.bin", dtype=np.float32)

CT_EID_256_run1 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x1_waterBHC_HU_256_float32.bin", dtype=np.float32)
CT_EID_256_run2 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x2_waterBHC_HU_256_float32.bin", dtype=np.float32)
CT_EID_256_run3 = np.fromfile("/../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x3_waterBHC_HU_256_float32.bin", dtype=np.float32)
CT_EID_256_run4 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x4_waterBHC_HU_256_float32.bin", dtype=np.float32)
CT_EID_256_run5 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_EID/srecon_x5_waterBHC_HU_256_float32.bin", dtype=np.float32)


CT_PCD_64_run1 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x1_waterBHC_HU_64_float32.bin", dtype=np.float32)
CT_PCD_64_run2 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x2_waterBHC_HU_64_float32.bin", dtype=np.float32)
CT_PCD_64_run3 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x3_waterBHC_HU_64_float32.bin", dtype=np.float32)
CT_PCD_64_run4 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x4_waterBHC_HU_64_float32.bin", dtype=np.float32)
CT_PCD_64_run5 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x5_waterBHC_HU_64_float32.bin", dtype=np.float32)

CT_PCD_256_run1 = np.fromfile("/../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x1_waterBHC_HU_256_float32.bin", dtype=np.float32)
CT_PCD_256_run2 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x2_waterBHC_HU_256_float32.bin", dtype=np.float32)
CT_PCD_256_run3 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x3_waterBHC_HU_256_float32.bin", dtype=np.float32)
CT_PCD_256_run4 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x4_waterBHC_HU_256_float32.bin", dtype=np.float32)
CT_PCD_256_run5 = np.fromfile("../inputs/xtomosim/output/spie2024_xfet/DDC_PCD/srecon_x5_waterBHC_HU_256_float32.bin", dtype=np.float32)


#reshape
CT_EID_64_run1 = np.flip(np.reshape(CT_EID_64_run1, [64,64]), 0)
CT_EID_64_run2 = np.flip(np.reshape(CT_EID_64_run2, [64,64]), 0)
CT_EID_64_run3 = np.flip(np.reshape(CT_EID_64_run3, [64,64]), 0)
CT_EID_64_run4 = np.flip(np.reshape(CT_EID_64_run4, [64,64]), 0)
CT_EID_64_run5 = np.flip(np.reshape(CT_EID_64_run5, [64,64]), 0)

CT_EID_256_run1 = np.flip(np.reshape(CT_EID_256_run1, [256,256]), 0)
CT_EID_256_run2 = np.flip(np.reshape(CT_EID_256_run2, [256,256]), 0)
CT_EID_256_run3 = np.flip(np.reshape(CT_EID_256_run3, [256,256]), 0)
CT_EID_256_run4 = np.flip(np.reshape(CT_EID_256_run4, [256,256]), 0)
CT_EID_256_run5 = np.flip(np.reshape(CT_EID_256_run5, [256,256]), 0)

CT_PCD_64_run1 = np.flip(np.reshape(CT_PCD_64_run1, [64,64]), 0)
CT_PCD_64_run2 = np.flip(np.reshape(CT_PCD_64_run2, [64,64]), 0)
CT_PCD_64_run3 = np.flip(np.reshape(CT_PCD_64_run3, [64,64]), 0)
CT_PCD_64_run4 = np.flip(np.reshape(CT_PCD_64_run4, [64,64]), 0)
CT_PCD_64_run5 = np.flip(np.reshape(CT_PCD_64_run5, [64,64]), 0)

CT_PCD_256_run1 = np.flip(np.reshape(CT_PCD_256_run1, [256,256]), 0)
CT_PCD_256_run2 = np.flip(np.reshape(CT_PCD_256_run2, [256,256]), 0)
CT_PCD_256_run3 = np.flip(np.reshape(CT_PCD_256_run3, [256,256]), 0)
CT_PCD_256_run4 = np.flip(np.reshape(CT_PCD_256_run4, [256,256]), 0)
CT_PCD_256_run5 = np.flip(np.reshape(CT_PCD_256_run5, [256,256]), 0)


#%% load in XFET images pull out ROIs

#select run (1 through 5)
run = "run2"

#load in  XFET data

#DEPTH 1 and 2
meanview = spio.loadmat(f'../inputs/CNR phantom data/gen4/results/object_w_scatter_subtracted_{run}.mat')['meanview']
newmeanview = np.zeros((64,64,60))  #increasing z bin width to 2 mm
for i in np.arange(60):
   newmeanview[:,:,i] = meanview[:,:,2*i] + meanview[:,:,2*i+1]
meanview = newmeanview.copy()

#DEPTH 3 and 4
meanview2 = spio.loadmat(f'../inputs/CNR phantom data/gen4/results/object_w_scatter_subtracted_additional_depth_{run}.mat')['meanview']
newmeanview2 = np.zeros((64,64,40)) #increasing z bin width to 2 mm
for i in np.arange(40):
   newmeanview2[:,:,i] = meanview2[:,:,2*i] + meanview2[:,:,2*i+1]
meanview2 = newmeanview2.copy()


#pull out 4 depths
XFET_depth1 = meanview[:,:,23]
XFET_depth2 = meanview[:,:,36]
XFET_depth3 = meanview2[:,:,23]
XFET_depth4 = meanview2[:,:,36]

#find CNRs
CNR_XFET_depth1 = find_DDC_CNR_from_circleROI_XFET(XFET_depth1, "shallow", 1)
CNR_XFET_depth2 = find_DDC_CNR_from_circleROI_XFET(XFET_depth2, "deep", 1)
CNR_XFET_depth3 = find_DDC_CNR_from_circleROI_XFET(XFET_depth3, "shallow", 1)
CNR_XFET_depth4 = find_DDC_CNR_from_circleROI_XFET(XFET_depth4, "deep", 1)

#save
spio.savemat(f'../inputs/CNR phantom data/gen4/results/XFET_depth_79.75mm_{run}.mat',{'XFET_deep':CNR_XFET_depth4})
spio.savemat(f'../inputs/CNR phantom data/gen4/results/XFET_depth_54.25mm_{run}.mat',{'XFET_shallow':CNR_XFET_depth3})
spio.savemat(f'../inputs/CNR phantom data/gen4/results/XFET_depth_28.75mm_{run}.mat',{'XFET_deep':CNR_XFET_depth2})
spio.savemat(f'../inputs/CNR phantom data/gen4/results/XFET_depth_3.25mm_{run}.mat',{'XFET_shallow':CNR_XFET_depth1})

#%% pull out ROIs CT 
#find CNRS
CNR_EICT_64_run1 = find_DDC_CNR_from_circleROI(CT_EID_64_run1, 1)
CNR_EICT_64_run2 = find_DDC_CNR_from_circleROI(CT_EID_64_run2, 0)
CNR_EICT_64_run3 = find_DDC_CNR_from_circleROI(CT_EID_64_run3, 0)
CNR_EICT_64_run4 = find_DDC_CNR_from_circleROI(CT_EID_64_run4, 0)
CNR_EICT_64_run5 = find_DDC_CNR_from_circleROI(CT_EID_64_run5, 0)

CNR_EICT_256_run1 = find_DDC_CNR_from_circleROI(CT_EID_256_run1, 1)
CNR_EICT_256_run2 = find_DDC_CNR_from_circleROI(CT_EID_256_run2, 0)
CNR_EICT_256_run3 = find_DDC_CNR_from_circleROI(CT_EID_256_run3, 0)
CNR_EICT_256_run4 = find_DDC_CNR_from_circleROI(CT_EID_256_run4, 0)
CNR_EICT_256_run5 = find_DDC_CNR_from_circleROI(CT_EID_256_run5, 0)

CNR_PCCT_64_run1 = find_DDC_CNR_from_circleROI(CT_PCD_64_run1, 1)
CNR_PCCT_64_run2 = find_DDC_CNR_from_circleROI(CT_PCD_64_run2, 0)
CNR_PCCT_64_run3 = find_DDC_CNR_from_circleROI(CT_PCD_64_run3, 0)
CNR_PCCT_64_run4 = find_DDC_CNR_from_circleROI(CT_PCD_64_run4, 0)
CNR_PCCT_64_run5 = find_DDC_CNR_from_circleROI(CT_PCD_64_run5, 0)

CNR_PCCT_256_run1 = find_DDC_CNR_from_circleROI(CT_PCD_256_run1, 1)
CNR_PCCT_256_run2 = find_DDC_CNR_from_circleROI(CT_PCD_256_run2, 0)
CNR_PCCT_256_run3 = find_DDC_CNR_from_circleROI(CT_PCD_256_run3, 0)
CNR_PCCT_256_run4 = find_DDC_CNR_from_circleROI(CT_PCD_256_run4, 0)
CNR_PCCT_256_run5 = find_DDC_CNR_from_circleROI(CT_PCD_256_run5, 0)

#save
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_64_run1.mat',{'CNR_EICT_64_run1':CNR_EICT_64_run1})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_64_run2.mat',{'CNR_EICT_64_run2':CNR_EICT_64_run2})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_64_run3.mat',{'CNR_EICT_64_run3':CNR_EICT_64_run3})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_64_run4.mat',{'CNR_EICT_64_run4':CNR_EICT_64_run4})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_64_run5.mat',{'CNR_EICT_64_run5':CNR_EICT_64_run5})

spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_256_run1.mat',{'CNR_EICT_256_run1':CNR_EICT_256_run1})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_256_run2.mat',{'CNR_EICT_256_run2':CNR_EICT_256_run2})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_256_run3.mat',{'CNR_EICT_256_run3':CNR_EICT_256_run3})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_256_run4.mat',{'CNR_EICT_256_run4':CNR_EICT_256_run4})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_EICT_256_run5.mat',{'CNR_EICT_256_run5':CNR_EICT_256_run5})

spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_64_run1.mat',{'CNR_PCCT_64_run1':CNR_PCCT_64_run1})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_64_run2.mat',{'CNR_PCCT_64_run2':CNR_PCCT_64_run2})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_64_run3.mat',{'CNR_PCCT_64_run3':CNR_PCCT_64_run3})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_64_run4.mat',{'CNR_PCCT_64_run4':CNR_PCCT_64_run4})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_64_run5.mat',{'CNR_PCCT_64_run5':CNR_PCCT_64_run5})

spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_256_run1.mat',{'CNR_PCCT_256_run1':CNR_PCCT_256_run1})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_256_run2.mat',{'CNR_PCCT_256_run2':CNR_PCCT_256_run2})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_256_run3.mat',{'CNR_PCCT_256_run3':CNR_PCCT_256_run3})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_256_run4.mat',{'CNR_PCCT_256_run4':CNR_PCCT_256_run4})
spio.savemat('../inputs/CNR phantom data/gen4/results/CNR_PCCT_256_run5.mat',{'CNR_PCCT_256_run5':CNR_PCCT_256_run5})




#%% CNR analysis & main plot
gold_concentrations = [0.05, 0.5,1,1.5,2,4]

#load in XFET CNR data 
#3.25 mm
Xdepth1_run1 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_3.25mm_run1.mat')['XFET_shallow']
Xdepth1_run2 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_3.25mm_run2.mat')['XFET_shallow']
Xdepth1_run3 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_3.25mm_run3.mat')['XFET_shallow']
Xdepth1_run4 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_3.25mm_run4.mat')['XFET_shallow']
Xdepth1_run5 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_3.25mm_run5.mat')['XFET_shallow']
#28.75 mm
Xdepth2_run1  = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_28.75mm_run1.mat')['XFET_deep']
Xdepth2_run2 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_28.75mm_run2.mat')['XFET_deep']
Xdepth2_run3 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_28.75mm_run3.mat')['XFET_deep']
Xdepth2_run4 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_28.75mm_run4.mat')['XFET_deep']
Xdepth2_run5 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_28.75mm_run5.mat')['XFET_deep']
#54.75 mm
Xdepth3_run1 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_54.25mm_run1.mat')['XFET_shallow']
Xdepth3_run2 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_54.25mm_run2.mat')['XFET_shallow']
Xdepth3_run3 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_54.25mm_run3.mat')['XFET_shallow']
Xdepth3_run4 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_54.25mm_run4.mat')['XFET_shallow']
Xdepth3_run5 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_54.25mm_run5.mat')['XFET_shallow']
#79.75 mm
Xdepth4_run1 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_79.75mm_run1.mat')['XFET_deep']
Xdepth4_run2 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_79.75mm_run2.mat')['XFET_deep']
Xdepth4_run3 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_79.75mm_run3.mat')['XFET_deep']
Xdepth4_run4 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_79.75mm_run4.mat')['XFET_deep']
Xdepth4_run5 = spio.loadmat('../inputs/CNR phantom data/gen4/results/XFET_depth_79.75mm_run5.mat')['XFET_deep']

#mean and standard error for each noise realization
n = 5 #sample size
mean_depth1 = np.squeeze((Xdepth1_run1 + Xdepth1_run2 + Xdepth1_run3 + Xdepth1_run4 + Xdepth1_run5)/5)
std_depth1 = np.squeeze(np.std([Xdepth1_run1, Xdepth1_run2, Xdepth1_run3, Xdepth1_run4, Xdepth1_run5],0))/n

mean_depth2 = np.squeeze((Xdepth2_run1 + Xdepth2_run2 + Xdepth2_run3 + Xdepth2_run4 + Xdepth2_run5)/5)
std_depth2 = np.squeeze(np.std([Xdepth2_run1, Xdepth2_run2, Xdepth2_run3, Xdepth2_run4, Xdepth2_run5],0))/n

mean_depth3 = np.squeeze((Xdepth3_run1 + Xdepth3_run2 + Xdepth3_run3 + Xdepth3_run4 + Xdepth3_run5)/5)
std_depth3 = np.squeeze(np.std([Xdepth3_run1, Xdepth3_run2, Xdepth3_run3, Xdepth3_run4, Xdepth3_run5],0))/n

mean_depth4 = np.squeeze((Xdepth4_run1 + Xdepth4_run2 + Xdepth4_run3 + Xdepth4_run4 + Xdepth4_run5)/5)
std_depth4 = np.squeeze(np.std([Xdepth4_run1, Xdepth4_run2, Xdepth4_run3, Xdepth4_run4, Xdepth4_run5],0))/n

CT_EID_64_CNR_mean = np.squeeze((CNR_EICT_64_run1 + CNR_EICT_64_run2 + CNR_EICT_64_run3 + CNR_EICT_64_run4 + CNR_EICT_64_run5)/5)
CT_EID_64_CNR_stdev = np.squeeze(np.std([CNR_EICT_64_run1, CNR_EICT_64_run2, CNR_EICT_64_run3, CNR_EICT_64_run4, CNR_EICT_64_run5],0))/n

CT_EID_256_CNR_mean = np.squeeze((CNR_EICT_256_run1 + CNR_EICT_256_run2 + CNR_EICT_256_run3 + CNR_EICT_256_run4 + CNR_EICT_256_run5)/5)
CT_EID_256_CNR_stdev = np.squeeze(np.std([CNR_EICT_256_run1, CNR_EICT_256_run2, CNR_EICT_256_run3, CNR_EICT_256_run4, CNR_EICT_256_run5],0))/n

CT_PCD_64_CNR_mean = np.squeeze((CNR_PCCT_64_run1 + CNR_PCCT_64_run2 + CNR_PCCT_64_run3 + CNR_PCCT_64_run4 + CNR_PCCT_64_run5)/5)
CT_PCD_64_CNR_stdev = np.squeeze(np.std([CNR_PCCT_64_run1, CNR_PCCT_64_run2, CNR_PCCT_64_run3, CNR_PCCT_64_run4, CNR_PCCT_64_run5],0))/n

CT_PCD_256_CNR_mean = np.squeeze((CNR_PCCT_256_run1 + CNR_PCCT_256_run2 + CNR_PCCT_256_run3 + CNR_PCCT_256_run4 + CNR_PCCT_256_run5)/5)
CT_PCD_256_CNR_stdev = np.squeeze(np.std([CNR_PCCT_256_run1, CNR_PCCT_256_run2, CNR_PCCT_256_run3, CNR_PCCT_256_run4, CNR_PCCT_256_run5],0))/n


#LINEAR FIT 
x = np.array(gold_concentrations)
m1, b1 = np.polyfit(gold_concentrations, mean_depth1, 1)
m2, b2 = np.polyfit(gold_concentrations, mean_depth2, 1)
m3, b3 = np.polyfit(gold_concentrations, mean_depth3, 1)
m4, b4 = np.polyfit(gold_concentrations, mean_depth4, 1)
m5, b5 = np.polyfit(gold_concentrations, CT_EID_64_CNR_mean, 1)
m6, b6 = np.polyfit(gold_concentrations, CT_EID_256_CNR_mean, 1)
m7, b7 = np.polyfit(gold_concentrations, CT_PCD_64_CNR_mean, 1)
m8, b8 = np.polyfit(gold_concentrations, CT_PCD_256_CNR_mean, 1)

#plot CNR graph
%matplotlib inline
plt.figure(dpi = 700, figsize=[5.4,3.7])
plt.errorbar(gold_concentrations, mean_depth1, yerr = std_depth1, ls='none', color = 'red', marker = 'o', markersize=3,  label = 'XFET 3.25 mm depth')
plt.plot(gold_concentrations, m1*x+b1, '-', color = 'red', linewidth = 0.5)
plt.errorbar(gold_concentrations, mean_depth2, yerr = std_depth2, color = 'blue',ls='none', marker = 'v', markersize=3, label = 'XFET 28.75 mm depth')
plt.plot(gold_concentrations, m2*x+b2, '-b',linewidth = 0.5)
plt.errorbar(gold_concentrations, mean_depth3, yerr = std_depth3, color = 'cyan',ls='none', marker = '^', markersize=3, zorder = 3, label = 'XFET 54.25 mm depth')
plt.plot(gold_concentrations, m3*x+b3, '-c',linewidth = 0.5)
plt.errorbar(gold_concentrations, mean_depth4, yerr = std_depth4, color = 'gold',ls='none', marker = 'd', markersize=3, label = 'XFET 79.75 mm depth')
plt.plot(gold_concentrations, m4*x+b4, '-', color = 'gold', linewidth = 0.5)
plt.errorbar(gold_concentrations, CT_EID_64_CNR_mean, yerr = CT_EID_64_CNR_stdev, color = 'green',ls='none', marker = 's', markersize=3, label = 'EID CT (matched resolution)')
plt.plot(gold_concentrations, m5*x+b5, '-g',linewidth = 0.5)
plt.errorbar(gold_concentrations, CT_EID_256_CNR_mean, yerr = CT_EID_256_CNR_stdev, ls='none', marker = 's', markerfacecolor='w', markeredgecolor = 'black', markeredgewidth = 0.5, markersize = 5, zorder = 1, label = 'EID CT (high resolution)')
plt.plot(gold_concentrations, m6*x+b6, '--g',linewidth = 0.5)
plt.errorbar(gold_concentrations, CT_PCD_64_CNR_mean, yerr = CT_PCD_64_CNR_stdev, color = 'magenta',ls='none', marker = 'o', markersize=3, label = 'PCD CT (matched resolution)')
plt.plot(gold_concentrations, m7*x+b7, '-m',linewidth = 0.5)
plt.errorbar(gold_concentrations, CT_PCD_256_CNR_mean, yerr = CT_PCD_256_CNR_stdev, color = 'magenta',ls='none', marker = 'o', markerfacecolor='w', markeredgecolor = 'black', markeredgewidth = 0.5, markersize = 5 ,zorder = 1, label = 'PCD CT (high resolution)')
plt.plot(gold_concentrations, m8*x+b8, '--m',linewidth = 0.5)
plt.axhline(y=4, color='black', linestyle='--', linewidth=0.5, label='Rose Criterion', zorder= -1)

plt.legend()
plt.xlabel(f'Gold concentration in soft tissue (\% by weight)')
plt.ylabel('CNR')


#%% Detection limit calculation using Rose criterion CNR = 4.

depths = [3.25, 28.75, 54.25, 79.75]   #mm
detection_limits = [(4-b1)/m1, (4-b2)/m2, (4-b3)/m3, (4-b4)/m4]   #rose criterion: CNR = 4

#define function
def monoExp(x, m, t, b):
    return m * np.exp(t * x) + b


from scipy.optimize import curve_fit
p0 = (1, .002, .50) # start with values near those we expect

# Fit the function a * np.exp(b * t) + c to x and y
popt, pcov = curve_fit(monoExp, depths, detection_limits, p0)
a = popt[0]
b = popt[1]
c = popt[2]
x_fitted = np.linspace(np.min(depths), np.max(depths), 100)
y_fitted = monoExp(x_fitted, a, b, c)

#round numbers for purpose of display
ar = np.round(a,2)
br = np.round(b, 3)
cr = np.round(c, 3)
equation_text = f'$y = {ar}*exp({br}x) {cr}$'


plt.figure(dpi = 500, figsize=[5.4,3.7])
plt.plot(depths, detection_limits, ls = 'none', marker = '*', color = 'red')
plt.plot(x_fitted, y_fitted, 'k', label='Fitted curve', color = 'red', linestyle ='-')
plt.xlabel('Beam depth of XFET imaging (mm)')
plt.ylabel(f'Detection limit (\% gold by weight in soft tissue)')
plt.text(0.5, 0.90, equation_text, transform=plt.gca().transAxes, ha='center', va='center', backgroundcolor='white', color = 'red')

#%% PLOT images

#row 1: XFET images, increasing depth as you go left to right
%matplotlib inline
plt.figure(dpi = 1000)
plt.subplot(2,4,1)
plt.imshow(XFET_depth1, cmap = 'bone', vmin = 0, vmax = 160)
#plt.colorbar()
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(XFET_depth2, cmap = 'bone',  vmin = 0, vmax = 160)
#plt.colorbar()
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(XFET_depth3, cmap = 'bone', vmin = 0, vmax = 160)
#plt.colorbar()
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(XFET_depth4, cmap = 'bone',  vmin = 0, vmax = 160)
#plt.colorbar()
plt.axis('off')
#row 2: CT, EID 64 then 256, PCD 64 then 256
plt.subplot(2,4,5)
plt.imshow(CT_EID_64_run1, cmap = 'bone', vmin = -1500, vmax = 1200)
#plt.colorbar()
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(CT_EID_256_run1, cmap = 'bone', vmin = -1500, vmax = 1200)
#plt.colorbar()
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(CT_PCD_64_run1, cmap = 'bone', vmin = -1500, vmax = 1200)
#plt.colorbar()
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(CT_PCD_256_run1, cmap = 'bone', vmin = -1500, vmax = 1200)
#plt.colorbar()
plt.axis('off')


#JUST XFET ----------------------------------------------------------------------------------
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# Create a figure and a gridspec layout
fig = plt.figure(dpi=1000, figsize=(7, 1.8))
gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05])
ax1 = plt.subplot(gs[0])
ax1.imshow(XFET_depth1, cmap='bone', vmin=0, vmax=160)
ax1.axis('off')
ax2 = plt.subplot(gs[1])
ax2.imshow(XFET_depth2, cmap='bone', vmin=0, vmax=160)
ax2.axis('off')
ax3 = plt.subplot(gs[2])
ax3.imshow(XFET_depth3, cmap='bone', vmin=0, vmax=160)
ax3.axis('off')
ax4 = plt.subplot(gs[3])
ax4.imshow(XFET_depth4, cmap='bone', vmin=0, vmax=160)
ax4.axis('off')
# Create a ScalarMappable object
norm = Normalize(vmin=0, vmax=160)
sm = ScalarMappable(norm=norm, cmap='bone')
sm.set_array([])
# Add colorbar as a separate subplot
cax = plt.subplot(gs[:,4])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('Counts')
plt.tight_layout()




fig = plt.figure(dpi=1000, figsize = (3.7,3))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1])
# Add subplots
ax1 = plt.subplot(gs[0, 0])
ax1.imshow(XFET_depth1, cmap='bone', vmin=0, vmax=160)
ax1.axis('off')
ax2 = plt.subplot(gs[0, 1])
ax2.imshow(XFET_depth2, cmap='bone', vmin=0, vmax=160)
ax2.axis('off')
ax3 = plt.subplot(gs[1, 0])
ax3.imshow(XFET_depth3, cmap='bone', vmin=0, vmax=160)
ax3.axis('off')
ax4 = plt.subplot(gs[1, 1])
ax4.imshow(XFET_depth4, cmap='bone', vmin=0, vmax=160)
ax4.axis('off')
# Create a ScalarMappable object
norm = Normalize(vmin=0, vmax=160)
sm = ScalarMappable(norm=norm, cmap='bone')
sm.set_array([])
# Adjust colorbar width to match subfigures
cax = plt.subplot(gs[:, 2])  # Span both rows
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('Fluorescence counts')
plt.tight_layout()
#plt.show()




#JUST CT----------------------------------------------------------------------------------
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# Create a figure and a gridspec layout
fig = plt.figure(dpi=1000, figsize = (3.7,3))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1])
# Add subplots
ax1 = plt.subplot(gs[0, 0])
ax1.imshow(CT_EID_64_run1, cmap='bone', vmin = -1500, vmax = 1200)
ax1.axis('off')
ax2 = plt.subplot(gs[0, 1])
ax2.imshow(CT_EID_256_run1, cmap='bone', vmin = -1500, vmax = 1200)
ax2.axis('off')
ax3 = plt.subplot(gs[1, 0])
ax3.imshow(CT_PCD_64_run1, cmap='bone', vmin = -1500, vmax = 1200)
ax3.axis('off')
ax4 = plt.subplot(gs[1, 1])
ax4.imshow(CT_PCD_256_run1, cmap='bone', vmin = -1500, vmax = 1200)
ax4.axis('off')
# Create a ScalarMappable object
norm = Normalize(vmin = -1500, vmax = 1200)
sm = ScalarMappable(norm=norm, cmap='bone')
sm.set_array([])
# Adjust colorbar width to match subfigures
cax = plt.subplot(gs[:, 2])  # Span both rows
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('HU')
plt.tight_layout()
#plt.show()

#%% zoom in image processing and imaging

#load XFET image and rebin
meanview = spio.loadmat('../inputs/CNR phantom data/gen4/results/object_w_scatter_subtracted.mat')['meanview']
newmeanview = np.zeros((64,64,60))
for i in np.arange(60):
   newmeanview[:,:,i] = meanview[:,:,2*i] + meanview[:,:,2*i+1]
meanview = newmeanview.copy()
XFET_depth1 = meanview[:,:,23]


#load zoom in and rebin
zoomed = spio.loadmat('../inputs/CNR phantom data/gen4/results/zoom_in_object_w_scatter_subtracted.mat')['meanview']
newzoom = np.zeros((21,21,60))
for i in np.arange(60):
   newzoom[:,:,i] = zoomed[:,:,2*i] + zoomed[:,:,2*i+1]
#%matplotlib qt
zoomed = newzoom.copy() 
zoomed_slice = zoomed[:,:,23]
plt.imshow(zoomed_slice)
plt.colorbar()


#add patch for ROI of zoomed in image
N = 64 #size of imaging array
res = 0.5 #resolution in mm
x_mm = 8.7
y_mm =  -5
x_pix = N/2 + (x_mm/res)
y_pix = N/2 - (y_mm/res)
width = 5/res
center = x_pix - (width/2), y_pix - (width/2)

r = patches.Rectangle(center,width, width, edgecolor = 'red', facecolor = 'None')

#plot zoom in next to XFET image
%matplotlib inline
plt.figure(dpi = 700)
plt.subplot(1,2,1)
img1 = plt.imshow(XFET_depth1, cmap = 'gray', vmin = 0, vmax = 160)

plt.gca().add_patch(r)
plt.title('Low Dose')
plt.axis('off')

divider1 = make_axes_locatable(plt.gca())
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
cbar1 = plt.colorbar(img1, cax=cax1)
cbar1.ax.yaxis.set_ticks_position('right')  # Position ticks on the left

plt.subplot(1,2,2)
img2 = plt.imshow(zoomed_slice, cmap = 'gray', vmin = 0, vmax = 3000)
plt.title('High Dose')

plt.axis('off')
#colorbar
divider2 = make_axes_locatable(plt.gca())
cax2 = divider2.append_axes("right", size="5%", pad=0.1)
cbar2 = plt.colorbar(img2, cax=cax2)
cbar2.ax.yaxis.set_ticks_position('right')  # Position ticks on the right
cbar2.set_label('Counts')
