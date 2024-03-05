#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:30:14 2022
READING in TOPAS DATA FOR MOBY MOUSE
@author: hadleys
"""

#This code was last edited March 2024, for loading and analyzing MOBY mouse runs

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
from scipy.ndimage import zoom
from skimage.segmentation import find_boundaries
from get_CNR import find_CNR_from_rectangleROI

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


#%% LOAD IN MOBY TOPAS raw data
#del Ldata0, Ldata1, Ldata2, Ldata3, Rdata0, Rdata1, Rdata2, Rdata3

#switch directory to one with data
numEBins = 8                      #num of energy bins
detcolum = 46                       #number of detector pixels in a one column
detrow = 120 
spatialDims = [detcolum, detrow]         #num of detector spatial pixels (y,z) 
testindex = 4                    #energy bin for fluorescence energy (4) or surrounding bins (3 and 5)
i = 868*detrow                         #number of object pixels

os.chdir('../../simulations/XFET/moby/moby0/')
NWdata0 = np.empty((i, detcolum)) #detector 1, and so on
SWdata0 = np.empty((i, detcolum)) #detector 2
Ndata0 = np.empty((i, detcolum)) #detector 3
NEdata0 = np.empty((i, detcolum)) #detector 4
SEdata0 = np.empty((i, detcolum)) #detector 5
Sdata0 = np.empty((i, detcolum)) #detector 6
for filename in os.listdir(os.getcwd()):
    if filename[-1] == 'v':   #pick out csvs only
        binnedTracks = np.loadtxt(filename,delimiter=',',usecols=np.linspace(4,numEBins+4,num=numEBins,dtype=int))
        tempdata = binnedTracks[:,testindex] #need to pull out right energy bin first!
        tempdata =np.transpose(np.flip(np.reshape(tempdata, [spatialDims[0],spatialDims[1]]),axis = 1))
        idxstart = int(filename[-8:-4])*detrow
        if filename[16] == '2':
            SWdata0[idxstart:idxstart+detrow,:] = tempdata
        if filename[16] == '3':
            Ndata0[idxstart:idxstart+detrow,:] = tempdata
        if filename[16] == '4':
            NEdata0[idxstart:idxstart+detrow,:] = tempdata      
        if filename[16] == '5':
            SEdata0[idxstart:idxstart+detrow,:] = tempdata   
        if filename[16] == '6':
            Sdata0[idxstart:idxstart+detrow,:] = tempdata   
        elif filename[12] == '_':
            NWdata0[idxstart:idxstart+detrow,:] = tempdata
            
os.chdir('../../simulations/XFET/moby/moby1/')
NWdata1 = np.empty((i, detcolum)) #detector 1, and so on
SWdata1 = np.empty((i, detcolum)) #detector 2
Ndata1 = np.empty((i, detcolum)) #detector 3
NEdata1 = np.empty((i, detcolum)) #detector 4
SEdata1 = np.empty((i, detcolum)) #detector 5
Sdata1 = np.empty((i, detcolum)) #detector 6
for filename in os.listdir(os.getcwd()):
    if filename[-1] == 'v':   #pick out csvs only
        binnedTracks = np.loadtxt(filename,delimiter=',',usecols=np.linspace(4,numEBins+4,num=numEBins,dtype=int))
        tempdata = binnedTracks[:,testindex] #need to pull out right energy bin first!
        tempdata =np.transpose(np.flip(np.reshape(tempdata, [spatialDims[0],spatialDims[1]]),axis = 1))
        idxstart = int(filename[-8:-4])*detrow
        if filename[16] == '2':
            SWdata1[idxstart:idxstart+detrow,:] = tempdata
        if filename[16] == '3':
            Ndata1[idxstart:idxstart+detrow,:] = tempdata
        if filename[16] == '4':
            NEdata1[idxstart:idxstart+detrow,:] = tempdata   
        if filename[16] == '5':
            SEdata1[idxstart:idxstart+detrow,:] = tempdata 
        if filename[16] == '6':
            Sdata1[idxstart:idxstart+detrow,:] = tempdata 
        elif filename[12] == '_':
            NWdata1[idxstart:idxstart+detrow,:] = tempdata
            
os.chdir('../../simulations/XFET/moby/moby2/')
NWdata2 = np.empty((i, detcolum)) #detector 1, and so on
SWdata2 = np.empty((i, detcolum)) #detector 2
Ndata2 = np.empty((i, detcolum)) #detector 3
NEdata2 = np.empty((i, detcolum)) #detector 4
SEdata2 = np.empty((i, detcolum)) #detector 5
Sdata2 = np.empty((i, detcolum)) #detector 6
for filename in os.listdir(os.getcwd()):
    if filename[-1] == 'v':   #pick out csvs only
        binnedTracks = np.loadtxt(filename,delimiter=',',usecols=np.linspace(4,numEBins+4,num=numEBins,dtype=int))
        tempdata = binnedTracks[:,testindex] #need to pull out right energy bin first!
        tempdata =np.transpose(np.flip(np.reshape(tempdata, [spatialDims[0],spatialDims[1]]),axis = 1))
        idxstart = int(filename[-8:-4])*detrow
        if filename[16] == '2':
            SWdata2[idxstart:idxstart+detrow,:] = tempdata
        if filename[16] == '3':
            Ndata2[idxstart:idxstart+detrow,:] = tempdata   
        if filename[16] == '4':
            NEdata2[idxstart:idxstart+detrow,:] = tempdata            
        if filename[16] == '5':
            SEdata2[idxstart:idxstart+detrow,:] = tempdata   
        if filename[16] == '6':
            Sdata2[idxstart:idxstart+detrow,:] = tempdata  
        elif filename[12] == '_':
            NWdata2[idxstart:idxstart+detrow,:] = tempdata
            
os.chdir('../../simulations/XFET/moby/moby3/')
NWdata3 = np.empty((i, detcolum)) #detector 1, and so on
SWdata3 = np.empty((i, detcolum)) #detector 2
Ndata3 = np.empty((i, detcolum)) #detector 3
NEdata3 = np.empty((i, detcolum)) #detector 4
SEdata3 = np.empty((i, detcolum)) #detector 5
Sdata3 = np.empty((i, detcolum)) #detector 6
for filename in os.listdir(os.getcwd()):
    if filename[-1] == 'v':   #pick out csvs only
        binnedTracks = np.loadtxt(filename,delimiter=',',usecols=np.linspace(4,numEBins+4,num=numEBins,dtype=int))
        tempdata = binnedTracks[:,testindex] #need to pull out right energy bin first!
        tempdata =np.transpose(np.flip(np.reshape(tempdata, [spatialDims[0],spatialDims[1]]),axis = 1))
        idxstart = int(filename[-8:-4])*detrow
        if filename[16] == '2':
            SWdata3[idxstart:idxstart+detrow,:] = tempdata
        if filename[16] == '3':
            Ndata3[idxstart:idxstart+detrow,:] = tempdata
        if filename[16] == '4':
            NEdata3[idxstart:idxstart+detrow,:] = tempdata    
        if filename[16] == '5':
            SEdata3[idxstart:idxstart+detrow,:] = tempdata 
        if filename[16] == '6':
            Sdata3[idxstart:idxstart+detrow,:] = tempdata  
        elif filename[12] == '_':
            NWdata3[idxstart:idxstart+detrow,:] = tempdata
      

#%% combine and save arrays

#combine all 4 runs into one:
Sdata = Sdata0 + Sdata1 + Sdata2 + Sdata3
Ndata = Ndata0 + Ndata1 + Ndata2 + Ndata3
SWdata = SWdata0 + SWdata1 + SWdata2 + SWdata3
NWdata = NWdata0 + NWdata1 + NWdata2 + NWdata3
SEdata = SEdata0 + SEdata1 + SEdata2 + SEdata3
NEdata = NEdata0 + NEdata1 + NEdata2 + NEdata3

#save data. 'p' indicates energy bin 5 and 'm' indicates energy bin 3. Neither represent fluroescence energy bin
spio.savemat('../inputs/MOBY/SWdata.mat',{'SWdata':SWdata})
spio.savemat('../inputs/MOBY/NWdata.mat',{'NWdata':NWdata})
spio.savemat('../inputs/MOBY/Ndata.mat',{'Ndata': Ndata})
spio.savemat('../inputs/MOBY/NEdata.mat',{'NEdata':NEdata})
spio.savemat('../inputs/MOBY/SEdata.mat',{'SEdata':SEdata})
spio.savemat('../inputs/MOBY/Sdata.mat',{'Sdata': Sdata})


#%% load in saved datasets
os.chdir('../inputs/MOBY/')

SWdata_p = spio.loadmat('SWdata_p.mat')['SWdata_p']
NWdata_p = spio.loadmat('NWdata_p.mat')['NWdata_p']
Ndata_p = spio.loadmat('Ndata_p.mat')['Ndata_p']
NEdata_p = spio.loadmat('NEdata_p.mat')['NEdata_p']
SEdata_p = spio.loadmat('SEdata_p.mat')['SEdata_p']
Sdata_p = spio.loadmat('Sdata_p.mat')['Sdata_p']

SWdata_m = spio.loadmat('SWdata_m.mat')['SWdata_m']
NWdata_m = spio.loadmat('NWdata_m.mat')['NWdata_m']
Ndata_m = spio.loadmat('Ndata_m.mat')['Ndata_m']
NEdata_m = spio.loadmat('NEdata_m.mat')['NEdata_m']
SEdata_m = spio.loadmat('SEdata_m.mat')['SEdata_m']
Sdata_m = spio.loadmat('Sdata_m.mat')['Sdata_m']

SWdata = spio.loadmat('SWdata.mat')['SWdata']
NWdata = spio.loadmat('NWdata.mat')['NWdata']
Ndata = spio.loadmat('Ndata.mat')['Ndata']
NEdata = spio.loadmat('NEdata.mat')['NEdata']
SEdata = spio.loadmat('SEdata.mat')['SEdata']
Sdata = spio.loadmat('Sdata.mat')['Sdata']


#%% FORM FINAL MOBY IMAGE with scatter correction


xnum = 31  #number of x voxels
ynum = 28 #number of y voxels
znum = 120 #number of original z voxels

#sum of counts for each energy bin
#fluorescence energy bin:
meandata = (np.sum(SWdata, 1) + np.sum(NWdata, 1) + np.sum(Ndata, 1) + np.sum(NEdata, 1) + np.sum(SEdata, 1) + np.sum(Sdata, 1))
#neighboring energy bins
meandata_m = (np.sum(SWdata_m, 1) + np.sum(NWdata_m, 1) + np.sum(Ndata_m, 1) + np.sum(NEdata_m, 1) + np.sum(SEdata_m, 1) + np.sum(Sdata_m, 1))
meandata_p = (np.sum(SWdata_p, 1) + np.sum(NWdata_p, 1) + np.sum(Ndata_p, 1) + np.sum(NEdata_p, 1) + np.sum(SEdata_p, 1) + np.sum(Sdata_p, 1))

#meanscatter image
meanscatter = (meandata_m + meandata_p)/2

#Scatter reshaped to image space
meanview_s = np.flip(np.flip(np.reshape(meanscatter,(ynum, xnum ,znum)), axis = 2), axis = 1)  #switched from flipping 0,1 to 2,1
#smooth scatter
smoothed = scipyim.gaussian_filter(meanview_s, sigma = 1)
#rehsape back to data space
meanscatter_smoothed = np.reshape(np.flip(np.flip(smoothed, 1), 2), (xnum*ynum*znum)) #switched flipping from 1,0 to 1,2. Might need to check
#subtract mean data and smoothed scatter
meandatafinal = np.subtract(meandata, meanscatter_smoothed)
#correct for any negative counts
meandatafinal[meandatafinal < 0] = 0

meanview =np.flip(np.reshape(meandatafinal,(ynum, xnum ,znum)), axis = 2)

%matplotlib qt
plotSlices(meanview, 1)

spio.savemat('../inputs//MOBY/XFET_meanview_scatter_subtracted.mat',{'meanview':meanview})


#resampled to 2-mm z width
newmeanview = np.zeros((28,31,60))
for i in np.arange(60):
   newmeanview[:,:,i] = meanview[:,:,2*i] + meanview[:,:,2*i+1]

plotSlices(newmeanview, 0)
spio.savemat('../inputs/MOBY/XFET_meanview_scatter_subtracted_rebinned_2mmzwidth.mat',{'newmeanview':newmeanview})



#%% Find and plot CNRs using rectangular ROIs

#load in MOBY
moby = np.fromfile("../inputs/MOBY/moby_flip.bin",dtype=np.int16)
moby = np.reshape(moby, [865, 256, 256])
moby_kidney_slice = np.flip(moby[476,:,:], 0)
moby_tumor_slice = np.flip(moby[685, :,:], 0)


#load in CT images (Change path)
CT_kidney_eid = np.fromfile('../inputs/xtomosim/output/spie2024_xfet/MOBY_KIDNEY_EID/srecon_waterBHC_32_float32.bin', dtype = np.float32)
CT_kidney_pcd = np.fromfile('../inputs/xtomosim/output/spie2024_xfet/MOBY_KIDNEY_PCD/srecon_waterBHC_32_float32.bin', dtype = np.float32)
CT_tumor_eid = np.fromfile('../inputs/xtomosim/output/spie2024_xfet/MOBY_TUMOR_EID/srecon_waterBHC_32_float32.bin', dtype = np.float32)
CT_tumor_pcd = np.fromfile('../inputs/xtomosim/output/spie2024_xfet/MOBY_TUMOR_PCD/srecon_waterBHC_32_float32.bin', dtype = np.float32)

CT_kidney_eid = np.reshape(CT_kidney_eid, [32,32])
CT_kidney_pcd = np.reshape(CT_kidney_pcd, [32,32])
CT_tumor_eid = np.reshape(CT_tumor_eid, [32,32])
CT_tumor_pcd = np.reshape(CT_tumor_pcd, [32,32])


#load in XFET images
meanview = spio.loadmat('../inputs/MOBY/XFET_meanview_scatter_subtracted_rebinned_2mmzwidth.mat')['newmeanview']
xfet_kidney_slice = meanview[:,:,27]
xfet_tumor_slice = meanview[:,:,13]

#REGISTER CT AND XFET IMAGES HERE -- apply offsets to match CT to XFET
CT_kidney_eid = CT_kidney_eid[3:31, 0:31]
CT_kidney_pcd = CT_kidney_pcd[3:31, 0:31]
CT_tumor_eid = CT_tumor_eid[3:31, 0:31]
CT_tumor_pcd = CT_tumor_pcd[3:31, 0:31]

#visualize registration
fig, ax = plt.subplots()
ax.imshow(xfet_tumor_slice, cmap='plasma', alpha=0.5)
ax.imshow(CT_tumor_eid, cmap='gray', alpha=0.5)


#FIND CNRS:
kidney1 = 7.5, 7.5, 3,3 #x,y,dx,dy
kidney2 = 18.5, 8.5, 3,3
background_kidney = 21.5, 14.5, 5, 5
tumor = 23.5,9.5, 5, 5
background_tumor = 12.5,15.5, 5, 5

xfet_kidney1_cnr = find_CNR_from_rectangleROI(xfet_kidney_slice, kidney1, background_kidney, 1)
xfet_kidney2_cnr = find_CNR_from_rectangleROI(xfet_kidney_slice, kidney2, background_kidney, 1)
xfet_tumor_cnr = find_CNR_from_rectangleROI(xfet_tumor_slice, tumor, background_tumor, 1)

ct_kidney1_eid_cnr = find_CNR_from_rectangleROI(CT_kidney_eid, kidney1, background_kidney, 1)
ct_kidney2_eid_cnr = find_CNR_from_rectangleROI(CT_kidney_eid, kidney2, background_kidney, 1)
ct_tumor_eid_cnr = find_CNR_from_rectangleROI(CT_tumor_eid, tumor, background_tumor, 1)


ct_kidney1_pcd_cnr = find_CNR_from_rectangleROI(CT_kidney_pcd, kidney1, background_kidney, 1)
ct_kidney2_pcd_cnr = find_CNR_from_rectangleROI(CT_kidney_pcd, kidney2, background_kidney, 1)
ct_tumor_pcd_cnr = find_CNR_from_rectangleROI(CT_tumor_pcd, tumor, background_tumor, 1)


#%% FIGURES

from matplotlib.cm import ScalarMappable

#Showing ROI for kidney------------------------------------------------------------------------------------------
x, y, dx, dy = kidney1
x2, y2, dx2, dy2 = kidney2
x3, y3, dx3, dy3 = tumor
xb, yb, dxb, dyb = background_kidney
xb2, yb2, dxb2, dyb2 = background_tumor

fig, ax =  plt.subplots(1, 1, dpi=300)
fig.patch.set_facecolor('white')  # Set the background color of the entire figure
ax.imshow(CT_kidney_pcd, cmap="bone")
rect0 = patches.Rectangle((x, y), dx, dx, linewidth=2, edgecolor='red', facecolor='none') #kidney 1 ROI
rect1 = patches.Rectangle((x2, y2), dx2, dx2, linewidth=2, edgecolor='red', facecolor='none') #kidney 1 ROI
rect2 = patches.Rectangle((xb, yb), dxb, dyb, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect0)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.set_facecolor('white')
sm = ScalarMappable(cmap="bone")
sm.set_array([])  # Dummy array, not used in this case
#cbar = plt.colorbar(sm, ax=ax)
#cbar.ax.yaxis.set_tick_params(color='black')
#plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
plt.axis('off')
#plt.show()


fig, ax =  plt.subplots(1, 1, dpi=300)
fig.patch.set_facecolor('white')  # Set the background color of the entire figure
ax.imshow(CT_tumor_pcd, cmap="bone")
rect3 = patches.Rectangle((x3, y3), dx3, dx3, linewidth=2, edgecolor='red', facecolor='none') #kidney 1 ROI
rect4 = patches.Rectangle((xb2, yb2), dxb2, dxb2, linewidth=2, edgecolor='red', facecolor='none') #kidney 1 ROI
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.set_facecolor('white')
sm = ScalarMappable(cmap="bone")
sm.set_array([])  # Dummy array, not used in this case
#cbar = plt.colorbar(sm, ax=ax)
#cbar.ax.yaxis.set_tick_params(color='white')
#plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
plt.axis('off')
#plt.show()


# Plot MOBY slices -------------------------------------------
fig, axes = plt.subplots(1, 2)
# Plot the first image
axes[0].imshow(moby_kidney_slice, cmap="bone")
axes[0].axis('off')

# Plot the second image
axes[1].imshow(moby_tumor_slice, cmap="bone")
axes[1].axis('off')
plt.show()

xfet_kidney_slice.astype(np.float32).tofile('xfet_kidney_slice_28x_31y.bin')

#plot three modalities next to each other ---------------------------------------------------------------------------

common_aspect = xfet_kidney_slice.shape[0] / xfet_kidney_slice.shape[0]
fig = plt.figure(dpi=1000, figsize=(6.6, 5))
fig.patch.set_facecolor('black')  # Set the background color of the entire figure

# Subplot 1
plt.subplot(2, 3, 1)
plt.imshow(xfet_kidney_slice, cmap='bone', vmin=0, vmax=120, aspect=common_aspect)
plt.axis('off')

# Subplot 2
plt.subplot(2, 3, 2)
plt.imshow(CT_kidney_eid, cmap='bone', vmin=0, vmax=0.55, aspect=common_aspect)
plt.axis('off')

# Subplot 3
plt.subplot(2, 3, 3)
plt.imshow(CT_kidney_pcd, cmap='bone', vmin=0, vmax=0.55, aspect=common_aspect)
plt.axis('off')

# Subplot 4
plt.subplot(2, 3, 4)
plt.imshow(xfet_tumor_slice, cmap='bone', vmin=0, vmax=120, aspect=common_aspect)
plt.axis('off')
cbar0 = plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.08, shrink=0.8)
cbar0.ax.tick_params(labelsize=10)
cbar0.set_label('Intensity (a.u.)', fontsize=10)  # Add colorbar label
#cbar0.ax..set_tick_params(color='white')

# Subplot 5
plt.subplot(2, 3, 5)
plt.imshow(CT_tumor_eid, cmap='bone', vmin=0, vmax=0.55, aspect=common_aspect)
plt.axis('off')
cbar1 = plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.08, shrink=0.8)
cbar1.ax.tick_params(labelsize=10)
cbar1.set_label(r'Attenuation coefficient (cm$^{-1}$)', fontsize=10)  # Add colorbar label

# Subplot 6
plt.subplot(2, 3, 6)
plt.imshow(CT_tumor_pcd, cmap='bone', vmin=0, vmax=0.55, aspect=common_aspect)
plt.axis('off')
cbar = plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.08, shrink=0.8)
cbar.ax.tick_params(labelsize=10)
cbar.set_label(r'Attenuation coefficient (cm$^{-1}$)', fontsize=10)  # Add colorbar label

plt.tight_layout()
#plt.show()


#bar chart---------------------------------------------------------------------------------------
%matplotlib qt

cat = ['Kidney 1', 'Kidney 2', 'Tumor']
x_pos = np.arange(len(cat))
xfet = [xfet_kidney1_cnr, xfet_kidney2_cnr, xfet_tumor_cnr]
pcct = [ct_kidney1_pcd_cnr, ct_kidney2_pcd_cnr, ct_tumor_pcd_cnr]
eict = [ct_kidney1_eid_cnr, ct_kidney2_eid_cnr, ct_tumor_eid_cnr]

#xfet = [5.65, 6.51]
#pcct = [5.3, 5.46]
#eict = [3.5, 4.21]
fig, ax = plt.subplots(dpi = 500, figsize = [3.5,2.5])
plt.axhline(y=4, color='red', linestyle='--', linewidth=2, label='Rose Criterion')

ax.bar(x_pos-0.2, xfet, 0.2,  color = 'dimgray', edgecolor = 'black', ecolor = 'black', label = 'XFET')
ax.bar(x_pos, pcct, 0.2, color = 'gray', edgecolor = 'black',ecolor = 'black', label = 'PCCT')
ax.bar(x_pos+0.2, eict, 0.2, color = 'lightgray', edgecolor = 'black', ecolor = 'black', label = 'EICT')
ax.set_ylabel('CNR')
ax.set_xticks(x_pos)
ax.set_xticklabels(cat)


plt.tight_layout()
plt.legend()
#plt.show()



#Gold map figure
%matplotlib inline
plt.imshow(moby[:,140,:], cmap = 'bone')
plt.axis('off')
#making activity map
livermask = np.where(moby == 7)
tumormask = np.where(moby == 9)
kidneymask = np.where(moby == 12)
goldmap = np.zeros(np.shape(moby))
goldmap[livermask] = 0.12
goldmap[tumormask] = 0.75
goldmap[kidneymask] = 4

%matplotlib inline
plt.figure(dpi = 1000, figsize = [3.2,4])
plt.subplots_adjust(wspace=0)
plt.subplot(1,2,1)
plt.imshow(moby[:,140,:], cmap = 'bone')
plt.axis('off')
plt.subplot(1,2,2)
img = plt.imshow(goldmap[:,140,:], cmap = 'cividis')
plt.axis('off')
cbar = plt.colorbar(img)
cbar.set_label('Gold concentration in soft tissue (\%)')














