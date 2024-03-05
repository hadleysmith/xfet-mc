#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 12:59:43 2023

@author: hadleys
"""
#be indirectory /topas_sims/code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as spio
from scipy.integrate import simpson
from convert_coordinates import *

N = 512 #size of phantom

#create list of rxyz, ind
#old phantom testing resolution, contrast, beam depth GENERATION1
#rlist = [45, 15, 5, 45, 15, 5, 45, 15, 5, 45, 15, 5,45, 15, 5, 45, 15, 5]  #radius of spehere
#xlist = [449, 409, 371, 307, 221, 171,115,68, 57, 63,103,141,204,291,341,397,444,455]
#ylist = [308, 385, 420, 449, 453, 437,397,324,273,204,127,92,63,59,75,115,188,239]
#zlist = [50,50,50, 460,460,460,50,50,50, 460,460,460 ,50,50,50, 460,460,460]
#ind =   [2, 2,2, 2,2,2,3,3,3,3,3,3,4,4,4,4,4,4]

#new phantom testing contrast, fluoro depth, beam depth NEW GENERATION 2
#xlist = [409,344,279,68,148,228,291,276,261,251,236,221,233,168,103,284,364,444]
#ylist = [385, 330,275,324, 295,266,59,143,227,286,369,453,237,182,127,246,217,188]
rholist = [75,150,225,75,150,225,75,150,225,75,150,225,75,150,225,75,150,225]
philist = [60,60,60,180,180,180,300,300,300,120,120,120,240,240,240,360,360,360]
zlist = [50,50,50,50,50,50,50,50,50,460,460,460,460,460,460,460,460,460]
ind =   [2,2,2,3,3,3,4,4,4,2,2,2,3,3,3,4,4,4]

#new phantom from polar coordinate list:
rxyzlist = []
for i in np.arange(len(zlist)):
    x,y = np.round(pol2cart(rholist[i], philist[i], N/2, N/2))
    rxyzlist.append([30,x,y,zlist[i],ind[i]])


#%%make phantom
phantom = np.zeros((N,N,N))  # initialize
x1, y1, z1 = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
test0 = np.where(((x1-(N/2))**2 + (y1-(N/2))**2) <= (N/2)**2)
phantom[test0] = 1

for rxyz in rxyzlist:
    r, x0, y0, z0, ind = rxyz
    print(ind)
    test= np.where(((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2) < r**2)
    phantom[test] = ind

#%matplotlib qt
#plotSlices(phantom, 0)
%matplotlib inline
plt.figure(dpi = 500)
plt.imshow(phantom[:,:,50])
plt.axis('off')
plt.colorbar()


phantom_sum = np.sum(phantom, axis = 2)
plt.figure(dpi = 500)
plt.imshow(phantom_sum)
plt.axis('off')

phantom.astype(np.uint8).tofile('/home/hadleys/topas/examples/Scoring/XFET_data_sim/paper3/DDC_generation4.bin')



#%%
energyspec = np.fromfile('poly120kVp_forpythonread.txt')
plt.plot(energyspec)


I0 = spio.loadmat('/home/hadleys/Documents/Inverse model paper 1/topas_sims/I0.mat')
I0 = I0['I0']
plt.plot(I0[:,0], I0[:,1])
Inew = I0[92:171,:]
plt.plot(Inew[:,0], Inew[:,1])

#sum of intensities approach
original_sum = np.sum(I0[:,1])
partial_sum = np.sum(Inew[:,1])

#area under approach (not relevant to topas)
#aues = np.trapz(I0[:,1], I0[:,0]) #0.69849
#aues_new = np.trapz(Inew[:,1], Inew[:,0]) #0.10697

Itest = Inew = I0[0:171,:]
New_intensities = Inew[:,1]/aues_new
print(New_intensities)

percent_energy_spec = partial_sum/original_sum #15.3%


#%% gen 4 phantom mimicking the topas version
#one slice only
#rholist = [150,150,150,150,150,150,150,150,150,150,150,150]
philist = [60,120,180,240,300,360,30,90,150,210,270,330]
zlist = [52,52,52,52,52,52, 460,460,460,460,460,460]
ind =   [2,3,4,5,6,7,2,3,4,5,6,7]
N = 512 #size of phantom

#new phantom from polar coordinate list:
rxyzlist = []
for i in np.arange(len(zlist)):
    x,y = np.round(pol2cart(160, philist[i], N/2, N/2))  #160 is rh0
    rxyzlist.append([32,x,y,zlist[i],ind[i]])


phantom = np.zeros((N,N,N))  # initialize
x1, y1, z1 = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
test0 = np.where(((x1-(N/2))**2 + (y1-(N/2))**2) <= (N/2)**2)
phantom[test0] = 1

for rxyz in rxyzlist:
    r, x0, y0, z0, ind = rxyz
    print(ind)
    test= np.where(((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2) < r**2)
    phantom[test] = ind

#%matplotlib qt
#plotSlices(phantom, 0)
%matplotlib qt
plt.figure(dpi = 500)
plt.imshow(phantom[:,:,460], cmap = 'gray')
plt.axis('off')
plt.colorbar()


phantom_sum = np.sum(phantom, axis = 2)
plt.figure(dpi = 500)
plt.imshow(phantom_sum)
plt.axis('off')


#for topas
xlist = []
ylist = []
for i in np.arange(len(zlist)):
    x,y = np.round(pol2cart(160, philist[i], N/2, N/2))
    xlist.append((x-256)*0.00625)   #xlist in cm
    ylist.append((y-256)*0.00625)  #ylist in cm

