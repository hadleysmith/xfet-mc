#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:14:45 2024

@author: hadleys
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from convert_coordinates import *
import matplotlib.patches as patches


def find_CNR_from_rectangleROI(image, signal, background, vis):  #x and y are sometimes not integers, for viewing patch
    #pull out values:
    x, y, dx, dy = signal
    xb, yb, dxb, dyb = background
    
    #find rois
    signal_roi = image[math.ceil(y):math.ceil(y+dy), math.ceil(x):math.ceil(x+dx)]
    background_roi = image[math.ceil(yb):math.ceil(yb+dyb), math.ceil(xb):math.ceil(xb+dxb)]
    
    #get average and standard deviation of counts:
    mean_signal = np.mean(signal_roi)
    mean_background = np.mean(background_roi)
    stdev_background = np.nanstd(background_roi)

    #compute CNR:
    CNR = (mean_signal - mean_background)/stdev_background
    
    #visualize if desired
    if vis == 1:
        fig, ax = plt.subplots(1,1)
        plt.imshow(image, cmap = 'bone')
        rect0 = patches.Rectangle((x, y), dx, dx, linewidth=2, edgecolor='red', facecolor='none')
        rect1 = patches.Rectangle((xb, yb), dxb, dyb, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect0)
        ax.add_patch(rect1)
        #plt.axis('off')
        plt.colorbar()
        plt.show()
    
    return CNR


    
#the following is specific to DDC phantom studied here:
    
def find_DDC_CNR_from_circleROI(image, vis): #"signal" and "background" include coordinates and radii. VIs is integer for visualizing
   
    N = len(image)  #image size or resolution
    if N == 64:
        rad = 161   #radius of slice that contains each sphere
    elif N == 256:
        rad = 165
    
    #cylindrical coordinates for each ROI
    conversion = N/512
    r1 = rad*conversion                 #N/512 is a conversion factor
    rholist = [r1,r1,r1,r1,r1,r1]        #radius
    philist = [30,90,150,210,270,330]       #this are specific angles for DDC phantom
    Gen4r = 25                              #radius of each sphere

    #make new list with cartesian coordinates
    rxyzlist = []
    for i in np.arange(len(rholist)):
        x,y = np.round(pol2cart(rholist[i], philist[i], N/2, N/2))  #N/2 is offset
        y = y-0.5
        rxyzlist.append([np.round(Gen4r*conversion),x,y])   #30* conversion is radius of each sphere
    
    #set up ROI finding
    x1, y1 = np.meshgrid(np.arange(N), np.arange(N))
    mean_roi = []
    stdev_roi = []
    
    #loop through and find Mean and STdev for ROI
    for rxyz in rxyzlist:
        r, x0, y0 = rxyz
        test = np.where((x1 - x0)**2 + (y1 - y0)**2  < r**2)
        test2 = image[test]
        mean_roi.append(np.mean(test2))  
        stdev_roi.append(np.std(test2)) 

    #find bgk roi
    xc, yc = N/2, N/2
    r2 = (Gen4r*conversion)
    rnew = 2*r2
    test = np.where((x1 - xc)**2 + (y1 - yc)**2  < (rnew)**2)  #i increased the radius of this roi to twice 
    test3 = image[test]
    bkg_mean_roi = np.mean(test3)#does this sum over everything?
    bkg_stdev_roi = np.std(test3) #does this sum over everything?
    rxyzlist.append([rnew, xc, yc])
    
    #Visualize if requested
    if vis == 1:
#visualize ROI placement:
        for i in np.arange(len(rxyzlist)):
            r,x,y = rxyzlist[i]
            center = x,y
            c = patches.Circle(center,radius = r, edgecolor = 'yellow', facecolor = 'None')
            plt.imshow(image, cmap = 'bone')
            plt.gca().add_patch(c)
            plt.axis('off')
    CNR = []
    for i in np.arange(len(mean_roi)):
        CNR.append((mean_roi[i] - bkg_mean_roi)/bkg_stdev_roi)
    
    CNR = np.array(CNR)
    #return list of CNR values ranging from high gold to low
    return CNR


def find_DDC_CNR_from_circleROI_XFET(image, slicedepth, vis): #"signal" and "background" include coordinates and radii. VIs is integer for visualizing
   
    N = len(image)  #image size or resolution
    rad = 162  #only one radius for one XFET resolution
    
    #cylindrical coordinates for each ROI
    conversion = N/512
    r1 = rad*conversion                 #N/512 is a conversion factor
    rholist = [r1,r1,r1,r1,r1,r1]        #radius
    Gen4r = 25                              #radius of each sphere

    if slicedepth == "shallow":
        philist = [30,90,150,210,270,330]
    if slicedepth == "deep":
        philist = [360,60,120,180,240,300] #angle

    #make new list with cartesian coordinates
    rxyzlist = []
    for i in np.arange(len(rholist)):
        x,y = np.round(pol2cart(rholist[i], philist[i], N/2, N/2))  #N/2 is offset
        y = y-0.5
        rxyzlist.append([np.round(Gen4r*conversion),x,y])   #30* conversion is radius of each sphere
    
    #set up ROI finding
    x1, y1 = np.meshgrid(np.arange(N), np.arange(N))
    mean_roi = []
    stdev_roi = []
    
    #loop through and find Mean and STdev for ROI
    for rxyz in rxyzlist:
        r, x0, y0 = rxyz
        test = np.where((x1 - x0)**2 + (y1 - y0)**2  < r**2)
        test2 = image[test]
        mean_roi.append(np.mean(test2))  
        stdev_roi.append(np.std(test2)) 

    #find bgk roi
    xc, yc = N/2, N/2
    r2 = (Gen4r*conversion)
    rnew = 2*r2
    test = np.where((x1 - xc)**2 + (y1 - yc)**2  < (rnew)**2)  #i increased the radius of this roi to twice 
    test3 = image[test]
    bkg_mean_roi = np.mean(test3)#does this sum over everything?
    bkg_stdev_roi = np.std(test3) #does this sum over everything?
    rxyzlist.append([rnew, xc, yc])
    
    #Visualize if requested
    if vis == 1:
#visualize ROI placement:
        for i in np.arange(len(rxyzlist)):
            r,x,y = rxyzlist[i]
            center = x,y
            c = patches.Circle(center,radius = r, edgecolor = 'yellow', facecolor = 'None')
            plt.imshow(image, cmap = 'bone')
            plt.gca().add_patch(c)
            plt.axis('off')
    CNR = []
    for i in np.arange(len(mean_roi)):
        CNR.append((mean_roi[i] - bkg_mean_roi)/bkg_stdev_roi)
    
    CNR = np.array(CNR)
    #return list of CNR values ranging from high gold to low
    return CNR