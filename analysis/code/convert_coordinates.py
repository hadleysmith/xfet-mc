#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:12:48 2023

@author: hadleys
"""
import numpy as np

def pol2cart(rho, phi, xoffset, yoffset):
    x = xoffset + (rho * np.cos(np.deg2rad(phi)))
    y = yoffset + (rho * np.sin(np.deg2rad(phi)))
    return(x, y)

