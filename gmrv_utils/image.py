# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:30:39 2019

@author: Marcos
@email: marcos.garcia@urjc.es
"""

#  This file is part of DeepSpinePreprocessing
#  Copyright (C) 2021 VG-Lab (Visualization & Graphics Lab), Universidad Rey Juan Carlos
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from glob import glob as glob
import tifffile as tiff
from os.path import join
#import numpy as np
#from os.path import join

#main1
#import matplotlib.pyplot as plt
#from matplotlib import cm

#main2
#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.offline import plot #,iplot, download_plotlyjs, init_notebook_mode, 

#main4 (SVD)
#import numpy.linalg  as linAlg



def load_tiff_from_folder(path ='.', ext='*', filt='*'):
    '''
    Loads all the TIFF files in a given folder (path) with a given 
    extension (ext <without dot>). The user can specify a filter (filt)
    '''
    fileNamesTmp = [f for f in glob(join(path, filt + '.' + ext))]
    tiffList = []
    fileNameList = []
       
    for f in fileNamesTmp:
        try:
            tiffList.append(tiff.imread(f))
            append = True
        except Exception:
            append = False
       
        if append:
            fileNameList.append(f)
        
    return fileNameList, tiffList


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray

def rgb2grayStack(rgb):
    rgbI = rgb.reshape(-1,rgb.shape[-2], rgb.shape[-1])
    gray = rgb2gray(rgbI)
    
    return gray.reshape(rgb.shape[0:len(rgb.shape)-1])


def multiChannel2SingleChannel(mcImg, channel=0):
    imgT = mcImg.reshape(-1, mcImg.shape[-1])
    scImg = imgT[:,channel]

    return scImg.reshape(mcImg.shape[0:len(mcImg.shape)-1])


def getAABBImg (img):
    '''
    mxAABBpixel + 1    
    '''
    nzElments = np.nonzero(img)
    mxAABB = np.max (nzElments, axis = 1)
    mxAABB += np.ones(mxAABB.shape,dtype=np.int)
    mnAABB = np.min (nzElments, axis = 1)
    
    
    return mnAABB, mxAABB

def idxAABB(mn,mx):
    return tuple(slice(i,j) for i,j in zip(mn, mx))

def ellipStruct(shape):
    struct = np.zeros(2 * shape + 1)
    index   = np.indices(2 * shape + 1)
    
    center = np.ones(index.shape)
    center = (center.T*shape).T
    mask   = ((index-center)**2/(center+0.001)**2).sum(axis=0)  <= 1
    struct[mask] = 1
    
    return struct.astype(np.bool)
    

