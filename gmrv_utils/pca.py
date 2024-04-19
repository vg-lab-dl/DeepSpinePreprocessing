# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:30:07 2019

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
import numpy.linalg  as la

#toDo: cambiar el código por el de abajo
def compute_pca (sysMat,normalize=False):
    '''
    PCA from scratch
    sysMat data in columns
    '''
   
    if normalize:
        m = np.mean(sysMat,axis=0)
        sysMat-=m
    
    #!!!!Todo: Optimizar esta traspuesta
    sysMat=sysMat.T

    # A^tA = WE^2W^t
    # T = AW = UEW^tW = UE
    covMat = sysMat.T@sysMat
    u, s, vh = la.svd(covMat)
    
    t = sysMat@(u)
    
    return (t,vh,m) if normalize else (t,vh)


#Todo: https://simplyml.com/benchmarking-the-singular-value-decomposition/
#https://stackoverflow.com/questions/50358310/how-does-numpy-linalg-eigh-vs-numpy-linalg-svd
#PCA Rutine?
#arr = np.memmap('corr_memmap.dat', dtype='float32', mode='w+', shape=(N,N)) 
#dot(z0, z0.T, out=arr)
def compute_pca_ok (sysMat,normalize=False):
    '''
    PCA from scratch
    sysMat data in columns
    '''
    

    
    if normalize:
        means = np.mean(sysMat,axis=1)
        sysMat -= means
    
    if (sysMat.shape[0]>sysMat.shape[1]):
        covMat = sysMat.T@sysMat
        vhh, s2, vh = la.svd(covMat)
            
        s = np.sqrt(s2)
        sinv = 1/s
        basisChange = sysMat@(vhh*sinv) # U*S^{-1}
    
        charcteristics = vh*s[:,np.newaxis] # S*VH
    else:
        covMat = sysMat@sysMat.T
        u, s2, uh = la.svd(covMat)
        basisChange = u
        charcteristics = uh@sysMat 
        
    
    
    
    return (basisChange, charcteristics, means) \
                if normalize else (basisChange,charcteristics)

def compute_pca_ok_v2 (sysMat,normalize=False):
    '''
    PCA from scratch
    sysMat data in columns
    '''
    
    def sort(s,u):
        s = np.abs(s)
        sort = np.argsort (s)[::-1]
        u = u[:,sort]
        s = s[sort]
        return s, u
        
        
        
    #sysMat = u@s@vh
    #sysMa.T@sysMat = v@s2@vh
    #sysMat@sysMa.T = u@s2@uh
    #basisChange = u
    #charcteristics = s@vh
    if normalize:
        means = np.mean(sysMat,axis=1)
        sysMat -= means
    
    if (sysMat.shape[0]>sysMat.shape[1]):
        covMat = sysMat.T@sysMat
        s2,v = la.eigh(covMat) #sysMat =(v*s2)@v.T   
        s2, v = sort(s2,v)
    
        s = np.sqrt(s2)
        sinv = 1/s
        
        basisChange = sysMat@(v*sinv) # U*S^{-1}
        charcteristics = (v*s).T # S*VT
    
    else:
        covMat = sysMat@sysMat.T
        s2, u = la.eigh(covMat) #sysMat =(u*s2)@u.T
        s2, u = sort(s2,u)        
        
        basisChange = u
        charcteristics = u.T@sysMat 
    
    
    return (basisChange, charcteristics, means) \
                if normalize else (basisChange,charcteristics)


