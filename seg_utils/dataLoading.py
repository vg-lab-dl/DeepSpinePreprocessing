63532  # -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:24:49 2019

@author: URJC
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

import sys

import numpy as np

from os.path import join, basename, splitext, isfile
# from glob import glob as glob

import tifffile as tiff

from gmrv_utils.generic import GenericObject, isIterable
from gmrv_utils.u3D import computeAABB, computeAABBFromList, load3DFileSet
from gmrv_utils.image import load_tiff_from_folder, multiChannel2SingleChannel


# =============================================================================
# !todo: Por ahora solo funciona con useTextFile=True
# =============================================================================
def loadMetadata(path, tifPath,
                 mtFileName='metadata.data',
                 tifFileName='org.tif',
                 useTexFile=True):
    assert useTexFile

    metaData = GenericObject()
    spacing = np.array([0.279911, 0.0751562, 0.0751562])

    tiffFileName = join(tifPath, tifFileName)
    if not isfile(tiffFileName):
        raise FileNotFoundError(
            '''File path {} does not exist.'''.format(tiffFileName))

    with tiff.TiffFile(tiffFileName) as tif:
        Sz = len(tif.pages)
        p0 = tif.pages[0]
        Sx, Sy = p0.shape
        bitspersample = p0.bitspersample
        dtype = p0.dtype

        setattr(metaData, 'Sx', Sx)
        setattr(metaData, 'Sy', Sy)
        setattr(metaData, 'Sz', Sz)
        setattr(metaData, 'bitspersample', bitspersample)
        setattr(metaData, 'dtype', dtype)

    shape = np.array([metaData.Sz, metaData.Sy, metaData.Sx])
    setattr(metaData, 'shape', shape)

    mdFileName = join(path, mtFileName)
    if not isfile(mdFileName):
        raise FileNotFoundError(
            'File path {} does not exist.'.format(mdFileName))

    with open(mdFileName, newline='\n') as fl:
        for line in fl:
            var, val = line.split('=')
            val = float(val)
            setattr(metaData, var, val)

    try:
        aabbMax = np.array([metaData.ExtMax2, metaData.ExtMax1, metaData.ExtMax0])
        aabbMin = np.array([metaData.ExtMin2, metaData.ExtMin1, metaData.ExtMin0])
    except AssertionError as error:
        print(error)
        return 0

    setattr(metaData, 'spacing', spacing)
    setattr(metaData, 'aabbMax', aabbMax)
    setattr(metaData, 'aabbMin', aabbMin)

    return metaData, tiff.imread(tiffFileName)


# =============================================================================
# =============================================================================

def computeMesh2ImageTransforms(aabbMax, aabbMin, shape, order=(2, 1, 0)):
    swapMat = np.eye(3)[np.array(order)]

    aabbSize = aabbMax - aabbMin
    aabbSize += aabbSize == 0  # solventa el problema del rango 0 cuando el triangulo
    # esta alineado con el eje
    scaleFactor = (shape - [1, 1, 1]) / aabbSize
    linearTransform = swapMat @ np.diag(scaleFactor)
    translation = [0.5, 0.5, 0.5] - (aabbMin * scaleFactor)

    return linearTransform, translation


# =============================================================================
# =============================================================================

def loadMeshSet(meshSet, meshPath,
                filtObj='*', extObj='obj', labelOffset=0):
    meshes, fileNameList = load3DFileSet(meshPath,
                                         filt=filtObj, ext=extObj)

    meshNames = list(map(lambda x: splitext(basename(x))[0], fileNameList))

    meshLabels = [i + labelOffset for i in range(len(fileNameList))]
    setattr(meshSet, 'names', meshNames)
    setattr(meshSet, 'fileNames', fileNameList)
    setattr(meshSet, 'labels', meshLabels)
    setattr(meshSet, 'meshes', meshes)

    # Calculo de AABB
    mx, mn = computeAABBFromList(meshes)

    setattr(meshSet, 'aabbMax', mx)
    setattr(meshSet, 'aabbMin', mn)
    print('Spines AABB,\n max:', mx, "\nmin", mn)


# =============================================================================
# =============================================================================


def loadTiffStacks(path, *args, ext='tif'):
    '''
       Args:
           fileNames
           multichannel2Singlechannel
           transpose
       Args:
           files: una array con las varialbles anteriores
    '''

    if len(args) == 3:
        fileNames = args[0]
        multichannel2singlechannel = args[1]
        transpose = args[2]

        if not isIterable(fileNames):
            fileNames = [fileNames]

        l = len(fileNames)

        if not isIterable(multichannel2singlechannel):
            multichannel2singlechannel = [multichannel2singlechannel] * l

        if not isIterable(transpose):
            transpose = [transpose] * l

        files2Load = zip(fileNames, multichannel2singlechannel, transpose)

    elif len(args) == 1:
        files2Load = args[0]

    data = dict()

    for filt, multiChannel, transpose in files2Load:
        _, img = load_tiff_from_folder(
            path=path, filt=filt, ext=ext)

        img = img[0]
        if multiChannel:
            img = multiChannel2SingleChannel(img)

        if transpose:
            img = np.transpose(img, (0, 2, 1))

        data[filt] = img

    return data

# =============================================================================
#  Leyendo los metadotos de los tiffs (por desgracia falatan datos)
# =============================================================================
##https://github.com/blink1073/tifffile
##https://pypi.org/project/tifffile/
# path = './data/if6.1.2-3enero/'
# fileName = join(path+'org.tif')
# tif=tiff.TiffFile(fileName)
#
##pintamos las cosas que tiene el tiff
# print (getAttribNames(tif))
##['byteorder', 'filehandle', 'filename', 'flags', #'fstat', 'imagej_metadata',
## 'is_bigtiff', 'is_lsm', 'is_mdgel', 'is_movie', 'isnative', 'offsetformat', 
##'offsetsize','pages', 'series', 'tagformat1', 'tagformat2', 'tagnoformat', 
##'tagnosize', 'tagsize']
##tif.flags # imagj
##tif.byteorder # string '>'
##tif.imagej_metadata['spacing'] # spacing Z?, min, max?, unidades = micras
##tif.fstat # datos del fichero, creación, modificación...
#
# print ("Pages len:", )
# print (getAttribNames(tif.pages[0]))
##['axes', 'bitspersample', 'compression', 'databytecounts', 'dataoffsets', 
## 'description', 'description1', 'dtype', 'extrasamples', 'fillorder', 'flags', 
## 'imagedepth', 'imagelength', 'imagewidth', 'index', 'is_andor',
## 'is_chroma_subsampled', 'is_contiguous', 'is_epics', 'is_fei', 'is_final',
## 'is_fluoview', 'is_geotiff', 'is_imagej', 'is_lsm', 'is_mdgel', 'is_mediacy',
## 'is_memmappable', 'is_metaseries', 'is_micromanager', 'is_nih', 'is_ome', 
## 'is_pilatus', 'is_qptiff', 'is_reduced', 'is_scanimage', 'is_scn', 'is_sem', 
## 'is_sgi', 'is_stk', 'is_svs', 'is_tiled', 'is_tvips', 'is_vista', 'keyframe',
## 'ndim', 'offset', 'offsets_bytecounts', 'parent', 'photometric', 
## 'planarconfig', 'predictor', 'rowsperstrip', 'sampleformat', 'samplesperpixel', 
## 'shape', 'size', 'software', 'tags', 'tiledepth', 'tilelength', 'tilewidth']
# print (getAttribNames(tif.pages[0],showFunc='valAndType'))
# print (tif.pages[0].tags)
