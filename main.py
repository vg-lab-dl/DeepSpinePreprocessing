#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:31:01 2020

@author: isabel
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

import argparse

import numpy as np
import tifffile as tiff

from os.path import join, basename

from gmrv_utils.generic import GenericObject  # , getAttribNames

from gmrv_utils.u3D import voxelizeMeshList

from seg_utils.dataLoading import loadMetadata, \
    computeMesh2ImageTransforms, \
    loadMeshSet
import os

from configuration.parser import YAMLConfig

# Fichero en el que se guardan los cÃ¡lculos
dataSetFileName = 'dataSet.npz'
computeData = False

# Mallas de espinas y dendritas
extObj = 'obj'
spineFiltObj = '*sp.*'

microscopyDataID = 'org'
microExt = 'tif'
microFilt = 'org'

parser = argparse.ArgumentParser(description='')
parser.add_argument('-cf', '--config_file', required=True, type=str, help='Configuration file path')

args = parser.parse_args()
config_file_path = args.config_file

configuration = YAMLConfig(config_file_path)

spinePath = configuration.get_entry(['Paths', 'spinesPath'])
dendritePath = configuration.get_entry(['Paths', 'dendriticShaftPath'])
metadataPath = configuration.get_entry(['Paths', 'metadataPath'])
tifPath = configuration.get_entry(['Paths', 'rawPath'])
outputPath = configuration.get_entry(['Paths', 'outputPath'])
joinUnconnectedElements = configuration.get_entry(['Options', 'joinUnconnectedElements'])

for entry in os.scandir(spinePath):
    sectionName = entry.name
    print(sectionName)

    if os.path.isfile(os.path.join(outputPath, sectionName) + '.tif'):
        continue
    # =============================================================================
    #  Carga de datos de fichero
    # =============================================================================
    try:
        if not computeData:
            print('Loading from file', flush=True)
            path = os.path.dirname(__file__)
            dataFile = np.load(os.path.join(path, dataSetFileName))
            print('File load')

            dataset = dataFile['dataset'].reshape(-1)[0]

            metadata = dataset.metadata
            linearTransform = metadata.linearTransform
            translation = metadata.translation

            microscopySet = dataset.microscopySets[microscopyDataID]
            microscopyImg = microscopySet.img

            dendritySet = dataset.dendritySets[sectionName]
            dendrityMeshes = dendritySet.meshes
            dendriteLabelImg = dendritySet.labelImg

            spineSet = dataset.spineSets[sectionName]
            spineMeshes = spineSet.meshes
            spineLabelImg = spineSet.labelImg

            dataFile.close()
    except:
        print('ComputeData file not found')
        computeData = True

    if computeData:
        # =============================================================================
        #  Leyendo los metadotos de un text
        # =============================================================================
        print('Carga de metadatos')
        dataset = GenericObject()
        try:
            metadata, microscopyImg = loadMetadata(metadataPath, tifPath, mtFileName=sectionName + '.txt',
                                                   tifFileName=sectionName + '.tif')
        except Exception:
            print('No hay metadatos suficientes')
            continue

        setattr(dataset, 'metadata', metadata)

        # Transformacions que pasan de coordenadas del modelo a coordenadas de imagen
        ############################################################################da
        print('Calculo de transformaciones afines')

        linearTransform, translation = \
            computeMesh2ImageTransforms(metadata.aabbMax,
                                        metadata.aabbMin,
                                        metadata.shape)

        setattr(metadata, 'linearTransform', linearTransform)
        setattr(metadata, 'translation', translation)

        print('############################################################################')
        print(flush=True)
        # print (getAttribNames(metadata,showFunc='valAndType'))

        # =============================================================================
        #  Imagen con las espinas etiquetadas
        # =============================================================================
        # Etiquetas de las espinas
        ############################################################################
        print('Etiquetado de las espinas a partir de los objs')
        spineLabelImg = np.zeros(metadata.shape, dtype=np.int32)

        setattr(dataset, 'spineSets', dict())
        dataset.spineSets[sectionName] = GenericObject()
        spineSet = dataset.spineSets[sectionName]

        spinePath_ = join(spinePath, sectionName)
        setattr(spineSet, 'path', spinePath_)
        setattr(spineSet, 'extObj', extObj)
        setattr(spineSet, 'filtObj', spineFiltObj)

        # Carga de los OBJs de las espinas
        ############################################################################
        print('Carga de los OBJs', spinePath_)

        loadMeshSet(spineSet, spinePath_,
                    filtObj='*', extObj=extObj, labelOffset=1)
        #
        # print("Files loaded:")
        # print(list(map(lambda x: basename(x), spineSet.fileNames)), flush=True)

        spineMeshes = spineSet.meshes

        # Imagen con todas las espainas etiqutadas. (VoxelizaciÃ³n)
        # =============================================================================
        print("Voxelization:")

        voxelizeMeshList(spineLabelImg, spineMeshes,
                         linearTransform, translation,
                         labelOffset=1)

        setattr(spineSet, 'labelImg', spineLabelImg)
        print('############################################################################')
        print(flush=True)

        # =============================================================================
        #  Imagen con la dendrita etiquetadas
        # =============================================================================
        print('Etiquetado de la dendrita')
        dendriteLabelImg = np.zeros(metadata.shape, dtype=np.int32)

        setattr(dataset, 'dendritySets', dict())
        dataset.dendritySets[sectionName] = GenericObject()
        dendritySet = dataset.dendritySets[sectionName]

        setattr(dendritySet, 'path', dendritePath)
        setattr(dendritySet, 'extObj', extObj)
        setattr(dendritySet, 'filtObj', sectionName)

        print('Carga de los OBJs', dendritePath)

        loadMeshSet(dendritySet, dendritePath, filtObj=sectionName, extObj=extObj, labelOffset=0)

        print("Files load:")
        print(list(map(lambda x: basename(x), dendritySet.fileNames)), flush=True)

        dendrityMeshes = dendritySet.meshes

        # Imagen con todas las espainas etiqutadas. (VoxelizaciÃ³n)
        # =============================================================================
        print("Voxelization:")

        voxelizeMeshList(dendriteLabelImg, dendrityMeshes,
                         linearTransform, translation,
                         labelOffset=0)

        setattr(dendritySet, 'labelImg', dendriteLabelImg)

        # Se juntan las componentes separadas:
        # =============================================================================
        print("Voxelization:")

        print('############################################################################')
        print(flush=True)

        if joinUnconnectedElements:
            from seg_utils.dataProcessing import joinUnconnectedComponets, attachElements

            print('Joining components of dendritic shaft ')
            joinUnconnectedComponets(dendritySet, microscopyImg, metadata.spacing,
                                     minRadius=np.array((0, 1, 1)), maxRadius=np.array((2, 6, 6)),
                                     tolFactor=0.1, costFactor=1, connectivity=3,
                                     mediaFilterSize=(2, 8, 8),
                                     useMedianCost=False, fillHoles=True)
            print('Joining components of spines ')

            joinUnconnectedComponets(spineSet, microscopyImg, metadata.spacing,
                                     minRadius=np.array((0, 1, 1)), maxRadius=np.array((2, 3, 3)),
                                     tolFactor=0.1, costFactor=1, connectivity=3,
                                     mediaFilterSize=(2, 8, 8),
                                     useMedianCost=False, fillHoles=True)

            print('Attaching spines to dendritic shaft ')
            reconImg = attachElements(spineSet, dendritySet.reconImg, microscopyImg,
                                      metadata.spacing,
                                      minRadius=np.array((0, 1, 1)), maxRadius=np.array((2, 3, 3)),
                                      tolFactor=0.1, costFactor=2, mediaFilterSize=(1, 4, 4),
                                      fillHoles=True,
                                      useMedianCost=False)
            print('Saving new GT from dendrite ', sectionName)
            reconGT = np.add(reconImg, dendritySet.reconImg)
            reconGT[reconGT > 2] = 2
            reconGT = reconGT.astype(np.uint8)
            print(reconGT.dtype)
            tiff.imsave(outputPath + sectionName + '.tif', reconGT)
        else:
            spinesBool = (spineSet.labelImg != 0) * 2
            outputImg = np.add(dendritySet.labelImg, spinesBool)
            outputImg[outputImg > 2] = 2
            tiff.imsave(outputPath + sectionName + '.tif', outputImg)
