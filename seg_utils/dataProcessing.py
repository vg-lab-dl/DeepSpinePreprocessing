# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:08:44 2019

@author: Marcos
ToDos:
1. Probar con un filtro gausiano en lugar de uno de medianas
2. No me convence calcular las AABB de los datos aquí
3. Se debe cambiar el flood. Se puede cambiar la librería scikit-image facilmente
4. AStar con el filtrado de medias!
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
from skimage import measure as ms
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

from skimage.segmentation import flood
from skimage.segmentation import find_boundaries
# from skimage.filters import median

from gmrv_utils.AStar import labeledImgAStar
from gmrv_utils.image import getAABBImg, idxAABB, ellipStruct

from gmrv_utils.cmpPlot import CmpPlot


def joinUnconnectedComponets(dataset, microImg, spcng,
                             connectivity=3,
                             costFactor=1,
                             tolFactor=0.0,
                             mediaFilterSize=(2, 8, 8),
                             fillHoles=True,
                             useMedianCost=False, minRadius=None, maxRadius=None):
    #    if fillHoles:
    #        fillHolesStruc = np.array(
    #                [np.zeros((3,3)),np.ones((3,3)), np.zeros((3,3))])

    labeledImg = dataset.labelImg

    lImgAABBmin, lImgAABBmax = getAABBImg(labeledImg)
    slices = idxAABB(lImgAABBmin, lImgAABBmax)

    # imagen etiqutada
    subLabeledImg = labeledImg[slices]

    # datos del microscopio
    subMicroImg = microImg[slices].astype(np.float)
    subMicroImg /= subMicroImg.max()

    # datos fitrados
    subFiltMicroImg = median_filter(subMicroImg, size=mediaFilterSize)

    reconList = []
    labelAABB = []
    for i in dataset.labels:
        if not np.any(subLabeledImg == i + 1):
            continue
        mn, mx = getAABBImg(subLabeledImg == i + 1)
        slices = idxAABB(mn, mx)
        labelAABB.append((lImgAABBmin + mn, lImgAABBmin + mx))

        subLabelImgBool = (subLabeledImg[slices] == i + 1).astype(np.ubyte)
        subLabelImg = ms.label(subLabelImgBool, background=0,
                               connectivity=connectivity)
        auxImg = np.copy(subLabelImg)
        subImg = subMicroImg[slices]
        subPathImgBool = np.zeros(subLabelImg.shape, dtype=np.bool)
        subMinImgBool = np.zeros(subLabelImg.shape, dtype=np.bool)
        subMaxImgBool = np.ones(subLabelImg.shape, dtype=np.bool)
        nLabels = subLabelImg.max() + 1
        # Todo: vectorización
        # https://towardsdatascience.com/data-science-with-python-turn-your-conditional-loops-to-numpy-vectors-9484ff9c622e
        nonZero = [np.count_nonzero(subLabelImg == ii) for ii in range(1, nLabels)]
        order = np.argsort(nonZero)[::-1]  # ordenando por número de pixels, de mayor a menor tam
        order += np.ones(order.shape, dtype=np.int)

        mImg = subFiltMicroImg[slices]
        if useMedianCost:
            def cost(idx1, idx2):
                return (1 - mImg[idx2]) * costFactor
        else:
            def cost(idx1, idx2):
                return (1 - subImg[idx2]) * costFactor

        pathList = []
        for l in order[1:]:
            aStarImg = find_boundaries(subLabelImg == l, connectivity=3,
                                       mode='inner').astype(np.int8)
            aStarImg[(subLabelImg != 0) & (subLabelImg != l)] = 2
            aStarImg[(aStarImg != 1) & (subLabelImg == l)] = 3
            aStarImg[(subLabeledImg[slices] != i + 1) & (subLabeledImg[slices] != 0)] = 3

            p = labeledImgAStar(aStarImg, hFunc='distanceField',
                                costFunc=cost, spacing=spcng)
            #            p,os,cs = labeledImgAStar(aStarImg, hFunc = 'distanceField',
            #                                      costFunc = cost,
            #                                      retDebugInfo=True,
            #                                      spacing = spcng)
            #            aStarImg[tuple(zip(*p))]=4
            if p is not None:
                dest = p[0]
                subLabelImg[tuple(zip(*p))] = subLabelImg[dest]
                subLabelImg[subLabelImg == l] = subLabelImg[dest]
                auxImg[tuple(zip(*p))] = nLabels
                # subLabelImg = ms.label(subLabelImg!=0, background=0,
                #                        connectivity=connectivity)
                subPathImgBool[tuple(zip(*p))] = True  # sacar de aquí y meter en otro bucle
                pathList.append(p)

        if minRadius is not None:
            struct = ellipStruct(minRadius)
            subMinImgBool = binary_dilation(subPathImgBool, structure=struct)

        if maxRadius is not None:
            struct = ellipStruct(maxRadius)
            subMaxImgBool = binary_dilation(subPathImgBool, structure=struct)
        # mImg[(subLabelImgBool)]=0
        mImg[(auxImg != 0) & (auxImg != nLabels)] = 0
        for p in pathList:
            r = (subMinImgBool) & np.logical_not(subLabelImgBool)

            for pi in p:
                tol = mImg[pi] * tolFactor
                r += flood(mImg, pi, tolerance=tol) & (subMaxImgBool)

            if fillHoles:
                r = binary_fill_holes(r)  # structure = fillHolesStruc)

            #            reconList.append((r,p, i, l, tolFactor))
            print('Spine ', i, 'Path ', p)
            reconList.append((r, p, i, l))

    #    print("Fin de bucle",flush=True)

    reconImg = np.copy(dataset.labelImg)
    for r, _, i, _ in reconList:
        if i not in dataset.labels:
            continue
        idx = dataset.labels.index(i)
        mn, mx = labelAABB[idx]
        slices = idxAABB(mn, mx)
        print(slices)
        print(r.shape)
        print(i)
        try:
            (reconImg[slices])[r] = i + 1
        except:
            print('dimensions didnt match. Removing label ', i)
            dataset.labels.remove(i)

    setattr(dataset, 'reconImg', reconImg)
    setattr(dataset, 'labelAABB', labelAABB)
    setattr(dataset, 'reconList', reconList)
    setattr(dataset, 'imgAABBmin', lImgAABBmin)
    setattr(dataset, 'imgAABBmax', lImgAABBmax)


def attachElements(dataset, img2Attach, microImg, spcng,
                   costFactor=10, tolFactor=0.0,
                   mediaFilterSize=(2, 8, 8), fillHoles=False,
                   useMedianCost=False, minRadius=None, maxRadius=None):
    cp = []

    #    if fillHoles:
    #        fillHolesStruc = np.array(
    #                [np.zeros((3,3)),np.ones((3,3)), np.zeros((3,3))])

    labeledImg = dataset.reconImg  # espinas

    lImgMn, lImgMx = getAABBImg(labeledImg)  # espinas
    aImgMn, aImgMx = getAABBImg(img2Attach)  # dendrita
    mn = np.min((lImgMn, aImgMn), axis=0)
    mx = np.max((lImgMx, aImgMx), axis=0)

    slices = idxAABB(mn, mx)

    # imagen etiqutada
    subLabeledImg = labeledImg[slices]  # subimg espinas
    subLabeledImgBool = subLabeledImg != 0
    subImg2AttachBool = img2Attach[slices] != 0
    subCompleteImgBool = (subImg2AttachBool) | (subLabeledImgBool)
    subPathImgBool = np.ones(subLabeledImgBool.shape, dtype=np.bool)
    subMinImgBool = np.zeros(subLabeledImgBool.shape, dtype=np.bool)
    subMaxImgBool = np.ones(subLabeledImgBool.shape, dtype=np.bool)

    # datos del microscopio
    subMicroImg = microImg[slices].astype(np.float)
    subMicroImg /= subMicroImg.max()

    # datos fitrados
    #    subFiltMicroImg = \
    #        (subMicroImg * np.iinfo(np.uint16).max).astype(np.uint16)
    #    subFiltMicroImg = median_filter(subFiltMicroImg,size=mediaFilterSize)
    subFiltMicroImg = median_filter(subMicroImg, size=mediaFilterSize)  # quitar ruido img microscopio

    cp.append(CmpPlot([subLabeledImg,
                       subLabeledImgBool,
                       subImg2AttachBool,
                       subCompleteImgBool,
                       subMicroImg,
                       subFiltMicroImg], 3))
    #    print (subPathImgBool.max())

    if useMedianCost:
        def cost(idx1, idx2):
            return (1 - subFiltMicroImg[idx2]) * costFactor
    else:
        def cost(idx1, idx2):
            return (1 - subMicroImg[idx2]) * costFactor

    reconList = list()
    for i in dataset.labels:
        subLabelImgBool = subLabeledImg == i + 1
        nLabels = ms.label((subLabelImgBool) | (subImg2AttachBool),
                           background=0, connectivity=3).max()
        print('Processing spine ', i, 'from ', len(dataset.labels))
        if nLabels < 2:
            #               print ("Sigue ", i)
            continue

        if nLabels > 2:
            print('Espina ', i, ' + etiquetas ', nLabels)
            continue
            raise AssertionError("There are more than 2 componets to attach")

        #        print ("Entra ", i)

        aStarImg = find_boundaries(subLabelImgBool, connectivity=3,
                                   mode='inner').astype(np.int8)
        aStarImg[subImg2AttachBool] = 2
        aStarImg[(aStarImg != 1) & (subLabeledImgBool)] = 3

        p = labeledImgAStar(aStarImg,
                            hFunc='distanceField', costFunc=cost,
                            spacing=spcng)

        print(p)
        subPathImgBool[tuple(zip(*p))] = False

        ######
        #        r = np.logical_not(subPathImgBool)
        #        print(np.count_nonzero(r))
        #####

        if minRadius is not None:
            subMinImgBool = np.logical_not(subPathImgBool)
            struct = ellipStruct(minRadius)
            subMinImgBool = binary_dilation(subMinImgBool, structure=struct)

        if maxRadius is not None:
            subMaxImgBool = np.logical_not(subPathImgBool)
            struct = ellipStruct(maxRadius)
            subMaxImgBool = binary_dilation(subMaxImgBool, structure=struct)

        mImg = np.copy(subFiltMicroImg)
        #        mImg[(subCompleteImgBool) & (subPathImgBool)] = 0
        mImg[(subCompleteImgBool)] = 0

        subPathImgBool[tuple(zip(*p))] = True

        r = (subMinImgBool) & np.logical_not(subCompleteImgBool)
        for pi in p:
            if mImg[pi] != 0:  # aquí se podría establecer un min de intensidad
                tol = mImg[pi] * tolFactor
                r += (flood(mImg, pi, tolerance=tol)) & (subMaxImgBool)

        if fillHoles:
            r = binary_fill_holes(r)  # ,structure = fillHolesStruc)

        # cp.append(CmpPlot([aStarImg,1 - subPathImgBool,mImg,r]))

        reconList.append((r, p, i, -1))

    dataset.reconList += reconList
    reconImg = dataset.reconImg
    for r, _, i, _ in reconList:
        (reconImg[slices])[r] = i + 1

    # print(np.count_nonzero(r))

    return reconImg
