# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:49:03 2020

@author: isabe
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
import os
import tifffile as tiff
from skimage import measure
import pymesh
from gmrv_utils.image import load_tiff_from_folder, getAABBImg, idxAABB

tiffPath = './outputData/'
spacing = np.array([0.279911, 0.0751562, 0.0751562])
for file in os.scandir(tiffPath):
    sectionName = file.name
    tifImage = tiff.imread(file)
    mn, mx = getAABBImg(tifImage)
    slices_ = idxAABB(mn, mx)
    imgSizeY = tifImage.shape[1]
    image = np.transpose(tifImage, (2, 1, 0))
    image = np.flip(image, 1)
    verts, faces, normals, _ = measure.marching_cubes_lewiner(image, level=0.5, spacing=spacing)
    bbOrigTransp = np.array([mn[2], imgSizeY - mx[1], mn[0]])
    offset = bbOrigTransp * spacing
    for i in range(len(verts)):
        verts[i] = verts[i] + offset
    pymesh.save_mesh(tiffPath + sectionName + '.obj', verts, faces)
