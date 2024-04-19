# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:21:53 2019

@author: Marcos Garcia 
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
from scipy.ndimage import binary_fill_holes

from os.path import join, basename, splitext, isfile
from glob import glob as glob

import pyassimp

import tifffile as tiff


def doAABBIntersec (bb1,bb2):
    bb = AABBIntersection (bb1, bb2)
    return np.all(bb[3:6]-bb[:3]>0)
    
#    dims1 = bb1[3:6] - bb1[:3]
#    dims2 = bb2[3:6] - bb2[:3]
#          
#    print ((np.abs(bb1[:3]-bb2[:3]) * 2) - (dims1 + dims2))
#    print (((np.abs(bb1[:3]-bb2[:3]) * 2) - (dims1 + dims2)) <0)
#    return np.all( ((np.abs(bb1[:3]-bb2[:3]) * 2) - (dims1 + dims2)) <0  )
##    return (((np.abs(bb1[0]-bb2[0]) * 2) < (dims1[0] + dims2[0])) and
##            ((np.abs(bb1[1]-bb2[1]) * 2) < (dims1[1] + dims2[1])) and
##            ((np.abs(bb1[2]-bb2[2]) * 2) < (dims1[2] + dims2[2])))

def AABBIntersection (bb1, bb2):
    return np.concatenate((
            np.max([bb1[:3],bb2[:3]], axis = 0),
            np.min([bb1[3:6],bb2[3:6]], axis = 0)
    ))
    
def AABBUnion (bb1, bb2):
    return np.concatenate((
            np.min([bb1[:3],bb2[:3]], axis = 0),
            np.max([bb1[3:6],bb2[3:6]], axis = 0)
            ))

def computeAABB (mesh):
    p = mesh['points']
    mx = np.max(p, axis = 0)
    mn = np.min(p, axis = 0)
    return mx,mn

def computeAABBFromList (meshList):
    #Calculo de AABB
    mx = np.array([-np.inf]*3)
    mn = np.array([np.inf]*3)
    for m in meshList:
        mx_tmp,mn_tmp = computeAABB(m)
        mx = np.max([mx_tmp, mx], axis = 0)
        mn = np.min([mn_tmp, mn], axis = 0)
    
    return mx,mn

def compute_mesh_area(mesh):
    triangles = mesh['triangles']
    points = mesh['points']
    
    area = 0
    for tri in triangles:
        p0 = points[tri[0]]
        p1 = points[tri[1]]
        p2 = points[tri[2]]
    
        area += la.norm(np.cross(p1-p0, p2-p0))*0.5
        
    return area

def compute_mesh_volume(mesh):
    triangles = mesh['triangles']
    points = mesh['points']
    
    vol = 0
    for tri in triangles:
        p0 = points[tri[0]]
        p1 = points[tri[1]]
        p2 = points[tri[2]]
        
        m = np.array([p0,p1,p2])
        
    
        vol += la.det(m)*0.16666666666666666
        
    return vol




def compute_position_map(displacementMap, mn=-1,mx=1):
    ''' 
    Transforms the node displacement map into a node position map
    mn = -1 minimum grid undisplaced value
    mx = 1 maximum grid undisplaced value
    '''

    eltosx, eltosy, eltosz = displacementMap.shape

    idxx = np.linspace(mn,mx, eltosx)
    idxy = np.linspace(mn,mx, eltosy)
    XX, YY = np.meshgrid(idxx,idxy)
    #    XX, YY = np.mgrid[-1:1+2/64:2/64, -1:1+2/64:2/64]
    
    positionMap = np.array(displacementMap, copy=True);      
    positionMap[:,:,0] += XX
    positionMap[:,:,1] += YY
            
    return positionMap

#!todo: Don't use Delaunay
from scipy.spatial import Delaunay #temporal
def compute_grid_mesh_topology(mn=-1,mx=1, eltosx = 65,eltosy = 65):
    '''
    Computes de mesh topology of a planar grid
    '''
    idxx = np.linspace(mn,mx, eltosx)
    idxy = np.linspace(mn,mx, eltosy)
    u,v = np.meshgrid(idxx,idxy)
    
    u=u.flatten()
    v=v.flatten()
    
    #Paramitric space points
    points2D=np.vstack([u,v]).T 
    
    #!Todo:QD. Allow different topologyies. Use simpler algorithms
    triS = Delaunay(points2D)
    
    #return ([triangle[c] for triangle in triS.simplices] for c in range(3))
    return np.array(
            [[triangle[c] for triangle in triS.simplices] 
            for c in range(3)]).T
    
    
    
def compute_mesh_from_displacement(diplacemntMap):
    '''
    Calculates a mesh from a displacement map
    '''
    pm = compute_position_map(diplacemntMap)      
    return compute_mesh_from_position(pm)


def compute_mesh_from_position(positionMap):
    '''
    Calculates a mesh from a position map
    '''
    pm = positionMap
     
    #Construcción de la malla
    x = pm[:,:,0].flatten()
    y = pm[:,:,1].flatten()
    z = pm[:,:,2].flatten()        
    points3D = np.vstack([x,y,z]).T 
    tri = compute_grid_mesh_topology(eltosx=pm.shape[0],eltosy=pm.shape[1])
    
    return dict(points=points3D,triangles=tri)


def saveObj(path,verts,normals,faces):
    thefile = open(path, 'w')
    faces = faces + 1
    for item in verts:
      thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))
    
    for item in normals:
      thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))
    
    for item in faces:
      thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  
    
    thefile.close()
    
def load3DFileSet(path, filt='*', ext='*', joinSceneMeshes = True):
    #!todo la búsqueda y la carga se pueden pasar a una función y reutilizarla
    #por ejemplo en la carga de tifs
    fileNamesTmp = [f for f in glob(join(path, filt + '.' + ext))]
    
    scenesList = []
    fileNameList = []

    for f in fileNamesTmp:
        try:
            scenesList.append(pyassimp.load(f,
                              processing = 
                      pyassimp.postprocess.aiProcess_JoinIdenticalVertices+
                      pyassimp.postprocess.aiProcess_Triangulate+
                      pyassimp.postprocess.aiProcess_GenNormals))
#            scenesList.append(pyassimp.load(f))
            append = True
            
        except Exception:
            #!todo: lo suyo es llevar un log si no se puede abrir
            append = False
            
       
        if append:
            fileNameList.append(f)
            
    
    
    meshes = []
   
    if (joinSceneMeshes): 
    
        for s in scenesList:
            points=np.concatenate([m.vertices for m in s.meshes], axis =0)
            normals = np.concatenate([m.normals\
                                      for m in s.meshes if m.normals.any()], 
                                     axis=0)
            
            l = np.concatenate([[0], 
                                np.array([len(m.vertices) for m in s.meshes])])
            
            triangles = np.concatenate([m.faces+np.array([l[i]]*3) 
                    for i, m in enumerate(s.meshes)])
                    
            meshes.append( 
                    dict(points=points,triangles=triangles, normals=normals))
            
    else:
        for s in scenesList:
            for m in s.meshes:
                points  = np.array([m.vertices])
                normals = np.array ([m.normals])
                triangles = np.array([m.faces])
                meshes.append( 
                    dict(points=points,triangles=triangles, normals=normals))
                
    return meshes, fileNameList

#http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
#vertices por filas
def aabbTriangleOverlap (aabbCenter, aabbSize, vertices):
    def AXISTEST(a, b, fa, fb, v, w, aabbhalf):	   
        p0 = a*v[0] - b*v[1]
        p1 = a*w[0] - b*w[1]
        
        mn = min(p0,p1)
        mx = max(p0,p1)
        
        rad = fa * aabbhalf[0] + fb * aabbhalf[1];
        end = mn>rad or mx<-rad
        
        return end#, p0, p1)#,mn,mx,rad) 
    
    
    def AXISTEST_X01(a, b, fa, fb):	 
        return AXISTEST(a, b, fa, fb, 
                        (v[0,Y],v[0,Z]),
                        (v[2,Y],v[2,Z]),
                        aabbhalf[[Y,Z]]) #p0,p2
        
        
    def AXISTEST_X2(a, b, fa, fb):
        return AXISTEST(a, b, fa, fb, 
                        (v[0,Y],v[0,Z]),
                        (v[1,Y],v[1,Z]),
                        aabbhalf[[Y,Z]]) #p0,p1
    
    def AXISTEST_Y02(a, b, fa, fb):
        return AXISTEST(b, a, fa, fb, 
                        (v[0,Z],v[0,X]),
                        (v[2,Z],v[2,X]),
                        aabbhalf[[X,Z]]) #p0,p2		   
                    
    
    def AXISTEST_Y1(a, b, fa, fb):
            return AXISTEST(b, a, fa, fb, 
                        (v[0,Z],v[0,X]),
                        (v[1,Z],v[1,X]),
                        aabbhalf[[X,Z]]) #p0,p1		
    
    def AXISTEST_Z12(a, b, fa, fb):
            return AXISTEST(a, b, fa, fb, 
                        (v[1,X],v[1,Y]),
                        (v[2,X],v[2,Y]),
                        aabbhalf[[X,Z]]) #p1,p2				   
    
    def AXISTEST_Z0(a, b, fa, fb):
            return AXISTEST(a, b, fa, fb, 
                        (v[0,X],v[0,Y]),
                        (v[1,X],v[1,Y]),
                        aabbhalf[[X,Z]]) #p0,p1	

	
    X,Y,Z= (0,1,2)
    aabbhalf = aabbSize*0.5
    v = vertices - aabbCenter
    
    
#    if np.all(np.abs(v[0])<=0.5):
#        print ('ok')

    e0 = v[1]-v[0]
    e1 = v[2]-v[1]
    e2 = v[0]-v[2]
    
    fex,fey,fez = np.abs(e0)
    if (AXISTEST_X01(e0[Z], e0[Y], fez, fey)):
        return False
    if (AXISTEST_Y02(e0[Z], e0[X], fez, fex)):
        return False
    if (AXISTEST_Z12(e0[Y], e0[X], fey, fex)):
        return False


    fex,fey,fez = np.abs(e1)
    if (AXISTEST_X01(e1[Z], e1[Y], fez, fey)):
       return False
    if (AXISTEST_Y02(e1[Z], e1[X], fez, fex)):
       return False
    if (AXISTEST_Z0(e1[Y], e1[X], fey, fex)):
       return False
   
    fex,fey,fez = np.abs(e2)
    if (AXISTEST_X2(e2[Z], e2[Y], fez, fey)):
       return False
    if (AXISTEST_Y1(e2[Z], e2[X], fez, fex)):
       return False
    if (AXISTEST_Z12(e2[Y], e2[X], fey, fex)):
       return False

    mx = np.max(v,axis=0)
    mn = np.min(v,axis=0)

    if (any(mn>aabbhalf) or any(mx<-aabbhalf)):
        return False

    normal = np.cross (e0,e1)
    
    return _aabbPlaneOverlap(normal, v[0], aabbhalf)

#solo funciona en coordenadas del voxel
def _aabbPlaneOverlap(normal, point, aabbmax):
    
    nsign = np.sign(normal)
    
    vmin = -(aabbmax*nsign) - point
    vmax = (aabbmax*nsign) - point
    
    return (np.dot(normal,vmin)<=0) and (np.dot(normal,vmax)>=0)
    
def aabbPlaneOverlap (aabbCenter, aabbSize, normal, point):
    aabbhalf = aabbSize*0.5
    v = point - aabbCenter
    
    return _aabbPlaneOverlap(normal, v, aabbhalf)

#esta solucion sobrescribe las etiquetas que hubiese    
def voxlizeMeshBorder(img, mesh, label = 1, 
                      linearTransform = None, translation = None):
    
    points = mesh['points']
    triangles = mesh['triangles']

    if linearTransform is not None and translation is not None:
        p = (points@linearTransform) + translation
    elif linearTransform is not None: 
        p = points@linearTransform 
    elif translation is not None:
        p = points + translation
    else:
        p = points #no hace falta copiar puesto que no se modifican más
         
    one = np.array([1]*3)
    half = np.array([0.5]*3)

    for t in triangles:
        vertices = p[t]

        start = np.floor(np.min(vertices,axis=0))
        aabbCenter = half + start

        start = start.astype(np.int32)
        end = np.floor(np.max(vertices,axis=0) + one).astype(np.int32)
        s0,s1,s2 = end-start

        for index in np.ndindex(s0,s1,s2):
            i,j,k = index + start
            #!todo esta solucion se queda con la última etiqueta
            if img[i,j,k] != label:
                img[i, j, k] = aabbTriangleOverlap(
                    aabbCenter + index, one, vertices) * label
                        
                        
def voxlizeMeshBorderSubImg(mesh, label = 1, 
                            linearTransform = None, translation = None):
    
    points = mesh['points']
    triangles = mesh['triangles']

    if linearTransform is not None and translation is not None:
        p = (points@linearTransform) + translation
    elif linearTransform is not None: 
        p = points@linearTransform 
    elif translation is not None:
        p = points + translation
    else:
        p = points #no hace falta copiar puesto que no se modifican más
    
    one = np.array([1]*3)
    half = np.array([0.5]*3)
        
    imgAABBsize = np.floor(np.max(p, axis = 0)+one).astype(np.int32)
    imgAABBmin = np.floor(np.min(p, axis = 0)).astype(np.int32)
    imgAABBsize -= imgAABBmin 
    img = np.zeros(imgAABBsize)
         
    one = np.array([1]*3)
    half = np.array([0.5]*3)

    iiii = 0
    for t in triangles:
        #print (iiii)
        iiii+=1
        vertices = p[t]

        start = np.floor(np.min(vertices,axis=0))
        aabbCenter = half + start

        start = start.astype(np.int32)
        end = np.floor(np.max(vertices,axis=0) + one).astype(np.int32)
        s0,s1,s2 = end-start

        start-=imgAABBmin
        
        for index in np.ndindex(s0,s1,s2):
            i,j,k = index + start
            img[i, j, k] = (aabbTriangleOverlap(
                    aabbCenter + index, one, vertices) or
                    img[i, j, k])
       
    return (img*label, imgAABBmin, imgAABBsize)



#!todo: no avisa cuando los datos se sobreescriben
#!todo: podría devolver las subimagnes y aabb
def voxelizeMeshList(img,meshList, linearTransform, translation, labelOffset=0):

    offset = labelOffset+1

#    z0 = np.ones(9)
#    z0[[0,2,-3,-1]] = [0]*4
#    z0= z0.reshape((3,3))
#    zn = np.zeros((3,3))
#    zn[1,1] = 1 
#    structure = np.array([zn,z0,zn])

    
    for i, m in enumerate(meshList):
        (subimg, imgAABBmin, imgSize) = voxlizeMeshBorderSubImg(
                m, #label = off + i, 
                linearTransform = linearTransform, 
                translation = translation)
        (rs,ss,ts) = imgAABBmin
        (re,se,te) = imgAABBmin + imgSize
        #se rellenan los contornos
        img[rs:re, ss:se, ts:te] = \
            binary_fill_holes(subimg)*(i+offset) # estructura en cruz
#            binary_fill_holes(subimg, structure=structure)*(i+offset)
        #tiff.imsave('C:/Users/isabe/PycharmProjects/spinesGTPreprocessing/outputData/outputRelleno'+str(i)+'.tif', img[rs:re, ss:se, ts:te])


        #    voxlizeMeshBorder(img, m label = 1 + i,
#                linearTransform = linearTransform, 
#                translation = translation)
            
    
        
    
    
