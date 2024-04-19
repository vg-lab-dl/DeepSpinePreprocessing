# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 08:29:14 2019

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

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D #USED in fig.gca(projection='3d')
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


import plotly.graph_objs as go
from plotly.offline import plot  
from plotly import tools

from gmrv_utils import u3D
#from gmrv_utils import u3D

# =============================================================================
# Static map plots
# =============================================================================
def plot_position_map(pm):
    fig =plt.figure()
    ax = fig.gca(projection='3d')
    
    mx = np.amax(pm[:,:,2])
    ax.set_xlim([-mx, mx])
    ax.set_ylim([-mx, mx])
    ax.plot_surface(pm[:,:,0],pm[:,:,1],pm[:,:,2], cmap=cm.coolwarm,
                    linewidth=1, antialiased=True)
    
def plot_displacement_map(dm):
    pm = u3D.compute_position_map(dm)
    plot_position_map(pm)
    
    
# =============================================================================
# Dynamic map plots
# =============================================================================
def plot_displacement_map_plotly (dm,filename='test', **kwargs):
    tri_mesh = u3D.compute_mesh_from_displacement(dm, **kwargs)
    plot_mesh_plotly(tri_mesh,filename=filename, **kwargs)
    
def plot_position_map_plotly (pm,filename='test', **kwargs):
    tri_mesh = u3D.compute_mesh_from_position(pm, **kwargs)
    plot_mesh_plotly(tri_mesh,filename=filename, **kwargs)   
    
def _plot_mesh_plotly(tri_mesh):
    p = tri_mesh['points']
    t = tri_mesh['triangles']
    
    tri_mesh_py = go.Mesh3d (x=p[:,0], y=p[:,1], z=p[:,2],
                             i=t[:,0], j=t[:,1], k=t[:,2])
        
    return tri_mesh_py

def _plot_mesh_wireframe_plotly(tri_mesh, lineColor = '#000000', width=2 ):
    p = tri_mesh['points']
    t = tri_mesh['triangles']
    
    x = [[p[tri[i%3],0] for i in range (4)]+[None] for tri in t]
    y = [[p[tri[i%3],1] for i in range (4)]+[None] for tri in t]
    z = [[p[tri[i%3],2] for i in range (4)]+[None] for tri in t]
    x = [i for j in x for i in j]
    y = [i for j in y for i in j]
    z = [i for j in z for i in j]
    line_marker = dict(color=lineColor, width=width)
    lines =[go.Scatter3d(x=x, y=y, z=z, mode='lines', line=line_marker)]

    
    
    return lines

def plot_mesh_plotly(tri_mesh,filename='test', drawWireframe = True, **kwargs):
    #elimina los warnings de plotlty
    import warnings
    warnings.simplefilter("ignore")
    
    data = [_plot_mesh_plotly(tri_mesh, **kwargs)]
    if drawWireframe:
        data += _plot_mesh_wireframe_plotly(tri_mesh, **kwargs)
    
    fig=go.Figure(data=data)
    plot(fig, filename=filename)
    

def plot_displacement_array_map_plotly (dm_array,filename='test',cols=2, 
                                        **kwargs):
    s = len(dm_array)
    mesh_array = [None]*s
       
    for i in range(s):
        mesh_array[i] = u3D.compute_mesh_from_displacement(dm_array[i])
        
    plot_mesh_array_plotly(mesh_array,filename=filename,cols=cols)
    

def plot_position_map_array_plotly (pm_array,filename='test',cols=2, **kwargs):
    s = len(pm_array)
    mesh_array = [None]*s
    
    for i in range(s):
        mesh_array[i] = u3D.compute_mesh_from_position(pm_array[i])
        
    plot_mesh_array_plotly(mesh_array,filename=filename,cols=cols)


def plot_mesh_array_plotly(mesh_array,filename='test', cols=2, 
                           drawWireframe=True, **kwargs):
#    #elimina los warnings de plotlty
#    import warnings
#    warnings.simplefilter("ignore")

    s = len(mesh_array)
    r = s//cols if s % cols == 0 else s//cols + 1
    specs =   [[{'is_3d': True} for j in range(cols)] for i in range(r)]
      
    fig = tools.make_subplots(rows=r, cols=cols, specs=specs)
    
    
    for i in range(s):
        
        ii = (i // cols) + 1
        jj = (i % cols) + 1
        subfig = _plot_mesh_plotly(mesh_array[i])
       
        fig.append_trace(subfig,ii,jj)
        
        if drawWireframe:
            for subfig in  _plot_mesh_wireframe_plotly(mesh_array[i], **kwargs):
                fig.append_trace(subfig,ii,jj)

        
#    fig['layout'].update(go.Layout(
#                            margin=dict(
#                            r=10, l=10,
#                            b=10, t=10)
#                          ))
                            
    plot(fig, filename=filename)
    
    
# =============================================================================
#     Color Maps
# =============================================================================
#https://matplotlib.org/tutorials/colors/colormap-manipulation.html
#https://github.com/connorgr/colorgorical
#http://vrl.cs.brown.edu/color
#https://matplotlib.org/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html

colorDict = {
        'baseColors' : [(1,0,0),(0,1,0),(0,0,1)],
        '28' : [(0.02352941, 0.58823529, 0.40784314), 
                (0.62352941, 0.04313725, 0.39215686),
                (0.44705882, 1.        , 0.44705882),
                (0.84313725, 0.48627451, 0.86666667),
                (0.78039216, 0.86666667, 0.56862745),
                (0.27058824, 0.3254902 , 0.76078431),
                (0.44313725, 0.82745098, 0.95686275),
                (0.14117647, 0.35294118, 0.38431373),
                (0.14117647, 1.        , 0.80392157),
                (0.64705882, 0.07843137, 0.03529412),
                (0.94509804, 0.79607843, 0.83529412),
                (0.51764706, 0.31764706, 0.27843137),
                (0.74117647, 0.89019608, 0.11372549),
                (0.18039216, 0.18823529, 0.90588235),
                (0.43921569, 0.62352941, 0.05882353),
                (0.96862745, 0.4627451 , 0.49019608),
                (0.08235294, 0.71764706, 0.11764706),
                (0.87058824, 0.09803922, 0.96862745),
                (0.95294118, 0.83137255, 0.14901961),
                (1.        , 0.24313725, 0.71372549),
                (0.58039216, 0.65098039, 0.99215686),
                (0.81568627, 0.49019608, 0.03529412),
                (0.99215686, 0.34901961, 0.09019608)],
        'cmp2': [(0.95,0.95,0.95),(0.85,0.15,0),(0.8,0.25,0),(0,1,0)],
        'cmp3': [(0.95,0.95,0.95), #fondo 0 = 0 + 0 i1fondo +i2fondo
                 (0.2,0.2,0.2), # 1 = 1 + 0 i1c1+i2fondo ()
                 (0.2,0.2,0.2), # 2 = 2 + 0 i1c1+i2fondo
                 (0.85,0.25,0), # 3 = 0 +3 i1fondo + i2c1
                 (0,1,0), # 4 = 1 +3 i1c1+ i2c1 OK clase 2
                 (0.85,0.85,0.0), # 5 = 2 +3 i1c2 + i2c1 Error de clasificación
                 (0.85,0.25,0), # 6 = 0 + 6 i1fondo+ i2c2 Falta información
                 (0.85,0.85,0.0), # 7 = 1 + 6 i1c1+ i2c2 Error de clasificación
                 (0,0,1)], # 7 = 2 + 6 i1c2+ i2c2 OK clase 2
        'white' : [(1,1,1)],
        'black' : [(0,0,0)],
        'lightGrey' : [(0.95,0.95,0.95)],
        'darkGrey' : [(0.2,0.2,0.2)],
        }

    
def createColorMap (name, colorSet, 
                    firstColorList = None, 
                    lastColorList = None, 
                    nlabels = None,
                    N = None):
    
    colorSetSize = len(colorSet)
    fc =  firstColorList if firstColorList is not None else []
    lc =  lastColorList  if lastColorList  is not None else []
    
    dec = len(fc)+len(lc)
    
    nl = colorSetSize+dec if nlabels is None else nlabels
    N = nl if N is None else N
    
    if (nl>dec):
        colors = (colorSet * ((nl-dec-1)//colorSetSize+1))[:nl-dec]
    else:
        colors = []
        
    colors = (fc + colors + lc)[:nl]

    return LinearSegmentedColormap.from_list(
        name, colors, N=N)

#def clampColorMap (colorMap, color, rang):
    
    
        

    
    
    
    
    
    
    