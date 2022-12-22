# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:14:47 2022

@author: Jingyan Yu jingyan.yu@surrey.ac.uk; Alex Hagen-Zanker a.hagen-zanker@surrey.ac.uk
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

def distance_transform(inmap, mask, category):
    """
    Return the Eculidean distance transform to pixels of value category in the inmap
    , excluding pixels excluded by mask
    """
    reclassify_map = (inmap == category)
    if reclassify_map.sum()==0: #the map does not have a category of change
        outmap = np.ones(inmap.shape) * np.inf
        outmap[mask] = np.nan
    else:
        outmap = distance_transform_edt(np.logical_not(inmap == category))
        outmap[mask==0] = np.nan
    return outmap


def fuzzy_kappa(map_a, map_b, mask, similarity_matrix, f):

    """
    OUTPUTS (fk, sim)
    fk is the fuzzy kappa statistic
    sim is the similarity raster layer 
    %
    INPUTS
    map_a is first categorical raster layer (as integer 2D numpy array)
    map_b is second categorical raster layer (as integer 2D numpy array)
    Mask is binary raster layer to delineate the study area (as boolean 2D numpy array) 
    similarity_matrix is category similarity matrix (as float 2D numpy array)
    f is a distance decay function
    %
    PRECONDITIONS
    Categories in map_a and map_b are coded as integers, consecutively numbered and 
    starting at 0. It is allowed for categories to be fully absent in the maps.
    Values in similarity_matrix must be in the range [0,1]; The number of rows must be
    equal to the number of categories in map_a; The number of columns must be equal 
    to the number of categories in map_b.
    The argument f must be function handle for a unary function with inputs 
    in the range [0,inf] with f(0) = 1 and f(inf) = 0, and f(x) <= f(y) for 
    all x > y. The function must work element by element on a numpy array

    POSTCONDITIONS
    fk in range [-inf, 1] 
    sim of the same shape as map_a, map_b and mask, B, Mask; Values inside study area in 
    range [0,1]; Values outside study area NaN
    Corner case: The study area is empty:  fk = NaN, sim = NaN matrix 
    Corner case: Both maps are uniform and same class: fk = NaN, sim = ones matrix

    REFERENCE
    Hagen-Zanker, A., 2009. An improved Fuzzy Kappa statistic that accounts 
    for spatial autocorrelation. International Journal of Geographical 
    Information Science, 23(1), pp.61-73.
    """
    
    R = mask.sum()
    if R==0:
      return np.nan,np.empty(mask.shape) * np.nan

    m,n = similarity_matrix.shape

    fmaps_a = [f(distance_transform(map_a,mask,i)) for i in range(m)]
    fmaps_b = [f(distance_transform(map_b,mask,j)) for j in range(n)]
    
    fmaps_a = [np.nanmax(np.dstack([fmaps_a[i] * similarity_matrix[i][j] for i in range(m)]),2) for j in range(n)];
    fmaps_b = [np.nanmax(np.dstack([fmaps_b[j] * similarity_matrix[i][j] for j in range(n)]),2) for i in range(m)];

    E = 0 
    for i in range(m):
        for j in range(n):
            dis_a = fmaps_a[j][map_a==i];
            dis_b = fmaps_b[i][map_b==j];
           
            Ax,Ac = np.unique(dis_a,return_counts=True)
            Bx,Bc = np.unique(dis_b,return_counts=True)
           
            x, indicesA = np.unique(np.concatenate((Ax,Bx)),return_index=True)
            x, indicesB = np.unique(np.concatenate((Bx,Ax)),return_index=True)
            Ac =  np.concatenate ((Ac, Bc*0))[indicesA]
            Bc =  np.concatenate ((Bc, Ac*0))[indicesB]
            
            Ac = np.flip(Ac)
            Bc = np.flip(Bc)
            x = np.flip(x)

            cumAc = np.cumsum(Ac)
            cumBc = np.cumsum(Bc)
            cumProbAB = cumAc * cumBc 
            
            probAB = np.diff(np.concatenate(([0],cumProbAB)))  
            
            E = E + np.sum(probAB * x)
    E = E/R**2
    
    for j in range(n):
        fmaps_a[j][map_b!=j] = np.nan
   
    muA = np.nansum(np.dstack(fmaps_a),2)

    for i in range(m):
        fmaps_b[i][map_a!=i] = np.nan
   
    muB = np.nansum(np.dstack(fmaps_b),2)

    sim = np.minimum(muA,muB)
    P = np.nan_to_num(sim).sum()/R
   
    if E == 1:
        return np.nan, sim
    else:
        return (P-E)/(1-E), sim
    
def fuzzy_kappa_simulation(map_a_before, map_b_before, map_a_after, map_b_after, mask, similarity_matrix, f):
    """
    OUTPUTS (fks, sim)
    fk is the fuzzy kappa  similarity statistic
    sim is the similarity raster layer 
    %
    INPUTS
    map_a_before is the before-map of the first pair of categorical raster layers (as integer 2D numpy array)
    map_a_after is the after-map of the first pair of categorical raster layers (as integer 2D numpy array)
    map_b_before is the before-map of the second pair of categorical raster layers (as integer 2D numpy array)
    map_b_after is the after-map of the second pair of categorical raster layers (as integer 2D numpy array)
    Mask is binary raster layer to delineate the study area (as boolean 2D numpy array) 
    similarity_matrix is category similarity matrix (as float 2D numpy array)
    f is a distance decay function
    %
    PRECONDITIONS
    Categories in all categorical raster layers are coded as integers, consecutively numbered and 
    starting at 0. It is allowed for categories to be fully absent in any of the maps.
    Values in similarity_matrix must be in the range [0,1]; The number of rows must be
    equal to the number of categories in the first map pair; The number of columns must be equal 
    to the number of categories in the second map pair.
    The argument f must be function handle for a unary function with inputs 
    in the range [0,inf] with f(0) = 1 and f(inf) = 0, and f(x) <= f(y) for 
    all x > y. The function must work element by element on a numpy array

    POSTCONDITIONS
    fks in range [-inf, 1] 
    sim of the same shape as map_a, map_b and mask, B, Mask; Values inside study area in 
    range [0,1]; Values outside study area NaN
    Corner case: The study area is empty:  fk = NaN, sim = NaN matrix 
    Corner case: Both maps are uniform and same class: fk = NaN, sim = ones matrix

    REFERENCE
    van Vliet, J., Hagen-Zanker, A., Hurkens, J. and van Delden, H., 2013. A fuzzy set approach to assess 
    the predictive accuracy of land use simulations. Ecological Modelling, 261, pp.32-42.
    """
    m, n = similarity_matrix.shape
    map_a = m*map_a_before + map_a_after
    map_b = n*map_b_before + map_b_after
    similarity_matrix_expanded = np.zeros((m*m, n*n))
    for i1 in range(m):
        for i2 in range(m):
            for j1 in range(n):
                for j2 in range(n):
                    both_change_or_both_persist = (i1 == i2) == (j1 == j2)
                    if both_change_or_both_persist:
                         similarity_matrix_expanded[i1*m+i2][j1*n+j2]=similarity_matrix[i2][j2]
  
    return fuzzy_kappa(map_a, map_b, mask, similarity_matrix_expanded,f)

