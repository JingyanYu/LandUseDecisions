# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:14:47 2022

@author: Jingyan Yu jingyan.yu@surrey.ac.uk; Alex Hagen-Zanker a.hagen-zanker@surrey.ac.uk
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

#eq1 step1 generate the change categories map
def map_changes_to_categories(changed_states_resulted_number):
    """
    Change_map = mapa_before*2-mapa_after
    Map the before and after states resulted number of a cell to a change category.
    0 (non-urban to non-urban) - change type 0;-1(non-urban to urban) - change type 1;
    1(urban to urban) - change type 2;2(urban to non-urban) - change type 3
    """
    if changed_states_resulted_number==0:
        return 0
    elif changed_states_resulted_number==-1:
        return 1
    elif changed_states_resulted_number==1:
        return 2
    elif changed_states_resulted_number==2:
        return 3
    else:
        return -200
map_changes_to_categories_vec = np.vectorize(map_changes_to_categories)

def change_catetory_map(map_before, map_after):
    """
    This function takes the before and after map, returns a map with types of categories change.
    For example, with a binary urban/non-urban map, the types of change categories are UU, UN, NU, NN.
    The result map's cell state represent the change categories happened in the cell from before to after.
    """
    #map_before*2-map_after to get a resulted number from changed states of each cell for each category of change
    change_map = map_before*2-map_after
    return map_changes_to_categories_vec(change_map)

#eq1 step2 euclidean distance transform
def change_catetory_edt_map(change_category_map,out_of_bounary_mask,change_category):
    """
    For a change_category_map and a change category, return the Euclidean distance transform map.
    """
    reclassify_map = (change_category_map == change_category)
    if reclassify_map.sum()==0: #the map does not have a category of change
        change_catetory_edt_map = np.ones(change_category_map.shape) * np.inf
        change_catetory_edt_map[out_of_bounary_mask] = np.nan
        return change_catetory_edt_map
    else:
        change_catetory_edt_map = distance_transform_edt(np.logical_not(change_category_map == change_category))
        change_catetory_edt_map[out_of_bounary_mask] = np.nan
        return change_catetory_edt_map
    
#eq1 step3 distance decay
def f_change_catetory_edt_map(change_catetory_edt_map,beta):
    """
    This function returns the distance decay result for a change_catetory_edt_map.
    """
    return np.exp(-beta*change_catetory_edt_map)  

def f_change_category_edt_maps(change_category_map,out_of_bounary_mask,change_categories,beta):
    change_category_edt_maps = [change_catetory_edt_map(change_category_map,out_of_bounary_mask,change_category=category)\
                               for category in change_categories]
    return [f_change_catetory_edt_map(change_catetory_edt_map,beta) for change_catetory_edt_map in change_category_edt_maps]


#eq1 step4 calculate mu
def mu(change_category_map_a,change_catetory_map_b,out_of_bounary_mask,change_categories,beta):
    """
    This function calculates the mu - degree of belong of cells in map A to the categories found in map B.
    """
    
    change_catetory_edt_mapas = [change_catetory_edt_map(change_category_map_a,out_of_bounary_mask,category) \
                                 for category in change_categories]
    fmapas = [f_change_catetory_edt_map(change_catetory_edt_mapa,beta) \
              for change_catetory_edt_mapa in change_catetory_edt_mapas]
    for category in change_categories:
        fmapas[category][np.logical_not(change_catetory_map_b==category)] = np.nan
    
    mu = np.nansum(np.dstack(fmapas),2)
    mu[np.isnan(fmapas[0]) & np.isnan(fmapas[0]) & np.isnan(fmapas[0])]=np.nan
    return mu

#eq3 The agreement of a cell - the degree to which the cell in map A belongs to the category found in map B and vice versa.
def maps_agreement(change_category_mapa,change_category_mapb,out_of_bounary_mask,change_categories,beta):
    muab = mu(change_category_mapb,change_category_mapa,out_of_bounary_mask,change_categories,beta)
    muba = mu(change_category_mapa,change_category_mapb,out_of_bounary_mask,change_categories,beta)
    return(np.minimum(muab,muba))


#eq4 Calculate P - the mean agreement of the two compared maps
def mean_map_agreement(change_category_mapa,change_category_mapb,out_of_bounary_mask,R,change_categories,beta):
    return np.nan_to_num(maps_agreement(change_category_mapa, change_category_mapb,
                                        out_of_bounary_mask,change_categories,beta)).sum()/R

#eq5 generate distributions of each change category in a map
def map_catogory_distributions(change_category_map,out_of_bounary_mask,change_categories,beta):
    fmaps = f_change_category_edt_maps(change_category_map,out_of_bounary_mask,change_categories,beta)
    return [[fmap[change_category_map==category] for category in change_categories] for fmap in fmaps]

#eq7-10 calculate PAB & E
def PAB_E(change_category_mapa,change_category_mapb,out_of_bounary_mask,R,change_categories,beta):
    dis_as = map_catogory_distributions(change_category_mapa,out_of_bounary_mask,change_categories,beta)
    dis_bs = map_catogory_distributions(change_category_mapb,out_of_bounary_mask,change_categories,beta)
    E = 0 
    for i in change_categories:
        for j in change_categories:
            Ax,Ac = np.unique(dis_as[i][j],return_counts=True)
            Pai = sum(Ac)/R
            Bx,Bc = np.unique(dis_bs[j][i],return_counts=True)
            Pbj = sum(Bc)/R
            
            x, indicesA = np.unique(np.concatenate((Ax,Bx)),return_index=True)
            x, indicesB = np.unique(np.concatenate((Bx,Ax)),return_index=True)
            Ac =  np.concatenate ((Ac, Bc*0))[indicesA]
            Bc =  np.concatenate ((Bc, Ac*0))[indicesB]
            
            Ac = np.flip(Ac)
            Bc = np.flip(Bc)
            x = np.flip(x)

            cumAc = np.cumsum(Ac) / np.sum(Ac)
            cumBc = np.cumsum(Bc) / np.sum(Bc)
            cumProbAB = cumAc * cumBc 
            
            probAB = np.diff(np.concatenate(([0],cumProbAB)))  
            
            Eij = np.sum(probAB * x)
            E  = E + Pai * Pbj * Eij
        return E
    
# eq 11 calculate FKS
def FKS(change_category_mapa,change_category_mapb,out_of_bounary_mask,R,change_categories=[0,1,2],beta=1.0):
    P = mean_map_agreement(change_category_mapa,change_category_mapb,out_of_bounary_mask,R,change_categories,beta)
    E = PAB_E(change_category_mapa,change_category_mapb,out_of_bounary_mask,R,change_categories,beta)
    return (P-E)/(1-E)