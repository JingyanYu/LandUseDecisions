# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:39:59 2021

@author: reneryu
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:47:44 2020

@author: JingyanYu
"""
import numpy as np
# from scipy.signal import oaconvolve, convolve2d
from scipy.signal import fftconvolve
# from cupyx.scipy.signal import fftconvolve
####Plot and print libraries
# from rasterio.plot import show

# from rasterio.windows import Window
# import sys
# np.set_printoptions(threshold=sys.maxsize)
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
#import time
#from scipy.stats import ks_2samp
#import seaborn as sns
#from itertools import combinations, product



####0 Data


####Initialization: 1. create initial land use map 2. store cell-to-cell distance 3.Create random map
#1 Initialize 100*100 CA grid. Two land use classes urban 1 and non-urban 0 are assigned to each cell.

####Neighboudhood Effect 1 - Urban Density
#HelpFunc1 Exponential decay 2d convolution kernel
def kernel_expo_square(cutoff,beta):
    '''
    Input: 1. exponential decay cutoff value - cutoff, 2. exponential lambda - beta;
    Output: a square exponential decay kernel matrix, 
            each entry has the exponential distance decay value from the square kernel's centre, normalised by the sum of the kernel.
    Algorithm:
    1. Given cutoff, calculate the distance where the cutoff value is reached, ceil round to the nearest larger integer a;
    2. Using numpy.meshgrid to generate distance matrix cooridnates; vectorization calculate distance of each cell to the centre of the square;
    3. Apply the exponential decay function to the distances matrix and return the normalized exponential decay kernel matrix.
    '''
    a = int(np.ceil(np.log(cutoff)/(-beta)))
    centre_row, centre_col = a, a
    xv, yv = np.meshgrid(np.arange(2*a+1),np.arange(2*a+1))
    d_celltocentre = ((xv-centre_row)**2+(yv-centre_col)**2)**0.5
    kernel = np.exp(-beta*d_celltocentre)
    return kernel/sum(sum(kernel))

####Neighbourhood Effect 3 - Service provided
#HelpFunc3 Service provided - 2D Convolution of 2D Convolution
def service_provided(kernel,landmap):
    '''
    Input: 1. A kernel calculated/returned by func kernel_expo_square; 2. Land map;
    Output: service provided NE3 matrix as the 2d convolution of the green usage intensity map - a 2d convolution of green map.
    Algorithm:
    1. Calculate green_density using 2d convolution of landmap;
    2. Calculate green_use_intensity;
    3. Calculate service_provided using 2d convolution of green_use_intensity.    
    Note: 
    1 To avoid zero division caused by urban cells with no non-urban cell in neighbourhood which get 0 in green_density,
    green_use_intensity divide only where green_density is not 0. Instead of inf under zero division, those urban cells get 0. as
    intensity. They will not affect green intensity density in the second 2d convolution as they are not in non-urban cells' neighbourhood.
    2 0.0001 chosen in the calculation of green_use_intensity is numerical based on the precision difference 
    between scipy.signal functions fftconvove and convolve2d regarding extremely small values in green_density.
    '''
    green = 1-landmap
    
    green_density = fftconvolve(green,kernel,mode='same')
    green_use_intensity = np.divide(landmap,green_density,where=green_density>=0.0001)
    return -fftconvolve(green_use_intensity, kernel, mode='same')

#####Transition Potential:NE1+NE3+Random
def Potentials(uPara,nuPara,kernels,landmap,randmap):
    '''
    Input: 1. A list of kernels calculated/returned by func kernel_expo_square; 2.func service_provided's arguments landmap; 3. random map;
           4. target parameters for urban density array uPara = [C_beta1, C_beta2, C_beta3] 
           and service provided array nuPara = [G_beta1, G_beta2, G_beta3] neighbourhood effects.
           3. transition_num number of transition cells.
    Output: potential matrix
    Algorithm: 
    1. calculate NE1 2. calculate NE3 3. return potentials matrix = NE1+NE3+randmap
    '''
    
    NE1 = uPara[0]*fftconvolve(landmap,kernels[0],mode='same')+\
          uPara[1]*fftconvolve(landmap,kernels[1],mode='same')+\
          uPara[2]*fftconvolve(landmap,kernels[2],mode='same')
    NE3 = nuPara[0]*service_provided(kernels[0],landmap)+\
          nuPara[1]*service_provided(kernels[1],landmap)+\
          nuPara[2]*service_provided(kernels[2],landmap)    
    return NE1+NE3+randmap


def transitions(uPara,nuPara,randmap,kernels,landmap,transition_num):
    '''
    Input: 1.func Potentials' arguments uPara,nuPara,kernels,landmap,randmap; 
           2. number of transition cells transition_num.
    Output: chosen non-urban cells' grid indices based on the computed potential matrix,
    in form - xs= [x1,x2,..],ys = [y1,y2,...].
    Note: revised based on previous version to handle the water class. 
        1. The water class is assumed to have values of 2 in the landmap.
        2. The transition function first makes a copy of the landmap, 
           and changes the water class to have value 0 as the green class.
        3. The landmap copy is used in the potential calculation, with the water class treated as green/nonurban.
        4. The transition cells are chosen among the green class.
    '''
    landmap_copy = landmap.copy()
    if (landmap==-200).sum()!=0:
        landmap_copy[landmap==-200]=0
    
    potentials = Potentials(uPara=uPara,nuPara=nuPara,kernels=kernels,landmap=landmap_copy,randmap=randmap)
    xs,ys = np.where(landmap==0)
    nonurban_potentials = potentials[xs,ys]
    # print(type(nonurban_potentials.shape[0]))
    chosen_indices = np.argpartition(nonurban_potentials,(nonurban_potentials.shape[0]-transition_num))[-transition_num:]
#    print(potentials[xs[chosen_indices],ys[chosen_indices]])
    return xs[chosen_indices],ys[chosen_indices]

####CCA Model - return snapshots at all iterations
# def CCA_all_snapshots(uPara,nuPara,
#                       urban_num = 1445, trans_num = 15,
#                       kernels=[kernel_expo_square(0.01,0.2),kernel_expo_square(0.01,0.5),kernel_expo_square(0.01,2.0)],
#                       landmap=None,randmap=None,rows=100,cols=100,seed=1):
#     """
#     The model run function, call signature: result = CCA_last_snapshot(1,1,1,1,1,1,seed=3).
#     Returns the last snapshot of an (iter_num, rows, cols) shape array which stores land use map snapshot at each iteration.
#     """
#     uPara = np.array(uPara)
#     nuPara = np.array(nuPara)
#     iter_num = urban_num//trans_num
#     rest_num = urban_num%trans_num
#     result = np.empty(shape=(iter_num+1,rows,cols))
#     if landmap is None:
#         landmap = np.zeros((rows,cols))
#     if seed is not None:
#         np.random.seed(seed)
#     if randmap is None:
#         randmap=np.random.normal(0,1.0,size=(rows,cols))
        
#     if rest_num != 0:
#         result = np.empty(shape=(iter_num+2,rows,cols))
#         result[0]=landmap
#         for time_step in range(iter_num):  
#             trans_xs,trans_ys = transitions(uPara=uPara,nuPara=nuPara,randmap=randmap,
#                                     kernels=kernels,landmap=landmap,
#                                     transition_num=trans_num)
#             landmap[trans_xs,trans_ys]=1
#             result[time_step+1]=landmap
        
#         trans_xs,trans_ys = transitions(uPara=uPara,nuPara=nuPara,randmap=randmap,
#                                     kernels=kernels,landmap=landmap,
#                                     transition_num=rest_num)
#         landmap[trans_xs,trans_ys]=1
#         result[iter_num+1]=landmap
#     elif rest_num == 0:
#         result = np.empty(shape=(iter_num+1,rows,cols))
#         result[0]=landmap
#         for time_step in range(iter_num):
            
#             trans_xs,trans_ys = transitions(uPara=uPara,nuPara=nuPara,randmap=randmap,
#                                     kernels=kernels,landmap=landmap,
#                                     transition_num=trans_num)
#             landmap[trans_xs,trans_ys]=1
#             result[time_step+1]=landmap
    
#     return result

####CCA Model - return the last snapshot
def CCA_last_snapshot(uPara,nuPara,
                      urban_num = 1445, trans_num = 15,
                      kernels=[kernel_expo_square(0.01,0.2),kernel_expo_square(0.01,0.5),kernel_expo_square(0.01,2.0)],
                      landmap=None,randmap=None,rows=100,cols=100,seed=None):
    """
    The model run function, call signature: result = CCA_last_snapshot(1,1,1,1,1,1,seed=3).
    Returns the last snapshot of an (iter_num, rows, cols) shape array which stores land use map snapshot at each iteration.
    """
    uPara = np.array(uPara)
    nuPara = np.array(nuPara)
    iter_num = urban_num//trans_num
    rest_num = urban_num%trans_num
    
    if landmap is None:
        landmap = np.zeros((rows,cols))
    if seed is not None:
        np.random.seed(seed)
   
    if randmap is None:
        randmap=np.random.normal(0,1.0,size=(rows,cols))
    
    if rest_num != 0: #urban_cell%trans_num!=0,add another round of transition of rest_num cells after the iterations.
        result = np.empty(shape=(iter_num+2,rows,cols))
        result[0]=landmap
        for time_step in range(iter_num):  
            trans_xs,trans_ys = transitions(uPara=uPara,nuPara=nuPara,randmap=randmap,
                                    kernels=kernels,landmap=landmap,
                                    transition_num=trans_num)
            landmap[trans_xs,trans_ys]=1
            result[time_step+1]=landmap
        
        trans_xs,trans_ys = transitions(uPara=uPara,nuPara=nuPara,randmap=randmap,
                                    kernels=kernels,landmap=landmap,
                                    transition_num=rest_num)
        landmap[trans_xs,trans_ys]=1
        result[iter_num+1]=landmap
    elif rest_num == 0:
        result = np.empty(shape=(iter_num+1,rows,cols))
        result[0]=landmap
        for time_step in range(iter_num):
            
            trans_xs,trans_ys = transitions(uPara=uPara,nuPara=nuPara,randmap=randmap,
                                    kernels=kernels,landmap=landmap,
                                    transition_num=trans_num)
            landmap[trans_xs,trans_ys]=1
            result[time_step+1]=landmap
    return result[-1]


# np.save(r'results\simulatedlandmaps_1000trials\C1100C2100C310C41C51C61_900.npy',results)

