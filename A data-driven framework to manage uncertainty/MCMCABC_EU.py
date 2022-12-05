# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:39:03 2021

@author: reneryu
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:35:37 2020

@author: JingyanYu
"""
import CCA_EU
import numpy as np
from numpy.random import multivariate_normal
from scipy.signal import fftconvolve

import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
# from rasterio.plot import show
# import matplotlib.pyplot as plt
# from itertools import combinations, product
from scipy.stats import ks_2samp

##Initialisation 1:specify priors: C1-C3 ~ U[0,100]; C4-C6 ~ U[0,10]
##use the mean of the uniform distributions as the initial values of Ci
C1_interval,C2_interval,C3_interval,C4_interval = [0,100],[0,100],[0,10],[0,10]
C1,C2,C3,C4 = np.mean(C1_interval),np.mean(C2_interval),np.mean(C3_interval),np.mean(C4_interval)

##Initialisation 2:specify the convariance matrix
initial_covmx_oxford = np.array([[19.2,0.0,0.0,0.0],
                                 [0.0,250.7,0.0,0.0],
                                 [0.0,0.0,3.8,0.0],
                                 [0.0,0.0,0.0,5.4]])

initial_covmx_swindon = np.array([[19.3,0.0,0.0,0.0],
                                  [0.0,257.5,0.0,0.0],
                                  [0.0,0.0,2.3,0.0],
                                  [0.0,0.0,0.0,0.37]])

##Initialisation 3: Store kernels for 2d convolution
kernels=[CCA_EU.kernel_expo_square(0.01,beta) for beta in [0.2,0.5,2.0]]

##Initialisation 4: Observation data           
ghs_built_raster_paths = [r'data\GHS_BUILT_LDS1975_GLOBE_R2018A_54009_250_V2_0\GHS_BUILT_LDS1975_GLOBE_R2018A_54009_250_V2_0.tif',
                          r'data\GHS_BUILT_LDS1990_GLOBE_R2018A_54009_250_V2_0\GHS_BUILT_LDS1990_GLOBE_R2018A_54009_250_V2_0.tif',
                          r'data\GHS_BUILT_LDS2000_GLOBE_R2018A_54009_250_V2_0\GHS_BUILT_LDS2000_GLOBE_R2018A_54009_250_V2_0.tif',
                          r'data\GHS_BUILT_LDS2014_GLOBE_R2018A_54009_250_V2_0\GHS_BUILT_LDS2014_GLOBE_R2018A_54009_250_V2_0.tif']
years = ['1975','1990','2000','2014']

def categorize_urban(percent):
    if percent >= 50: #urban centre
        return 1
    if percent >= 20 and percent < 50: #urban cluster
        return 1 
    if percent >= 0 and percent < 20: #non-urban
        return 0 
    if percent < 0: #no value
        return -200
categorize_urban_vec = np.vectorize(categorize_urban)

def condition_urban_density_medium(categorized_FUA,urban_cells,kernels=kernels):
    obs_urbandensity1 = fftconvolve(categorized_FUA,kernels[1],mode='same')[urban_cells==1].flatten()
    return np.transpose(np.expand_dims(obs_urbandensity1, axis=0))


#def FUAfromGHSL(CntyName,FUAname,kernels=kernels):
#    fua_gdf = gpd.read_file(r'data\GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0\GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg')
#    country_gpd = fua_gdf[fua_gdf['Cntry_name']==CntyName]
#    shape = country_gpd[country_gpd['eFUA_name']==FUAname].iloc[0]['geometry']
#    
#    with rio.open(ghs_built_raster_paths[2]) as src:    
#        out_img, out_transform = mask(src, shape, crop=True)
#    categorized_FUA = np.squeeze(categorize_urban_vec(out_img))
#    initialmap = categorized_FUA.copy()
#    initialmap[initialmap==1]=0
#    initialmap_zeropadded =  initialmap.copy()
#    initialmap_zeropadded[initialmap_zeropadded==-200] = 0
#    
#    rows, cols = categorized_FUA.shape
#    urban_num = (categorized_FUA==1).sum()
#    
#    changemap = categorized_FUA.copy()
#    changemap[changemap==-200] = 0
#    obs_densities_d1 = condition_urban_density_medium(initialmap_zeropadded,changemap)
#    obs_densities_d2 = condition_urban_density_medium(changemap,changemap)
#    obs_densities = np.concatenate((obs_densities_d1,obs_densities_d2), axis=1)    
#    # obs_densities = [np.sort(fftconvolve(initialmap_zeropadded,kernels[i],mode='same')[zeropadboundary_FUA==1].flatten()) for i in range(3)]    
#    return initialmap,initialmap_zeropadded,obs_densities,urban_num,rows,cols

# countries = ['France','France','France','France','France','France','France',
#  'Germany','Germany','Germany','Netherlands','Netherlands','Netherlands','UnitedKingdom','UnitedKingdom','Belgium']
# FUAs = ['Angers','Poitiers','Tours','Clermont-Ferrand','Montpellier','Nimes','Metz',
        # 'Mönchengladbach','Dusseldorf','Wuppertal','Utrecht',"'s-Hertogenbosch",'Enschede','Belfast','Blackwater','Mons']
        
countries = ['Austria','Denmark','Finland','Greece','Hungary','Italy','Luxembourg','Norway',
             'Poland','Portugal','Slovakia','Spain','Sweden','Switzerland','Turkey']
FUAs = ['Linz','Aarhus','Helsinki','Thessaloniki','Debrecen','Verona','Luxembourg','Bergen',
        'Częstochowa','Braga','Bratislava','Seville','Malmö','Bern','Ankara']
#initialmap,initialmap_zeropadded,obs_densities,urban_num,rows,cols = FUAfromGHSL(countries[0],FUAs[0])


def FUAfromGHSL_changes(CntyName,FUAname,kernels=kernels):
    fua_gdf = gpd.read_file(r'data\GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0\GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg')
    country_gpd = fua_gdf[fua_gdf['Cntry_name']==CntyName]
    shape = country_gpd[country_gpd['eFUA_name']==FUAname].iloc[0]['geometry']
    
    with rio.open(ghs_built_raster_paths[0]) as src:    
        out_img, out_transform = mask(src, shape, crop=True)
    categorized_FUA = np.squeeze(categorize_urban_vec(out_img))
    initialmap = categorized_FUA.copy()
    initialmap_zeropadded = categorized_FUA.copy()
    initialmap_zeropadded[initialmap_zeropadded==-200] = 0
    
    with rio.open(ghs_built_raster_paths[1]) as src:    
        out_img, out_transform = mask(src, shape, crop=True)
    categorized_FUA = np.squeeze(categorize_urban_vec(out_img))
    changemap = categorized_FUA.copy()
    
    urban_num = (categorized_FUA==1).sum()-(initialmap==1).sum()
    rows, cols = categorized_FUA.shape
    
    changemap[changemap==-200] = 0
     # obs_densities = [np.sort(fftconvolve(changemap,kernels[i],mode='same')[categorized_FUA==1].flatten()) for i in range(3)]
     # return categorized_FUA,initialmap,changemap,obs_densities,urban_num,rows,cols
    obs_densities_d1 = condition_urban_density_medium(initialmap_zeropadded,changemap)
    obs_densities_d2 = condition_urban_density_medium(changemap,changemap)
    obs_densities = np.concatenate((obs_densities_d1,obs_densities_d2), axis=1)
    return initialmap,initialmap_zeropadded,obs_densities,urban_num,rows,cols

initialmap,initialmap_zeropadded,obs_densities,urban_num,rows,cols = FUAfromGHSL_changes(countries[2],FUAs[2])



#1975 Oxford urban cells: 1445 = 17*85 rols196 cols176 iter_num=86 transition_num=17
#Chain1 initial para [6.7333486,68.87145612,2.61832417,3.96224226]
#Chain2 initial para [6.67504623,78.6366322,2.00148601,5.39610628]
#Chain3 initial para [5.74713275,78.95402972,1.45394707,5.51829238]  

#1975 Swindon urban cells: 1141 = 7*163 rols200 cols212:iter_num = 164;trans_num=7
#Chain1 initial para [7.09343533,48.88358648,5.33893115,2.11079493]
#Chain2 initial para [12.78305599,83.59569675,2.31424336,1.41159063]
#Chain3 initial para [10.60331248,80.86676804,2.14985003,1.29958185]

# def ks_large_urban(simulation,obs_density,kernel=kernels[0]):
    
#     sim_density = np.sort(fftconvolve(simulation,kernel,mode='same')[simulation==1].flatten())
#     statistic, pvalue = ks_2samp(obs_density,sim_density)
#     return statistic

# def ks_medium_urban(simulation,obs_density,kernel=kernels[1]):
    
#     sim_density = np.sort(fftconvolve(simulation,kernel,mode='same')[simulation==1].flatten())
#     statistic, pvalue = ks_2samp(obs_density,sim_density)
#     return statistic

# def ks_sum(simulation,obs_densities,kernels=kernels):
#     '''
#     Input:1. obs_densities - flattened 1d arrays of observation density on three spatial scale;
#           2. simulation - simulated land use; 3. kernels
#     Output: sum of KS statistics between an observation and a simulation on three spatial scales
#     '''
#     ks_sums = []
#     for i in range(3):
#         sim_density = fftconvolve(simulation,kernels[i],mode='same')[simulation==1].flatten()
#         statistic, pvalue = ks_2samp(obs_densities[i],sim_density)
#         ks_sums.append(statistic)
#     return 1*ks_sums[0]+1*ks_sums[1]+1*ks_sums[2]


# def ks_small_urban(simulation,obs_density,kernel=kernels[2]):
#     '''
#     Input:1. obs_densities - flattened 1d arrays of observation density on three spatial scale;
#           2. simulation - simulated land use; 3. kernels
#     Output: sum of KS statistics between an observation and a simulation on three spatial scales
#     '''
#     sim_density = fftconvolve(simulation,kernel,mode='same')[simulation==1].flatten()
#     statistic, pvalue = ks_2samp(obs_density,sim_density)
#     return statistic

def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    ontinuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]
    
    index0_r = np.where(r==0)
    index0_s = np.where(s==0)
    ignore_indices = np.unique(np.squeeze(np.concatenate((index0_r,index0_s),axis=1)))
    r_new = np.delete(r,ignore_indices)
    s_new = np.delete(s,ignore_indices)
    

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return abs(-np.log(r_new/s_new).sum() * d / n) + np.log(m / (n - 1.))



####version 4.0: revise gof based on empirical observation cdfs' differences
def mcmcabc(nsteps,
            initial_paras=[C1,C2,C3,C4],
            para_intervals=np.array([C1_interval,C2_interval,C3_interval,C4_interval]),
            initial_covmx=None, 
            kernels=kernels,
            rows=rows,cols=cols,
            urban_num=urban_num,trans_num=15,
            initial_landmap=initialmap,
            initial_landmap_zeropadded = initialmap_zeropadded,
            obs_densities_urban = obs_densities,
            initial_epsilon=0.8):
    '''
    This function uses three spatial scales' KS statistics as goodness-of-fit/distance function in the MCMCABC algorithm.
    In step 4 in function mcmcabc, if on all three spatial scales, (mean_simulation-mean_difference) is smaller than 2 std_simulation,
    the simulated land use patterns are determined as the same to the observation, and the proposed thetas are accepted; otherwise,
    the proposed thetas are rejected.
    '''
    #Initialize: 1 get target parameters' initial values from the input; 
    current_thetas = initial_paras  
    accepted_thetas = []
    #Initialize: 2 variables to adapt covariance matrix 
    covmx = initial_covmx
    consequent_reject_times = 0
    #Initialize: 3 variables to adapt epsilon
    current_epsilon = initial_epsilon
    accepted_distance = []
    #Initialize: test purpose
    # accepted_landpatterns = []
    seeds = []
    
    
    #Iterations - for i in specified number of draws, 
    for i in range(nsteps):
        #Draw proposed thetas from a multivariate normal distribution;
        while True: 
            thetas = multivariate_normal(current_thetas,covmx)
            if np.all([thetas>=para_intervals[:,0],thetas<=para_intervals[:,1]]):
                break
        proposed_thetas = thetas
        #Simulate with proposed thetas
        uPara = proposed_thetas[:2]
        nuPara = proposed_thetas[2:]
        if initial_landmap  is not None: 
            landmap = initial_landmap.copy()      
        else:
            landmap = None
        seed = np.random.randint(1000000000)
        sim_landmap = CCA_EU.CCA_last_snapshot([uPara[0],0,uPara[1]],
                                            [0,nuPara[0],nuPara[1]],
                                            seed=seed,landmap=landmap,
                                            rows=rows,cols=cols,urban_num=urban_num,trans_num=trans_num) 
        
        sim_landmap_zeropadded = sim_landmap.copy()
        sim_landmap_zeropadded[sim_landmap_zeropadded==-200] = 0
        sim_densities_d1 = condition_urban_density_medium(initial_landmap_zeropadded,sim_landmap_zeropadded)
        sim_densities_d2 = condition_urban_density_medium(sim_landmap_zeropadded,sim_landmap_zeropadded)
        sim_densities = np.concatenate((sim_densities_d1,sim_densities_d2), axis=1)
        distance = KLdivergence(obs_densities, sim_densities)
        
        # distance = ks_large_urban(simulation=sim_landmap, obs_density=obs_densities_urban)
        # distance = ks_medium_urban(simulation=sim_landmap, obs_density=obs_densities_urban)
        # distance = ks_small_urban(simulation=sim_landmap, obs_density=obs_densities_urban)
        # distance = ks_sum(simulation=sim_landmap, obs_densities=obs_densities_urban)
        if distance <= current_epsilon:
            print('accept: ',i,proposed_thetas)
            accepted_thetas.append(proposed_thetas)
            current_thetas = proposed_thetas
#            consequent_reject_times = 0
            # accepted_landpatterns.append(sim_landmap)
            accepted_distance.append(distance)
            seeds.append(seed)
            
        else:
            print('reject: ',i,proposed_thetas,'Distance: ',distance)
#            consequent_reject_times += 1

        if (i//500>0) and (i%500==0):
            covmx = np.cov(np.transpose(accepted_thetas))
            current_epsilon = np.median(accepted_distance[-20:])
    return accepted_thetas,seeds,accepted_distance


