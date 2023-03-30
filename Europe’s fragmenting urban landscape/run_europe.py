# Import required Python modules
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import ks_2samp
from tqdm import tqdm
import pickle

tqdm.pandas()

# Import CCA model from previous study
import CCA_EU

# preprocess the input data 
# Function to binary reclassify GHSL urban percentage to binary urban, using 20% threshold

def categorize_urban_vec(percent):
    threshold = 20
    out = np.full_like(percent, nodata_value)
    out[(percent >= 0) & (percent < threshold) ] = 0
    out[percent >= threshold] = 1
    return out
 
#create columns - urban units at 1975, 1990, 2000, 2014,and urban growth between 75-90, 90-00, 00-14
def add_raster_to_dataframe(dataframe, raster_path, column_name):
        def process(x):
            shape = x['geometry']
            with rio.open(raster_path) as src:    
                out_img, out_transform = mask(src, [shape], crop=True)
            fua_urban_raster = np.squeeze(categorize_urban_vec(out_img))
            try:
                x[column_name] = np.unique(fua_urban_raster,return_counts=True)[1][2]
            except:
                x[column_name] = np.nan
            try:
                x['Transform'] = out_transform;
            except:
                x['Transform'] = np.nan
            try:
                x[column_name + ' raster'] = fua_urban_raster.copy();
            except:
                x[column_name + ' raster'] = np.nan
            return x
        
        dataframe = dataframe.apply(process, axis=1)
        return dataframe 
        
# Load the parameters for modelling growth using the four spatial development scenarios 

# 0 - compact; 1 - medium compact; 2 - medium dispersed; 3 - dispersed
        
paras_fourscenarios = np.load(r'parameters\paras_fourscenarios.npy',allow_pickle=True)
seeds_fourscenarios = np.load(r'parameters\seeds_fourscenarios.npy',allow_pickle=True)

def get_random_parameters_for_scenario(scenario, size): 
    max_size = paras_fourscenarios[scenario].size
    nums = np.random.choice(max_size, size=size, replace=False)
    return paras_fourscenarios[scenario][nums], seeds_fourscenarios[scenario][nums] 
    
    #This is from MCMCABC and CCA and should be probably imported rather than copied


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

kernels=[kernel_expo_square(0.01,beta) for beta in [0.2,0.5,2.0]]

#function to measure the difference between an observation and a simulation map
def ks_dis(obs,sim):
    """
    This function measures the difference between an observation and a simulation map
    using KS statistic.
    Input: an observation and a simulation map - numpy arrays
    Output: the sum of KS statistics on three spatial scales.
    """
    obs_urbandensity0 = fftconvolve(obs,kernels[0],mode='same')[obs==1].flatten()
    obs_urbandensity1 = fftconvolve(obs,kernels[1],mode='same')[obs==1].flatten()
    obs_urbandensity2 = fftconvolve(obs,kernels[2],mode='same')[obs==1].flatten()
    simulation_urbandensity0 = fftconvolve(sim,kernels[0],mode='same')[sim==1].flatten()
    statistic0, pvalue = ks_2samp(obs_urbandensity0,simulation_urbandensity0)
    simulation_urbandensity2 = fftconvolve(sim,kernels[2],mode='same')[sim==1].flatten()
    statistic2, pvalue = ks_2samp(obs_urbandensity2,simulation_urbandensity2)
    simulation_urbandensity1 = fftconvolve(sim,kernels[1],mode='same')[sim==1].flatten()
    statistic1, pvalue = ks_2samp(obs_urbandensity1,simulation_urbandensity1)
    return (statistic0 + statistic2 + statistic1) 

def calculate_distance(observed, simulated):
    o = observed.copy()
    s = simulated.copy()
    o[o == -200] = 0
    s[s == -200] = 0
    return ks_dis(o,s)

# For each FUA and Period:
#   Do 4 X 60 runs 
#   Calcute GOF
#   Select 60 best runs
#   Calculate percentage for each (soft classify)
#   Identify best fit (hard classify)

def run_model(before, after, param_set, seed):
    n_before = np.count_nonzero(before == 1);
    n_after = np.count_nonzero(after == 1);
    total_growth = int(n_after - n_before)
    rows, cols = before.shape
    # set the increment size (default = 15)
    growth_per_step = 15
    # apply the model
    final_map = CCA_EU.CCA_last_snapshot([param_set[0], 0, param_set[1]], [0, param_set[2], param_set[3]],
                                         seed = seed, landmap = before.copy(),
                                         rows = rows, cols = cols, urban_num = total_growth,
                                         trans_num = growth_per_step)
    
    gof = calculate_distance(final_map, after) 
    return gof

def run_period_single_fua(before, after):
        gof = []
        for s in range(4):
            param_sets, seeds = get_random_parameters_for_scenario(s,60)
            for p in range(60):
                gof.append(run_model(before, after, param_sets[p], seeds[p]))

        counts, bins = np.histogram(np.floor(np.argsort(gof)[:60]/60), range(5))  
        soft_class = counts / 60
        soft_class_dispersed = soft_class[3]
        hard_class = np.argmax(counts)
        return hard_class, soft_class, soft_class_dispersed

def run_genesis_single_fua(after):
    before = after.copy();
    before[before==1] = 0;
    return run_period_single_fua(before, after)

def run_period(fua_row, before_str, after_str):
    before = fua_row[before_str + ' urban raster'];
    after = fua_row[after_str + ' urban raster'];
    hard_class, soft_class, soft_class_dispersed = run_period_single_fua(before, after);
    
    return {before_str + '-' + after_str + ' dominant mode': hard_class,
            before_str + '-' + after_str + ' dispersed': soft_class_dispersed}


def run_genesis(fua_row, after_str):
    after = fua_row[after_str + ' urban raster'];
    hard_class, soft_class, soft_class_dispersed = run_genesis_single_fua(after);
    return {'0-' + after_str + ' dominant mode': hard_class,
            '0-' + after_str + ' dispersed': soft_class_dispersed}

def run_period_multiple_fua(dataframe, before_str, after_str):
    applied_df = dataframe.progress_apply(lambda row: run_period(row, before_str, after_str), axis='columns', result_type='expand')
    dataframe = pd.concat([dataframe, applied_df], axis='columns')
    return dataframe 

def run_genesis_multiple_fua(dataframe, after_str):
    applied_df = dataframe.progress_apply(lambda row: run_genesis(row, after_str), axis='columns', result_type='expand')
    dataframe = pd.concat([dataframe, applied_df], axis='columns')
    return dataframe     
    
    
ghs_built_raster_paths = [
    r'..\data\GHS_BUILT_LDS1975_GLOBE_R2018A_54009_250_V2_0\GHS_BUILT_LDS1975_GLOBE_R2018A_54009_250_V2_0.tif',
    r'..\data\GHS_BUILT_LDS1990_GLOBE_R2018A_54009_250_V2_0\GHS_BUILT_LDS1990_GLOBE_R2018A_54009_250_V2_0.tif',
    r'..\data\GHS_BUILT_LDS2000_GLOBE_R2018A_54009_250_V2_0\GHS_BUILT_LDS2000_GLOBE_R2018A_54009_250_V2_0.tif',
    r'..\data\GHS_BUILT_LDS2014_GLOBE_R2018A_54009_250_V2_0\GHS_BUILT_LDS2014_GLOBE_R2018A_54009_250_V2_0.tif']

fua_path = r'..\data\GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0\GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg'

nodata_value = -200


# List of countries of interest
countries = np.loadtxt('parameters\europe_countries.txt', dtype=str).tolist()

# Read the FUA file with global FUA's
fua_gdf = gpd.read_file(fua_path)

# Restrict dataset to countries and columns of interest
countries_gpd = fua_gdf[fua_gdf['Cntry_name'].isin(countries)]
countries_gpd = countries_gpd[['eFUA_name','Cntry_name','FUA_area','UC_area','geometry']]
countries_gpd = add_raster_to_dataframe(countries_gpd, ghs_built_raster_paths[0],'1975 urban')
countries_gpd = add_raster_to_dataframe(countries_gpd, ghs_built_raster_paths[1],'1990 urban')
countries_gpd = add_raster_to_dataframe(countries_gpd, ghs_built_raster_paths[2],'2000 urban')
countries_gpd = add_raster_to_dataframe(countries_gpd, ghs_built_raster_paths[3],'2014 urban')

# Remove missing data
countries_gpd.dropna(inplace=True)

# Pre-calculate urban growth quantities
countries_gpd['90-00 UG'] = countries_gpd['2000 urban'] - countries_gpd['1990 urban'] 
countries_gpd['00-14 UG'] = countries_gpd['2014 urban'] - countries_gpd['2000 urban'] 
countries_gpd['75-90 UG'] = countries_gpd['1990 urban'] - countries_gpd['1975 urban'] 

#Keep only the FUAs with urban growth
countries_gpd = countries_gpd[(countries_gpd['75-90 UG']>0)
                              &(countries_gpd['90-00 UG']>0)
                              &(countries_gpd['00-14 UG']>0)]
                              
                              # Reading the parameters for each scenario, this is 
# This cell is doing extensive analysis: 240 model runs for all FUA and for four periods. This may take days.

print("0-1975")
countries_gpd = run_genesis_multiple_fua(countries_gpd,'1975')
print("1975-1990")
countries_gpd = run_period_multiple_fua(countries_gpd,'1975','1990')
print("1990-2000")
countries_gpd = run_period_multiple_fua(countries_gpd,'1990','2000')
print("2000-2014")
countries_gpd = run_period_multiple_fua(countries_gpd,'2000','2014')

# After all this work, store results as a pickly for post-processing
countries_gpd.to_pickle("europe_results_countries_gpd.pkl")
