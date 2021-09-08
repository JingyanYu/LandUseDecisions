import CCA
import numpy as np
from numpy.random import multivariate_normal
from scipy.signal import fftconvolve
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
kernels=[CCA.kernel_expo_square(0.01,beta) for beta in [0.2,0.5,2.0]]

##Initialisation 4: Observation data
#Extracted 150*150 windows of Oxford, Swindon from 2000 CORINE data as numpy data
w_oxford = np.load('data\oxford.npy')
w_swindon = np.load('data\swindon.npy')
w_oxford_water=w_oxford.copy()
w_oxford_nonwater=w_oxford.copy()
w_swindon_water=w_swindon.copy()
w_swindon_nonwater=w_swindon.copy()
for w in [w_oxford_nonwater,w_swindon_nonwater]:
    w[w<10]=1
    w[w>=10]=0
for w in [w_oxford_water,w_swindon_water]:
    w[w<10]=1
    w[w==41]=2
    w[w>=10]=0

#Oxford2000: iter_num=164;trans_num=25
#Chain1 initial para [6.7333486,68.87145612,2.61832417,3.96224226]
#Chain2 initial para [6.67504623,78.6366322,2.00148601,5.39610628]
#Chain3 initial para [5.74713275,78.95402972,1.45394707,5.51829238]    
initial_landmap = w_oxford_water.copy()
initial_landmap[w_oxford_water!=2] = 0
# obs_densities_urban = np.sort(fftconvolve(w_oxford_nonwater,kernels[0],mode='same')[w_oxford_nonwater==1].flatten()) 


#Swindon2000 from blank:iter_num = 256;trans_num=21
#Chain1 initial para [10.4523679,80.7589637,2.19134939,1.02197193]
#Chain2 initial para [12.78305599,83.59569675,2.31424336,1.41159063]
#Chain3 initial para [10.60331248,80.86676804,2.14985003,1.29958185]
obs_densities_urban = np.sort(fftconvolve(w_swindon_nonwater,kernels[0],mode='same')[w_swindon_nonwater==1].flatten()) 



#goodness-of-fit measure - Kolmogorov–Smirnov statistic of two urban density distributions
def ks_large_urban(simulation,obs_density=obs_densities_urban,kernel=kernels[0]):
    
    sim_density = np.sort(fftconvolve(simulation,kernel,mode='same')[simulation==1].flatten())
    statistic, pvalue = ks_2samp(obs_density,sim_density)
    return statistic

#calibration method - Markov chain Monte Carlo Approximate Bayesian Computation for CCA parameter estimation
def mcmcabc(nsteps,initial_paras=[C1,C2,C3,C4],para_intervals=np.array([C1_interval,C2_interval,C3_interval,C4_interval]),
            initial_covmx=None, 
            kernels=kernels,rows=150,cols=150,iter_num=164,transition_num=25,seed=None,initial_landmap=None,
            initial_epsilon=3000):
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
    accepted_landpatterns = []
    
    
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
        sim_landmap = CCA.CCA_last_snapshot([uPara[0],0,uPara[1]],
                                            [0,nuPara[0],nuPara[1]],
                                            seed=seed,landmap=landmap,
                                            rows=rows,cols=cols,iter_num=iter_num,transition_num=transition_num) 
        
        distance = ks_large_urban(simulation=sim_landmap)
        if distance <= current_epsilon:
            print('accept: ',i,proposed_thetas)
            accepted_thetas.append(proposed_thetas)
            current_thetas = proposed_thetas
            consequent_reject_times = 0
            accepted_landpatterns.append(sim_landmap)
            accepted_distance.append(distance)
            
        else:
            print('reject: ',i,proposed_thetas,'Distance: ',distance)
            consequent_reject_times += 1

        if (i//500>0) and (i%500==0):
            covmx = np.cov(np.transpose(accepted_thetas))
            current_epsilon = np.median(accepted_distance[-20:])
    return accepted_thetas,accepted_landpatterns,accepted_distance



