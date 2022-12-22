# -*- coding: utf-8 -*-
"""
@author: Jingyan Yu jingyan.yu@surrey.ac.uk; Alex Hagen-Zanker a.hagen-zanker@surrey.ac.uk
"""
from fuzzy_kappa import fuzzy_kappa, fuzzy_kappa_simulation
import numpy as np

A1 = np.array(([0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,1,1,1,1,1,0,0,0],
               [0,0,0,1,1,0,1,0,0,0],
               [0,0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0]))

B1 = np.array(([0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,1,1,0],
               [0,0,0,0,0,1,1,1,1,0],
               [0,0,0,0,0,1,1,1,1,0],
               [0,0,0,0,0,0,0,0,0,0]))

mask = np.array(([1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1,1,1,1]))

similarity_matrix = np.array(([1, 0], [0, 1]))
beta = 1;
f = lambda d: np.exp(-beta * d)

fk, sim1 = fuzzy_kappa(A1,B1,mask,similarity_matrix,f)
fks, sim2 = fuzzy_kappa_simulation(A1*0, B1* 0, A1,B1,mask,similarity_matrix,f)

print(fk)
print(fks)
print(sim1)
print(sim2)

