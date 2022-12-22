# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:14:47 2022

@author: Jingyan Yu jingyan.yu@surrey.ac.uk; Alex Hagen-Zanker a.hagen-zanker@surrey.ac.uk
"""

import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:14:47 2022

@author: Jingyan Yu jingyan.yu@surrey.ac.uk; Alex Hagen-Zanker a.hagen-zanker@surrey.ac.uk
"""

import numpy as np
from fuzzy_kappa import fuzzy_kappa_simulation

# this function only kept to not break existing code
def FKS(map_a_before, map_b_before, map_a_after, map_b_after, mask,R,change_categories=[0,1,2],beta=1.0):
  similarity_matrix = np.array(([1, 0],[0,1])) 
  f = lambda d: np.exp(-beta * d)
  fks, sim = fuzzy_kappa_simulation(map_a_before, map_b_before, map_a_after, map_b_after, mask, similarity_matrix, f)
  return fks

