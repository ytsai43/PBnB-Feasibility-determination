
# coding: utf-8

# In[1]:

import numpy as np
import math
def Sinusoidal(x):  
    dim = len(x)
    first_term = 1
    second_term = 1
    for i in range(dim):
        first_term *= math.sin(math.pi*float(x[i])/180)
        second_term *= math.sin(math.pi*float(x[i])/36) 
    fx = -2.5*first_term - second_term
    if fx <= -2.3:
        distant_con1 = -2.3-(fx)  #positive value; feasible
    else:
        distant_con1 = (-2.3)-fx  #negtive value; Infeasible

    return [fx,np.array([distant_con1])]


# In[ ]:




# In[ ]:



