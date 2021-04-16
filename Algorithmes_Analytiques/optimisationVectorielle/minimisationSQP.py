
import numpy as np
from scipy.optimize import minimize
import sys
import time
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')


sys.path.append("..")
from _minimize_SQP_BFGS import *

x0 = np.array([0.5,1.25])
xmin = np.array([0.0,-0.5])
xmax = np.array([1.0,2.0])

p = 100
rosen = lambda x : (x[0]-1)**2 + p*(x[0]**2-x[1])**2
jac = lambda x : np.array([2.0*(x[0]-1)+4.0*p*x[0]*(x[0]**2-x[1]),
                           -2.0*p*(x[0]**2-x[1])],dtype=float)



c1 = lambda x : - x[0] - 2*x[1] + 1
dc1 = lambda x : np.array([-1.0,-2.0])

c2 = lambda x : -x[0]**2-x[1] + 1
dc2 = lambda x : np.array([-2.0*x[0],-1.0])

c3 = lambda x : -x[0]**2+x[1] + 1
dc3 = lambda x : np.array([-2.0*x[0],1.0])

c4 = lambda x : -2*x[0] - x[1] + 1
dc4 = lambda x : np.array([-2.0,-1.0])




cons = [{'type': 'ineq', 'fun': c1,"jac":dc1},
       {'type': 'ineq', 'fun': c2,"jac":dc2},
       {'type': 'ineq', 'fun': c3,"jac":dc3},
       {'type': 'eq', 'fun': c4,"jac":dc4}]



#------- GRADIENT CONJUGUE -------#
start = time.time()

dictmin = SQP_BFGS(rosen,x0,xmin,xmax,
                            gf = jac,
                            returnDict=True,
                            penalityFactor=1000.0,
                            constraints=cons)

end = time.time()


for s in dictmin :
    if not( s.endswith("History") ):
        print(s," : ",dictmin[s])







