
import numpy as np
from scipy.optimize import minimize
import sys
import time
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')


sys.path.append("..")
from _minimize_gradient import *
from _minimize_BFGS import *
from _minimize_Powell import *
from _minimize_NelderMead import *

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
dictgrad = conjugateGradient(rosen,x0,xmin,xmax,
                            gf = jac,
                            returnDict=True,
                            storeIterValues=True,
                            methodConjugate="",
                            maxIter=2000,
                            constraints=cons)

dictconjgrad = conjugateGradient(rosen,x0,xmin,xmax,
                            gf = jac,
                            returnDict=True,
                            storeIterValues=True,
                            methodConjugate="PR",
                            penalityFactor=100.0,
                            constraints=cons)

dictbfgs = BFGS(rosen,x0,xmin,xmax,
                            gf = jac,
                            returnDict=True,
                            storeIterValues=True,
                            penalityFactor=100.0,
                            constraints=cons)


dictPowell = Powell(rosen,x0,xmin,xmax,
                            returnDict=True,
                            storeIterValues=True,
                            constraints=cons)

dictNelderMead = NelderMead(rosen,x0,xmin,xmax,
                            returnDict=True,
                            storeIterValues=True,
                            constraints=cons)
end = time.time()

xgrad = dictgrad["xHistory"]
xconjgrad = dictconjgrad["xHistory"]
xbfgs = dictbfgs["xHistory"]
xPowell = dictPowell["xHistory"]
xNelderMead = dictNelderMead["xHistory"]

for s in dictgrad :
    if not( s.endswith("History") ):
        print(s," : ",dictgrad[s])

print("\n")
for s in dictconjgrad :
    if not( s.endswith("History") ):
        print(s," : ",dictconjgrad[s])

print("\n")
for s in dictbfgs :
    if not( s.endswith("History") ):
        print(s," : ",dictbfgs[s])

print("\n")
for s in dictPowell :
    if not( s.endswith("History") ):
        print(s," : ",dictPowell[s])

print("\n")
for s in dictNelderMead :
    if not( s.endswith("History") ):
        print(s," : ",dictNelderMead[s])






