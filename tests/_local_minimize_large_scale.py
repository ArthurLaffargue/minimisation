from scipy.optimize import minimize
import numpy as np
import sys

sys.path.append("..")
from _minimize_gradient import *
from _minimize_BFGS import *
from _minimize_Powell import *
from _minimize_NelderMead import *

n = 50

def func(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def gradfunc(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

xmin,xmax=-2,2
x0 = -np.ones(n)
xmin = np.ones_like(x0)*xmin
xmax = np.ones_like(x0)*xmax



linesearchMethod = "wolfe"
#------- GRADIENT CONJUGUE -------#

dictconjgrad = conjugateGradient(func,x0,xmin,xmax,gf=gradfunc,
                            returnDict=True,
                            storeIterValues=True,
                            methodConjugate="PR",
                            maxIter=2000,
                            linesearchMethod=linesearchMethod)

dictbfgs = BFGS(func,x0,xmin,xmax,gf=gradfunc,
                            returnDict=True,
                            storeIterValues=True,
                            maxIter=2000,
                            linesearchMethod=linesearchMethod)


xconjgrad = dictconjgrad["xHistory"]
xbfgs = dictbfgs["xHistory"]

print("\n")
for s in dictconjgrad :
    if not( s.endswith("History") ):
        print(s," : ",dictconjgrad[s])

print("\n")
for s in dictbfgs :
    if not( s.endswith("History") ):
        print(s," : ",dictbfgs[s])

