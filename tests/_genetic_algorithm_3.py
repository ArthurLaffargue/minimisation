import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rc('font',family='Serif')
## Fonction objectif
f0 = lambda x : ( x[0]**2 + x[1]**2 - x[0]*x[1]  + x[1] )
f = lambda x : -f0(x)

xmin = [-2,-2]
xmax = [2,2]
## Optimisation

import sys
sys.path.append("..")
from _genetic_algorithm import optimizeMonoAG

npop = 40
ngen = 100
minAg = optimizeMonoAG(f,xmin,xmax)
Xag,Yag = minAg.optimize(npop,ngen,verbose=False)
fitnessArray = minAg.getStatOptimisation()
lastPop = minAg.getLastPopulation()

## SCIPY
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,startX,bounds=bounds)
xScipy = res.x
## Graphe


n = 150
x = np.linspace(-2,2,n)
y = np.linspace(-2,2,n)
X,Y = np.meshgrid(x,y)
Z = np.zeros((2,n**2))
Z[0] = X.flatten()
Z[1] = Y.flatten()
W = (f(Z)).reshape((n,n))


figContour = plt.figure("Contour")
contour = plt.contour(X,Y,W,levels=np.linspace(W.min(),W.max(),25),cmap = 'PuBuGn')

plt.plot(lastPop[:,0],lastPop[:,1],'kx')
plt.plot(xScipy[0],xScipy[1],
        label='Solution SCIPY',
        marker='D',
        ls='',
        markeredgecolor='k',
        markerfacecolor="c")
plt.plot(Xag[0],Xag[1],label='Solution AG',
        marker='o',
        ls='',
        markeredgecolor='k',
        markerfacecolor="r")
plt.clabel(contour)
plt.xlabel("x",fontsize=14)
plt.ylabel("y",fontsize=14)
plt.title("Optimisation bi-variables",fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("figure.svg",dpi=300)



figFitness = plt.figure('Fitness')
plt.plot(list(range(len(fitnessArray))),fitnessArray,label='fmin')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.show()