import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rc('font',family='Serif')
## Fonction objectif
epsilon = 1
f0 = lambda x : ( x[0]**2 + x[1]**2 - x[0]*x[1]  + x[1] )
f = lambda x : -f0(x)
c1 = lambda x : x[0] + 2*x[1] - 1 - epsilon
c2 = lambda x : -c1(x) + 2*epsilon

xmin = [-2,-2]
xmax = [2,2]
## Optimisation

import sys
sys.path.append("..")
from _genetic_algorithm import optimizeMonoAG

cons = [{'type': 'ineq', 'fun': c1},
       {'type': 'ineq', 'fun': c2}]

npop = 60
ngen = npop*3
minAg = optimizeMonoAG(f,xmin,xmax,cons)
Xag,Yag = minAg.optimize(npop,ngen,verbose=False)
fitnessArray = minAg.getStatOptimisation()
lastPop = minAg.getLastPopulation()

## SCIPY
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,startX,bounds=bounds,constraints=cons)
xScipy = res.x



## Graphe


n = 150
x = np.linspace(-2,2,n)
y = np.linspace(-2,2,n)
X,Y = np.meshgrid(x,y)
Z = np.zeros((2,n**2))
Z[0] = X.flatten()
Z[1] = Y.flatten()
W = (f(Z)*(c1(Z)>=0)*(c2(Z)>=0)).reshape((n,n))


figContour = plt.figure("Contour")
contour = plt.contour(X,Y,W,levels=np.linspace(W.min(),W.max(),25),cmap = 'PuBuGn')
plt.clabel(contour)
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

plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.title("Optimisation bi-variables avec contraintes",fontsize=14)
plt.grid(True)
plt.legend(fontsize=12,loc=0)
plt.savefig("figure.svg",dpi=300)

plt.figure(figsize=(8,4))
plt.plot(fitnessArray,label='fmin',marker='o',ls='--',markeredgecolor='k',markerfacecolor="y",color='grey')
plt.grid(True)
plt.xlabel("Nombre de générations")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)
plt.show()