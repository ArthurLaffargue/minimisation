import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rc('font',family='Serif')
## Fonction objectif
epsilon = 1
f0 = lambda x : ( 2*x[0]**2 + 1.33*x[1]**2 + x[1]*x[0]**2 + 0.25*x[0]*x[1]**2)
fc1 = lambda x : (0.85*x**2 - 1.33 -x)
c0 = lambda x : x[1] - fc1(x[0])
c1 = lambda x : c0(x) + epsilon
c2 = lambda x : -c0(x) + epsilon

xmin = [-2,-2]
xmax = [2,2]
## Optimisation

import sys
sys.path.append("..")
from _simulated_annealing import simulatedAnnealing

cons = [{'type': 'ineq', 'fun': c1},
       {'type': 'ineq', 'fun': c2}]

maxIter = 2000
minSA = simulatedAnnealing(f0,xmin,xmax,maxIter=maxIter,constraints=cons)
minSA.autoSetup(npermutations=100)
Xsa = minSA.minimize()
statistique = minSA.statMinimize

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
W = (f0(Z)*(c1(Z)>=0)*(c2(Z)>=0)).reshape((n,n))


figContour = plt.figure("Contour")
contour = plt.contour(X,Y,W,levels=np.linspace(W.min(),W.max(),25))
plt.clabel(contour)
plt.plot(xScipy[0],xScipy[1],
        label='Solution SCIPY',
        marker='D',
        ls='',
        markeredgecolor='k',
        markerfacecolor="c")
plt.plot(Xsa[0],Xsa[1],label='Solution SA',
        marker='o',
        ls='',
        markeredgecolor='k',
        markerfacecolor="r")

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.title("Optimisation bi-variables avec contrainte d'égalité",fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("figure.svg",dpi=300)

plt.figure(figsize=(8,4))
plt.plot(statistique[:,0]-f0(xScipy),
        label='fmin',
        marker='o',
        ls='--',
        markeredgecolor='k',
        markerfacecolor="y",
        color='grey')
plt.yscale("log")
plt.grid(True)
plt.xlabel("Nombre de générations")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)

plt.figure(figsize=(8,4))
plt.plot(statistique[:,1],label='Tempk',color='c')
plt.grid(True)
plt.xlabel("Nombre de générations")
plt.ylabel("Temperature")
plt.title("Paliers de temperatures")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("temperature.svg",dpi=300)
plt.show()