import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize,dual_annealing
plt.rc('font',family='Serif')
## Fonction objectif
f0 = lambda x : (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

xmin = [-75,-75]
xmax = [75,75]
## Optimisation

import sys
sys.path.append("..")
from _differential_evolution import differential_evolution
from _minimize_NelderMead import *
from _minimize_Powell import *


cons = []
listXde = []
maxIter = 500
for i in range(10):
        mindict = differential_evolution(f0,
                                        xmin,
                                        xmax,
                                        maxIter=maxIter,
                                        returnDict=True,
                                        storeIterValues=True
                                )



        Xde = mindict["x"]
        fitnessArray = mindict["fHistory"]
        listXde.append(Xde)

listXde = np.array(listXde)

mindict_local = Powell(f0,Xde,xmin,xmax,constraints=cons,returnDict=True,tol=1e-6)
for si in mindict_local :
        print(si," : ",mindict_local[si])

xlocal = mindict_local['x']

## Graphe


n = 150
x = np.linspace(-75,75,n)
y = np.linspace(-75,75,n)
X,Y = np.meshgrid(x,y)
Z = np.zeros((2,n**2))
Z[0] = X.flatten()
Z[1] = Y.flatten()
W = (f0(Z)).reshape((n,n))

figContour = plt.figure("Contour")
contour = plt.contour(X,Y,W,levels=np.linspace(W.min(),W.max(),25))
plt.clabel(contour)


plt.plot(listXde[:,0],listXde[:,1],label='Solution DE',
        marker='o',
        ls='',
        markeredgecolor='k',
        markerfacecolor="r",
        alpha=0.8)

plt.plot(xlocal[0],xlocal[1],
        label='Solution locale',
        marker='D',
        ls='',
        markeredgecolor='k',
        markerfacecolor="c")

plt.xlim(-75,75)
plt.ylim(-75,75)
plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.title("Problème 'eggholder' : évolution différentielle",fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("figure.svg",dpi=300)


plt.figure(figsize=(8,4))
plt.plot(fitnessArray,label='fmin',marker='o',ls='--',markerfacecolor="orange",alpha=0.5)
plt.grid(True)
plt.xlabel("Nombre itération")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)

plt.show()