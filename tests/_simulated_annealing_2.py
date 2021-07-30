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
from _simulated_annealing import simulatedAnnealing,minimize_simulatedAnnealing

cons = []

maxIter = 2000
mindict = minimize_simulatedAnnealing(f0,
                                    xmin,
                                    xmax,
                                    maxIter=maxIter,
                                    autoSetUpIter=100,
                                    returnDict=True,
                        )
Xsa = mindict["x"]

for si in mindict :
        print(si," : ",mindict[si])


## SCIPY
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,Xsa,bounds=bounds)
xScipy = res.x

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

plt.xlim(-75,75)
plt.ylim(-75,75)
plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.title("Optimisation bi-variables avec contrainte d'égalité",fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("figure.svg",dpi=300)

plt.show()