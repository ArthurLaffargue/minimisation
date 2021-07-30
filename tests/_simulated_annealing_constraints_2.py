import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rc('font',family='Serif')
## Fonction objectif
epsilon = 80
f0 = lambda x : (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))
c1 = lambda x : -x[0] + 2*x[1] + 0.01*x[1]**2 - 1 
c2 = lambda x : -c1(x) + 2*epsilon

xmin = [-75,-75]
xmax = [75,75]
## Optimisation

import sys
sys.path.append("..")
from _simulated_annealing import minimize_simulatedAnnealing

cons = [{'type': 'ineq', 'fun': c1},
       {'type': 'ineq', 'fun': c2}]

maxIter = 2000
mindict = minimize_simulatedAnnealing(f0,
                                    xmin,
                                    xmax,
                                    maxIter=maxIter,
                                    autoSetUpIter=100,
                                    returnDict=True,
                                    constraints=cons
                        )
Xsa = mindict["x"]

for si in mindict :
        print(si," : ",mindict[si])

## SCIPY
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,Xsa,bounds=bounds,constraints=cons)
xScipy = res.x



## Graphe


n = 300
x = np.linspace(-75,75,n)
y = np.linspace(-75,75,n)
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

plt.xlim(-75,75)
plt.ylim(-75,75)
plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.title("Optimisation bi-variables avec contrainte d'égalité",fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("figure.svg",dpi=300)

plt.show()