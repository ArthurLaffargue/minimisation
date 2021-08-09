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
from _minimize_NelderMead import *
from _minimize_Powell import *


cons = []
listXsa = []
maxIter = 500
for i in range(10):
        mindict = minimize_simulatedAnnealing(f0,
                                        xmin,
                                        xmax,
                                        maxIter=maxIter,
                                        autoSetUpIter=100,
                                        returnDict=True,
                                        storeIterValues=True
                                )



        Xsa = mindict["x"]
        fitnessArray = mindict["fHistory"]
        listXsa.append(Xsa)

listXsa = np.array(listXsa)
# for si in mindict :
#         if not( si.endswith("History") ):
#                 print(si," : ",mindict[si])


## SCIPY
# bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
# startX = np.mean(bounds,axis=1)
# res = minimize(f0,Xsa,bounds=bounds)
# xlocal = res.x

mindict_local = Powell(f0,Xsa,xmin,xmax,constraints=cons,returnDict=True,tol=1e-6)
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


plt.plot(listXsa[:,0],listXsa[:,1],label='Solution SA',
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
plt.title("Problème 'eggholder' : recuit simulé",fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("figure.svg",dpi=300)


plt.figure(figsize=(8,4))
plt.plot(fitnessArray,label='fmin',marker='o',ls='--',markerfacecolor="orange",alpha=0.5)
plt.grid(True)
plt.xlabel("Nombre de générations")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)

plt.show()