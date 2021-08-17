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
popsize = 40
maxIter = 500//40
convergence = []
for i in range(10):
        mindict = differential_evolution(f0,
                                        xmin,
                                        xmax,
                                        maxIter=maxIter,
                                        popsize=popsize,
                                        returnDict=True,
                                        storeIterValues=True
                                )



        Xde = mindict["x"]
        fitnessArray = mindict["fHistory"]
        convergence.append(fitnessArray)
        listXde.append(Xde)

listXde = np.array(listXde)

mindict_local = Powell(f0,Xde,xmin,xmax,constraints=cons,returnDict=True,tol=1e-6)
for si in mindict_local :
        print(si," : ",mindict_local[si])

xlocal = mindict_local['x']
flocal = mindict_local["fmin"]

convergence = [ np.log10(np.abs(np.array(ci,dtype=float)-flocal)/np.abs(flocal)) for ci in convergence ]
## Graphe


n = 150
x = np.linspace(-75,75,n)
y = np.linspace(-75,75,n)
X,Y = np.meshgrid(x,y)
Z = np.zeros((2,n**2))
Z[0] = X.flatten()
Z[1] = Y.flatten()
W = (f0(Z)).reshape((n,n))

figContour = plt.figure("Contour",figsize=(12,4))

plt.subplot(121)
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
plt.title("Fonction 'eggholder'",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)

plt.subplot(122)
for ci in convergence : 
        plt.plot(ci,'b-',lw=1.0)
plt.grid(True)
plt.title('Convergence de la solution',fontsize=12)
plt.xlabel("Appels de fonction")
plt.ylabel("Log10 erreur")


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