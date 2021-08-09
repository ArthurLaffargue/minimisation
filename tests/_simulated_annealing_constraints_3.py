import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rc('font',family='Serif')
## Fonction objectif
f0 = lambda x : (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))
fc1 = lambda x : -(0.001*x**3-x)
c0 = lambda x : x[1] - fc1(x[0])

xmin = [-75,-75]
xmax = [75,75]
## Optimisation

import sys
sys.path.append("..")
from _simulated_annealing import minimize_simulatedAnnealing

cons = [{'type': 'eq', 'fun': c0}]

maxIter = 10000
listXsa = []
for i in range(10):
        print("#SOLVE : ",i)
        mindict = minimize_simulatedAnnealing(f0,
                                        xmin,
                                        xmax,
                                        maxIter=maxIter,
                                        autoSetUpIter=100,
                                        constraintAbsTol=0.1,
                                        penalityFactor=100,
                                        returnDict=True,
                                        constraints=cons
                                )
        Xsa = mindict["x"]
        listXsa.append(Xsa)

listXsa = np.array(listXsa)

for si in mindict :
        print(si," : ",mindict[si])


## SCIPY
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,Xsa,bounds=bounds,constraints=cons)
xScipy = res.x

## Graphe


n = 150
x = np.linspace(-75,75,n)
y = np.linspace(-75,75,n)
X,Y = np.meshgrid(x,y)
Z = np.zeros((2,n**2))
Z[0] = X.flatten()
Z[1] = Y.flatten()
# W = (f0(Z)*(c1(Z)>=0)*(c2(Z)>=0)).reshape((n,n))
W = (f0(Z)).reshape((n,n))

figContour = plt.figure("Contour")
contour = plt.contour(X,Y,W,levels=np.linspace(W.min(),W.max(),25))
plt.plot(x,fc1(x),color='k',label='Contrainte')
plt.clabel(contour)

plt.plot(xScipy[0],xScipy[1],
        label='Solution SCIPY',
        marker='D',
        ls='',
        markeredgecolor='k',
        markerfacecolor="c")
plt.plot(listXsa[:,0],listXsa[:,1],label='Solution SA',
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




figure2 = plt.figure(3)
ax1 = figure2.add_subplot(211)
ax2 = figure2.add_subplot(212) 

pts_curve = np.array([x,fc1(x)]).T
filtre_curve = (pts_curve[:,1] <= 75) & (pts_curve[:,1] >= -75)
pts_curve = pts_curve[filtre_curve]
Ysa = [f0(Xsa_i) for Xsa_i in listXsa]

f0_curve = [f0(xi) for xi in pts_curve]
ax1.plot(pts_curve[:,0],f0_curve,'.')
ax1.plot(listXsa[:,0],Ysa,"o")
ax2.plot(pts_curve[:,1],f0_curve,'.')
ax2.plot(listXsa[:,1],Ysa,"o")


plt.show()
plt.show()