import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from matplotlib import cm
plt.rc('font',family='Serif')
## Fonction objectif

f0 = lambda x : (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))
fc1 = lambda x : -(0.001*x**3-x)
c0 = lambda x : x[1] - fc1(x[0])

xmin = np.array([-75,-75])
xmax = np.array([75,75])
## Optimisation

import sys
sys.path.append("..")
from _genetic_algorithms import continousSingleObjectiveGA

cons = [{'type': 'eq', 'fun': c0}]

npop = 75
ngen = 10000//npop

ga_instance = continousSingleObjectiveGA(f0,xmin,xmax,cons)
ga_instance.setPenalityParams(constraintAbsTol=0.1,penalityFactor=100,penalityGrowth=1.0)
ga_instance.setConvergenceCriteria(stagnationThreshold=100)
ga_instance.setElitisme(True)


listXga = []
xopt = None
yopt = None
for i in range(10):
        print("#RESOLVE : ",i)
        Xag,Yag = ga_instance.minimize(npop,ngen,verbose=False)
        listXga.append(Xag)

        if yopt is None : 
                xopt = Xag
                yopt = Yag
        elif yopt>Yag : 
                xopt = Xag
                yopt = Yag

fitnessArray = ga_instance.getStatOptimisation()
lastPop = ga_instance.getLastPopulation()
listXga = np.array(listXga)
## SCIPY
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,Xag,bounds=bounds,constraints=cons)
xScipy = res.x


distance_opt = np.mean(np.sqrt(np.sum( ((xopt-listXga)/(xmax-xmin))**2,axis=1 )))

print("ACCURACY %.3f"%( (1-distance_opt)*100) )
## Graphe


n = 300
x = np.linspace(-75,75,n)
y = np.linspace(-75,75,n)
X,Y = np.meshgrid(x,y)
Z = np.zeros((2,n**2))
Z[0] = X.flatten()
Z[1] = Y.flatten()
W = (f0(Z)).reshape((n,n))

figContour = plt.figure("Contour")
contour = plt.contour(X,Y,W,levels=np.linspace(W.min(),W.max(),25),cmap = 'PuBuGn')
plt.plot(x,fc1(x),color='k',label='Contrainte')
plt.clabel(contour)

plt.plot(lastPop[:,0],lastPop[:,1],'kx')
plt.plot(xScipy[0],xScipy[1],
        label='Solution SCIPY',
        marker='D',
        ls='',
        markeredgecolor='k',
        markerfacecolor="c")
plt.plot(listXga[:,0],listXga[:,1],label='Solution AG',
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

plt.figure(2,figsize=(8,4))
plt.plot(fitnessArray,label='fmin',marker='o',ls='--',markeredgecolor='k',markerfacecolor="y",color='grey')
plt.grid(True)
plt.xlabel("Nombre de générations")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)


figure2 = plt.figure(3)
ax1 = figure2.add_subplot(211)
ax2 = figure2.add_subplot(212) 

pts_curve = np.array([x,fc1(x)]).T
filtre_curve = (pts_curve[:,1] <= 75) & (pts_curve[:,1] >= -75)
pts_curve = pts_curve[filtre_curve]

Yag = [f0(xagi) for xagi in listXga]
f0_curve = [f0(xi) for xi in pts_curve]
ax1.plot(pts_curve[:,0],f0_curve,'.')
ax1.plot(listXga[:,0],Yag,"o")
ax2.plot(pts_curve[:,1],f0_curve,'.')
ax2.plot(listXga[:,1],Yag,"o")


plt.show()

