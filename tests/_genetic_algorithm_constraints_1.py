import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
from _genetic_algorithm import continousSingleObjectiveGA

cons = [{'type': 'ineq', 'fun': c1},
       {'type': 'ineq', 'fun': c2}]

npop = 60
ngen = npop*3
ga_instance = continousSingleObjectiveGA(f0,xmin,xmax,cons)
Xag,Yag = ga_instance.minimize(npop,ngen,verbose=False)
fitnessArray = ga_instance.getStatOptimisation()
lastPop = ga_instance.getLastPopulation()

## SCIPY
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,Xag,bounds=bounds,constraints=cons)
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
contour = plt.contour(X,Y,W,levels=np.linspace(W.min(),W.max(),25))#,cmap = 'PuBuGn')
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



fig3D = plt.figure(figsize=(8,8))
ax3D = fig3D.gca(projection='3d')
# ax3D.set_axis_off()

surf = ax3D.plot_surface(X, Y, W, 
                        cmap=cm.coolwarm,
                        antialiased=True,
                        # linewidths=0.2,
                        # edgecolor="k",
                        alpha=0.7
                        )
ax3D.scatter3D(lastPop[:,0],
               lastPop[:,1],
               [f0(xi) for xi in lastPop],
               color='grey',
               marker='x',
               alpha=0.8,
               label='Last population')

ax3D.scatter3D(Xag[0],
               Xag[1],
               f0(Xag),
               color='r',
               marker='D',
               alpha=1.0,
               label='Solution AG')

ax3D.legend()
ax3D.view_init(5,170)

ax3D.set_xticklabels([])
ax3D.set_yticklabels([])
ax3D.set_zticklabels([])

ax3D.set_xlabel("X")
ax3D.set_ylabel("Y")
ax3D.set_zlabel("Z")

plt.tight_layout()
plt.savefig("figure.svg",dpi=300)
plt.grid(False)
plt.show()