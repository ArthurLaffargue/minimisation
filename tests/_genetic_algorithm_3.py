import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
plt.rc('font',family='Serif')
## Fonction objectif

f0 = lambda x : (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

xmin = [-75,-75]
xmax = [75,75]
## Optimisation

import sys
sys.path.append("..")
from _genetic_algorithm import continousSingleObjectiveGA

npop = 40
ngen = 50
ga_instance = continousSingleObjectiveGA(f0,xmin,xmax)
Xag,Yag = ga_instance.minimize(npop,ngen,verbose=False)
fitnessArray = ga_instance.getStatOptimisation()
lastPop = ga_instance.getLastPopulation()

## SCIPY
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,Xag,bounds=bounds)
xScipy = res.x
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
contour = plt.contour(X,Y,W,levels=np.linspace(W.min(),W.max(),25))#,cmap = 'PuBuGn')

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
plt.clabel(contour)
plt.xlabel("x",fontsize=14)
plt.ylabel("y",fontsize=14)
plt.title("Optimisation bi-variables",fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)




plt.figure(figsize=(8,4))
plt.plot(np.array(list(range(ngen)))*npop,fitnessArray,label='fmin',marker='o',ls='--',markerfacecolor="orange",alpha=0.5)
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
                        cmap="terrain",
                        antialiased=True,
                        # linewidths=0.2,
                        # edgecolor="k",
                        alpha=1.0
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