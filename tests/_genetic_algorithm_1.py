import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family="Serif")
## Fonction objectif
f = lambda x : ( (1-2*x)**2 + x**2 - x*(1-2*x) + x )*np.sin(2*np.pi*x)

x = np.linspace(-2,2,500)
y = f(x)


## Optimisation

import sys
sys.path.append("..")
from _genetic_algorithm import realSingleObjectiveGA

npop = 25
ngen = npop*3
ga_instance = realSingleObjectiveGA(f,[-2],[2])
Xag,Yag = ga_instance.minimize(npop,ngen,verbose=False,returnDict=False)
fitnessArray = ga_instance.getStatOptimisation()
lastPop = ga_instance.getLastPopulation()


## Graphe

figure1 = plt.figure(1,figsize=(8,3))

plt.plot(x,y,color='k',label="y = f(x)")
plt.plot(lastPop,f(lastPop),'bx')
plt.plot(Xag,Yag,
        label="Solution opitmisation",
        marker='o',
        ls='',
        markeredgecolor='k',
        markerfacecolor="r")


plt.title("Optimisation mono-variable",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig("figure3D.svg",dpi=300)
plt.show()