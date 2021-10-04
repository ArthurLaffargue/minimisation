import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')
## Fonction objectif
f = lambda x : ( (1-2*x)**2 + x**2 - x*(1-2*x) + x )*np.sin(2*np.pi*x)
c = lambda x : -(1-2*x)
x = np.linspace(-2,2,500)
y = f(x)


## Optimisation

import sys
sys.path.append("..")
from _genetic_algorithms import realSingleObjectiveGA
cons = [{'type': 'strictIneq', 'fun': c}]

npop = 20
ngen = npop*3
ga_instance = realSingleObjectiveGA(f,[-2],[2],cons)
Xag,Yag = ga_instance.minimize(npop,ngen,verbose=False)
fitnessArray = ga_instance.getStatOptimisation()
lastPop = ga_instance.getLastPopulation()


## Graphe

figure1 = plt.figure(1,figsize=(8,3))

plt.plot(x,y,color='k',label="y = f(x)")
plt.plot(x,c(x),color='grey',lw=1.0,label="Contrainte c(x) > 0")
plt.plot(lastPop,f(lastPop),'bx')
plt.plot(Xag,Yag,
        label="Solution opitmisation",
        marker='o',
        ls='',
        markeredgecolor='k',
        markerfacecolor="r")


plt.title("Optimisation mono-variable sous contrainte",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig("figure.svg",dpi=300)
plt.show()