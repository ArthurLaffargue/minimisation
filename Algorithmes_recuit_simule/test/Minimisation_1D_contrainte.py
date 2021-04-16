import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')
## Fonction objectif
f = lambda x : np.sin(x) + np.sin(10/3*x) + 10*np.log10(x) - 0.84*x + 3
c1 = lambda x : (x-6)
x = np.linspace(2.7,7.5,500)
y = f(x)


## Optimisation

import sys
sys.path.append("..")
from SimulatedAnnealing import simulatedAnnealing
cons = [{'type': 'strictIneq', 'fun': c1}]

maxIter = 300
minSA = simulatedAnnealing(f,[2.7],[7.5],maxIter=maxIter,constraints=cons)
minSA.autoSetup(npermutations=100)
Xsa = []
for i in range(100) :
    Xsa.append(minSA.minimize())
Xsa = np.array(Xsa)
Ysa = f(Xsa)


## Graphe

figure1 = plt.figure(1,figsize=(8,3))

plt.plot(x,y,color='k',label="y = f(x)")
plt.plot(Xsa,Ysa,
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