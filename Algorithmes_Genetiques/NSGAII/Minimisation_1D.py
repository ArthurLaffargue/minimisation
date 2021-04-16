import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from PyGen import optiBiAG
plt.rc("font",family='Serif')

f1 = lambda x : -( (x+2)**2 - 10 )
f2 = lambda x : -( (x-2)**2 + 20 )





## Méthode epsilon contrainte
npop = 20
ngen = npop*3
front_size = 100
xmin,xmax = [-10],[10]

minAg = optiBiAG(f1,f2,xmin,xmax)
pop,front_f1,front_f2 = minAg.optimize(npop,ngen,nfront=front_size)

## Graphes
fig0 = plt.figure(0,figsize=(8,4))
xvec = np.linspace(-10,10,500)
plt.plot(xvec,-f1(xvec),label="f1(x)",c='b')
plt.plot(xvec,-f2(xvec),label="f2(x)",c='r')
# plt.plot(x_f2max,-f1_min,'bo')
# plt.plot(x_f1max,-f2_min,'ro')
# plt.plot(x_f1max,-f1_max,'bs')
# plt.plot(x_f2max,-f2_max,'rs')
#
# plt.plot(front_x,front_f1,'xb')
# plt.plot(front_x,front_f2,'xr')
plt.grid(True)
plt.xlabel("x",fontsize=12)
plt.ylabel("y",fontsize=12)
plt.legend(fontsize=12)
plt.title('Fonctions objectifs',fontsize=14)
fig0.tight_layout()

plt.savefig("Courbes_1D.svg",dpi=300)



fig1= plt.figure(1)

plt.plot(-front_f1,-front_f2,ls='',marker='o',markerfacecolor='r',markeredgecolor='k')
plt.grid(True)
plt.xlabel("f1",fontsize=12)
plt.ylabel("f2",fontsize=12)
plt.title('Optimisation bi-critères : implémentation NSGA',fontsize=14)
fig1.tight_layout()
plt.savefig("Front_pb1.svg",dpi=300)



plt.show()