import numpy as np
import matplotlib.pyplot as plt


## Probleme

f1 = lambda x : ( -10*np.exp(-0.2*np.sqrt(x[0]**2+x[1]**2)) - \
                    10*np.exp(-0.2*np.sqrt(x[1]**2+x[2]**2)) )
f2 = lambda x : ( np.abs(x[0])**0.8 + 5*np.sin(x[0]**3) +\
                np.abs(x[1])**0.8 + 5*np.sin(x[1]**3)  + \
                np.abs(x[2])**0.8 + 5*np.sin(x[2]**3))

cons = []
## Optimisation

import sys
sys.path.append("..")
from _genetic_algorithm import realBiObjective_NSGA2

npop = 20*5
ngen = npop*5
front_size = None
xmin,xmax = [-5,-5,-5],[5,5,5]


nsga_instance = realBiObjective_NSGA2(f1,
                                    f2,
                                    xmin,
                                    xmax,
                                    constraints=cons,
                                    func1_criterion='min',
                                    func2_criterion='min')

xfront,f1front,f2front = nsga_instance.optimize(npop,ngen,
                                                nfront=front_size,
                                                verbose=True)



## Graphiques
fig1= plt.figure(1)

plt.plot(f1front,f2front,ls='',marker='o',markerfacecolor='r',markeredgecolor='k')
plt.grid(True)
plt.xlabel("f1",fontsize=12)
plt.ylabel("f2",fontsize=12)
plt.title('Optimisation bi-critères : implémentation NSGA',fontsize=14)
fig1.tight_layout()
plt.savefig("Front_pb4.svg",dpi=300)

plt.show()





