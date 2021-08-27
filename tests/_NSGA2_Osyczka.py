import numpy as np
import matplotlib.pyplot as plt


## Probleme

f1 = lambda x : ( -25*(x[0]-2)**2 - (x[1]-2)**2 - (x[2]-1)**2 - (x[3]-4)**2 - (x[4]-1)**2  )
f2 = lambda x : (  x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2)

g1 = lambda x : x[0] + x[1] - 2
g2 = lambda x : 6-x[0]-x[1]
g3 = lambda x : 2-x[1]+x[0]
g4 = lambda x : 2-x[0]+3*x[1]
g5 = lambda x : 4-(x[2]-3)**2-x[3]
g6 = lambda x : (x[4]-3)**2+x[5]-4

cons = [{"type" : "ineq","fun":g1},
        {"type" : "ineq","fun":g2},
        {"type" : "ineq","fun":g3},
        {"type" : "ineq","fun":g4},
        {"type" : "ineq","fun":g5},
        {"type" : "ineq","fun":g6}]
## Optimisation

import sys
sys.path.append("..")
from _genetic_algorithm import continousBiObjective_NSGA

npop = 100
ngen = 1000
front_size = None
xmin,xmax = [0,0,1,0,1,0],[10,10,5,6,5,10]


nsga_instance = continousBiObjective_NSGA(f1,
                                        f2,
                                        xmin,
                                        xmax,
                                        constraints=cons,
                                        func1_criterion='min',
                                        func2_criterion='min',
                                        penalityFactor=1E6,
                                        # sharing_distance=1/npop,
                                        constraintMethod="penality")

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





