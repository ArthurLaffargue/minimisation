import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from PyGen import optiBiAG
plt.rc("font",family='Serif')


f1 = lambda x : -( 4*x[0]**2 + 4*x[1]**2 )
f2 = lambda x : -( (x[0]-5)**2 + (x[1]-5)**2  )


g1 = lambda x : 25 - (x[0]-5)**2 - x[1]**2
g2 = lambda x : (x[0]-8)**2 + (x[1]+3)**2 - 7.7

cons0 = [{"type" : "ineq","fun":g1},
        {"type" : "ineq","fun":g2}]
## Méthode epsilon contrainte
npop = 20*2
ngen = npop*3
front_size = 100
xmin,xmax = [0,0],[5,3]

minAg = optiBiAG(f1,f2,xmin,xmax,constraints=cons0)
minAg.setConstraintMethod("feasibility")
pop,front_f1,front_f2 = minAg.optimize(npop,ngen,nfront=front_size)

front_f1 = -front_f1
front_f2 = -front_f2
## Optimisation bicritères

fig1= plt.figure(1)

plt.plot(front_f1,front_f2,ls='',marker='o',markerfacecolor='r',markeredgecolor='k')
plt.grid(True)
plt.xlabel("f1",fontsize=12)
plt.ylabel("f2",fontsize=12)
plt.title('Optimisation bi-critères : implémentation NSGA',fontsize=14)
fig1.tight_layout()
plt.savefig("Front_pb2.svg",dpi=300)

plt.show()