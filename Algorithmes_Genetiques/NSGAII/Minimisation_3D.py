import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from PyGen import optiBiAG
plt.rc("font",family="Serif")


f1 = lambda x : -( -10*np.exp(-0.2*np.sqrt(x[0]**2+x[1]**2)) - \
                    10*np.exp(-0.2*np.sqrt(x[1]**2+x[2]**2)) )
f2 = lambda x : -( np.abs(x[0])**0.8 + 5*np.sin(x[0]**3) +\
                np.abs(x[1])**0.8 + 5*np.sin(x[1]**3)  + \
                np.abs(x[2])**0.8 + 5*np.sin(x[2]**3))


npop = 20*10
ngen = npop*10
front_size = 450
xmin,xmax = [-5,-5,-5],[5,5,5]
## Méthode epsilon contrainte

minAg = optiBiAG(f1,f2,xmin,xmax)
pop,front_f1,front_f2 = minAg.optimize(npop,ngen,nfront=front_size,verbose=True)

front_f1 = -front_f1
front_f2 = -front_f2

data = np.array([front_f1,front_f2]).T
np.savetxt("kursaweNSGAII.txt",data)
## Optimisation bicritères
fig1= plt.figure(1)

plt.plot(front_f1,front_f2,ls='',marker='o',markerfacecolor='r',markeredgecolor='k')
plt.grid(True)
plt.xlabel("f1",fontsize=12)
plt.ylabel("f2",fontsize=12)
plt.title('Optimisation bi-critères : implémentation NSGA',fontsize=14)
fig1.tight_layout()
plt.savefig("Front_pb4.svg",dpi=300)

plt.show()