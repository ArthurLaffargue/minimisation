import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from PyGen import optiMonoAG
plt.rc('font',family='Serif')

f1 = lambda x : -( (x+2)**2 - 10 )
f2 = lambda x : -( (x-2)**2 + 20 )





## Méthode epsilon contrainte
npop = 20
ngen = npop*3
front_size = 50
xmin,xmax = [-10],[10]

minAg = optiMonoAG(f1,xmin,xmax)
x_f1max,f1_max= minAg.optimize(npop,ngen,verbose=False)

minAg = optiMonoAG(f2,xmin,xmax)
x_f2max,f2_max= minAg.optimize(npop,ngen,verbose=False)

f1_min = f1(x_f2max)
f2_min = f2(x_f1max)

f2_contrainte = np.linspace(f2_min,f2_max,front_size+2)[1:-1:]
f1_contrainte = np.linspace(f1_min,f1_max,front_size+2)[1:-1:]

front_f1 = [f1_max,f1_min]
front_f2 = [f2_min,f2_max]
front_x = [x_f1max,x_f2max]


for k,ck in enumerate(f2_contrainte) :
    fc = lambda x : f2(x)-ck
    cons = [{"type" : "ineq","fun":fc}]
    minAg = optiMonoAG(f1,xmin,xmax,cons)
    try :
        xk,f1k = minAg.optimize(npop,ngen,verbose=False)
        f2k = f2(xk)

        front_f1.append(f1k)
        front_f2.append(f2k)
        front_x.append(xk)
    except : pass

for k,ck in enumerate(f1_contrainte) :
    fc = lambda x : f1(x)-ck
    cons = [{"type" : "ineq","fun":fc}]
    minAg = optiMonoAG(f2,xmin,xmax,cons)
    try :
        xk,f2k = minAg.optimize(npop,ngen,verbose=False)
        f1k = f1(xk)

        front_f1.append(f1k)
        front_f2.append(f2k)
        front_x.append(xk)
    except : pass

front_f1 = -np.array(front_f1)
front_f2 = -np.array(front_f2)
## Optimisation bicritères

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

plt.plot(front_f1,front_f2,ls='',marker='o',markerfacecolor='r',markeredgecolor='k')
plt.grid(True)
plt.xlabel("f1",fontsize=12)
plt.ylabel("f2",fontsize=12)
plt.title('Optimisation bi-critères : Epsilon contrainte',fontsize=14)
fig1.tight_layout()
plt.savefig("Front_pb1.svg",dpi=300)



plt.show()