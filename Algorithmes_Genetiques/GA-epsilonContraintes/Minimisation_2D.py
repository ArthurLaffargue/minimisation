import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from PyGen import optiMonoAG
plt.rc('font',family='Serif')

f1 = lambda x : -( 4*x[0]**2 + 4*x[1]**2 )
f2 = lambda x : -( (x[0]-5)**2 + (x[1]-5)**2  )





## Méthode epsilon contrainte
npop = 20*2
ngen = npop*4
front_size = 50
xmin,xmax = [0,0],[5,3]

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

fig1= plt.figure(1)

plt.plot(front_f1,front_f2,ls='',marker='o',markerfacecolor='r',markeredgecolor='k')
plt.grid(True)
plt.xlabel("f1",fontsize=12)
plt.ylabel("f2",fontsize=12)
plt.title('Optimisation bi-critères : Epsilon contrainte',fontsize=14)
fig1.tight_layout()
plt.savefig("Front_pb2.svg",dpi=300)

plt.show()