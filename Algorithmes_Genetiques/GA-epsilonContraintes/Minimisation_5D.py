import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from PyGen import optiMonoAG


f1 = lambda x : -( -25*(x[0]-2)**2 - (x[1]-2)**2 - (x[2]-1)**2 - (x[3]-4)**2 - (x[4]-1)**2  )
f2 = lambda x : -(  x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2)

g1 = lambda x : x[0] + x[1] - 2
g2 = lambda x : 6-x[0]-x[1]
g3 = lambda x : 2-x[1]+x[0]
g4 = lambda x : 2-x[0]+3*x[1]
g5 = lambda x : 4-(x[2]-3)**2-x[3]
g6 = lambda x : (x[4]-3)**2+x[5]-4

cons0 = [{"type" : "ineq","fun":g1},
        {"type" : "ineq","fun":g2},
        {"type" : "ineq","fun":g3},
        {"type" : "ineq","fun":g4},
        {"type" : "ineq","fun":g5},
        {"type" : "ineq","fun":g6}]

npop = 20*6
ngen = npop*3
front_size = 100
xmin,xmax = [0,0,1,0,1,0],[10,10,5,6,5,10]

minAg = optiMonoAG(f1,xmin,xmax,constraints=cons0)
minAg.setSelectionMethod("tournament")
x_f1max,f1_max= minAg.optimize(npop,ngen,verbose=False)

minAg = optiMonoAG(f2,xmin,xmax,constraints=cons0)
minAg.setSelectionMethod("tournament")
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
    cons = cons0 + [{"type" : "ineq","fun":fc}]
    minAg = optiMonoAG(f1,xmin,xmax,cons)
    minAg.setSelectionMethod("tournament")
    try :
        xk,f1k = minAg.optimize(npop,ngen,verbose=False)
        f2k = f2(xk)

        front_f1.append(f1k)
        front_f2.append(f2k)
        front_x.append(xk)
    except : pass

for k,ck in enumerate(f1_contrainte) :
    fc = lambda x : f1(x)-ck
    cons = cons0 + [{"type" : "ineq","fun":fc}]
    minAg = optiMonoAG(f2,xmin,xmax,cons)
    minAg.setSelectionMethod("tournament")
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

np.savetxt("opti5D_1.txt",np.array([front_f1,front_f2]).T)

fig1= plt.figure(1)

plt.plot(front_f1,front_f2,'d')
plt.grid(True)
plt.xlabel("f1")
plt.ylabel("f2")
plt.title('Optimisation bi-critères')
fig1.tight_layout()

plt.show()