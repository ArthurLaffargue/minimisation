
import numpy as np
from scipy.optimize import minimize
import sys
import time
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')


sys.path.append("..")
from _minimize_gradient import *
from _minimize_BFGS import *
from _minimize_Powell import *
from _minimize_NelderMead import *

x0 = np.array([-1,1])
xmin = np.array([-1.1,-0.25])
xmax = np.array([1.1,1.25])

p = 10
rosen = lambda x : (x[0]-1)**2 + p*(x[0]**2-x[1])**2
jac = lambda x : np.array([2.0*(x[0]-1)+4.0*p*x[0]*(x[0]**2-x[1]),
                           -2.0*p*(x[0]**2-x[1])],dtype=float)

x1,y1 = 0.25,0
x2,y2 = 0.5,0.3
a1 = (y1-y2)/(x1-x2)
b1 = -a1*x1 + y1
fc1 = lambda x : a1*x + b1
c1 = lambda x : -( fc1(x[0]) - x[1]  )
dc1 = lambda x : np.array([-a1,1.0])

fc2 = lambda x : 2-x
c2 = lambda x : ( fc2(x[0]) - x[1] )
dc2 = lambda x : np.array([-1.0,-1.0])

x1,y1 = -1.00,0.40
x2,y2 = 0.0,0.0
a3 = (y1-y2)/(x1-x2)
b3 = -a3*x1 + y1
fc3 = lambda x : a3*x + b3
c3 = lambda x : -( fc3(x[0])- x[1]  )
dc3 = lambda x : np.array([-a3,1.0])

cons = [{'type': 'ineq', 'fun': c1,"jac":dc1},
       {'type': 'ineq', 'fun': c2,"jac":dc2},
       {'type': 'ineq', 'fun': c3,"jac":dc3}]

#------- GRADIENT CONJUGUE -------#
start = time.time()
dictgrad = conjugateGradient(rosen,x0,xmin,xmax,
                            gf = jac,
                            returnDict=True,
                            storeIterValues=True,
                            methodConjugate="",
                            maxIter=2000,
                            constraints=cons)

dictconjgrad = conjugateGradient(rosen,x0,xmin,xmax,
                            gf = jac,
                            returnDict=True,
                            storeIterValues=True,
                            constraints=cons)

dictbfgs = BFGS(rosen,x0,xmin,xmax,
                            gf = jac,
                            returnDict=True,
                            storeIterValues=True,
                            constraints=cons)


dictPowell = Powell(rosen,x0,xmin,xmax,
                            returnDict=True,
                            storeIterValues=True,
                            constraints=cons)

dictNelderMead = NelderMead(rosen,x0,xmin,xmax,
                            returnDict=True,
                            storeIterValues=True,
                            constraints=cons)
end = time.time()

xgrad = dictgrad["xHistory"]
xconjgrad = dictconjgrad["xHistory"]
xbfgs = dictbfgs["xHistory"]
xPowell = dictPowell["xHistory"]
xNelderMead = dictNelderMead["xHistory"]

for s in dictgrad :
    if not( s.endswith("History") ):
        print(s," : ",dictgrad[s])

print("\n")
for s in dictconjgrad :
    if not( s.endswith("History") ):
        print(s," : ",dictconjgrad[s])

print("\n")
for s in dictbfgs :
    if not( s.endswith("History") ):
        print(s," : ",dictbfgs[s])

print("\n")
for s in dictPowell :
    if not( s.endswith("History") ):
        print(s," : ",dictPowell[s])

print("\n")
for s in dictNelderMead :
    if not( s.endswith("History") ):
        print(s," : ",dictNelderMead[s])

# Représentation fonction
n = 150
x = np.linspace(xmin[0],xmax[0],n)
y = np.linspace(xmin[1],xmax[1],n)
X,Y = np.meshgrid(x,y)
z = np.zeros((2,n**2))
z[0] = X.flatten()
z[1] = Y.flatten()

fz = rosen(z).reshape((n,n))
c1z = c1(z).reshape((n,n))
c2z = c2(z).reshape((n,n))
c3z = c3(z).reshape((n,n))
filtre =(c1z<0.0)|(c2z<0.0)|(c3z<0.0)
levels = np.linspace(fz.min(),fz.max(),50)
fz[filtre] = None


figContour = plt.figure("Rosen")
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=levels,cmap="Greys_r")
plt.plot(xgrad[0,0],xgrad[0,1],marker="s",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.plot(xgrad[-1,0],xgrad[-1,1],marker="o",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.text(-1,0.9,"Point de départ",fontsize=12)
plt.text(0.7,0.9,"Minimum",fontsize=12)
plt.grid(True)
plt.title("Fonction de Rosenbrock")
plt.tight_layout()
plt.savefig("rosenbrock.svg",dpi=150)

figContour = plt.figure("Gradient")
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=levels,cmap="Greys_r")

plt.plot(x,fc1(x),"k--",label="c1=0")
plt.plot(x,fc2(x),"k-.",label="c2=0")
plt.plot(x,fc3(x),"k-",label="c3=0")

plt.plot(xgrad[:,0],xgrad[:,1],lw=2,label='Gradient')
plt.plot(xconjgrad[:,0],xconjgrad[:,1],lw=2,label='Gradient conjugué')
plt.plot(xbfgs[:,0],xbfgs[:,1],lw=2,label='BFGS')
plt.plot(xPowell[:,0],xPowell[:,1],lw=2,label='Powell')
plt.plot(xNelderMead[:,0],xNelderMead[:,1],lw=2,label='NelderMead')

plt.plot([-1.0],[1.0],marker="s",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.plot([1.0],[1.0],marker="o",markeredgecolor="k",markerfacecolor="None",ms=10)

plt.grid(True)
plt.title("Rosenbrock avec contraintes")
plt.tight_layout()

plt.legend()
plt.xlim(xmin[0],xmax[0])
plt.ylim(xmin[1],xmax[1])

plt.savefig("optimisationContrainte.svg",dpi=150)
plt.show()






