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


fc = lambda x : x**2
c = lambda x : fc(x[0]) - x[1]
dc = lambda x : np.array([2*x[0],-1.0])
cons = [{'type': 'eq', 'fun': c,"jac":dc}]
#------- SCIPY -------#

xscipy = [x0]
res = minimize(rosen, x0,method="CG",callback=xscipy.append)
xscipy = np.array(xscipy)
# nfev = res.nfev + 3*res.njev


print("\nSCIPY : ")
print(res)
# print("nfev : ",nfev)

#------------------------------------------------------------------------------#
start = time.time()
dictgrad = conjugateGradient(rosen,x0,xmin,xmax,
                            returnDict=True,
                            storeIterValues=True,
                            methodConjugate="",
                            maxIter=2000,
                            constraints=cons)

dictconjgrad = conjugateGradient(rosen,x0,xmin,xmax,
                            returnDict=True,
                            storeIterValues=True,
                            methodConjugate="PR",
                            constraints=cons)

dictbfgs = BFGS(rosen,x0,xmin,xmax,
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

fz = rosen(z)
levels = np.linspace(fz.min(),fz.max(),50)


figContour = plt.figure("Rosen")
plt.plot(x,fc(x),"k",label='Contrainte')
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
plt.legend()

figContour = plt.figure("Gradient")
plt.plot(x,fc(x),"k",label='Contrainte')
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=levels,cmap="Greys_r")
plt.plot(xgrad[:,0],xgrad[:,1],'r',lw=2)
plt.plot(xgrad[0,0],xgrad[0,1],marker="s",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.plot(xgrad[-1,0],xgrad[-1,1],marker="o",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.grid(True)
plt.title("Méthode gradient")
plt.tight_layout()
plt.savefig("gradient.svg",dpi=150)
plt.legend()


figContour = plt.figure("Gradient conjugué")
plt.plot(x,fc(x),"k",label='Contrainte')
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=levels,cmap="Greys_r")
plt.plot(xconjgrad[:,0],xconjgrad[:,1],'r',lw=2)
plt.plot(xconjgrad[0,0],xconjgrad[0,1],marker="s",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.plot(xconjgrad[-1,0],xconjgrad[-1,1],marker="o",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.grid(True)
plt.title("Méthode gradient conjugué PR")
plt.tight_layout()
plt.savefig("conjGradient.svg",dpi=150)
plt.legend()

figContour = plt.figure("BFGS")
plt.plot(x,fc(x),"k",label='Contrainte')
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=levels,cmap="Greys_r")
plt.plot(xbfgs[:,0],xbfgs[:,1],'r',lw=2)
plt.plot(xbfgs[0,0],xbfgs[0,1],marker="s",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.plot(xbfgs[-1,0],xbfgs[-1,1],marker="o",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.grid(True)
plt.title("Méthode BFGS")
plt.tight_layout()
plt.savefig("BFGS.svg",dpi=150)
plt.legend()

figContour = plt.figure("Powell")
plt.plot(x,fc(x),"k",label='Contrainte')
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=levels,cmap="Greys_r")
plt.plot(xPowell[:,0],xPowell[:,1],'r',lw=2)
plt.plot(xPowell[0,0],xPowell[0,1],marker="s",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.plot(xPowell[-1,0],xPowell[-1,1],marker="o",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.grid(True)
plt.title("Méthode Powell")
plt.tight_layout()
plt.savefig("Powell.svg",dpi=150)
plt.legend()


figContour = plt.figure("NelderMead")
plt.plot(x,fc(x),"k",label='Contrainte')
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=levels,cmap="Greys_r")
plt.plot(xNelderMead[:,0],xNelderMead[:,1],'r',lw=2)
plt.plot(xNelderMead[0,0],xNelderMead[0,1],marker="s",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.plot(xNelderMead[-1,0],xNelderMead[-1,1],marker="o",markeredgecolor="k",markerfacecolor="None",ms=10)
plt.grid(True)
plt.title("Méthode NelderMead")
plt.tight_layout()
plt.savefig("NelderMead.svg",dpi=150)
plt.legend()
plt.show()






