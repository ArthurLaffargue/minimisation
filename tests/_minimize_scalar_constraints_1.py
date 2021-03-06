import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from scipy.optimize import minimize
plt.rc('font',family='Serif')

sys.path.append("..")
from minimization import minimize_scalar

# FONCTION ET MINIMISATION
xmin,xmax = 3,7
xinit = (xmin+xmax)/2

f = lambda x : -np.sin(0.25*x**1.2+x) + np.sin(10/3*x - 2*np.pi/3)
c = lambda x : (4.5-x)*(x-5.8)
r = 500
fpenal = lambda x : f(x) + r*np.minimum(c(x),0)**2
x = np.linspace(xmin,xmax,250)
y = f(x)

constraints = [{"type":"ineq",'fun':c}]

start = time.time()
dictgold = minimize_scalar(f,xmin,xmax,returnDict=True,constraints=constraints,method='goldenSearch')
inter = time.time()
dictgrad = minimize_scalar(f,xmin,xmax,returnDict=True,constraints=constraints,method='scalarGradient',xinit=xinit)
end = time.time()

xgold = dictgold["x"]
xgrad = dictgrad["x"]

print(f"Golden-Search {inter-start} s")
for s in dictgold :
    print(s," : ",dictgold[s])
print(f"\nGradient {end-inter} s")
for s in dictgrad :
    print(s," : ",dictgrad[s])


plt.figure(figsize=(9,3.5))

plt.plot(x,y,color='k',label="$f(x)$")
plt.plot(x,c(x),color="grey",ls='-.',label="$c>=0$")
plt.plot(x,fpenal(x),color="b",label="fonction pénalisée")
plt.plot(xgold,f(xgold),ls='',marker='D',markeredgecolor='k',label='Golden-Search')
plt.plot(xgrad,f(xgrad),ls='',marker='o',markeredgecolor='k',label='Gradient')
plt.title("Minimisation scalaire avec contraintes",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("f(x)",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.ylim(-7,3.6)
plt.tight_layout()
plt.savefig("Minimisation1D.svg",dpi=150)
plt.show()