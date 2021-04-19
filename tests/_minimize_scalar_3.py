import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from scipy.special import j1
plt.rc('font',family='Serif')

sys.path.append("..")
from _minimize_scalar import *

# FONCTION ET MINIMISATION
xmin,xmax = -3.5,2.0
xinit = (xmax+xmin)/2


f = lambda x : np.sin(1.2*x)/(np.sin(1.5*x)**2+1)
x = np.linspace(xmin,xmax,250)
y = f(x)

start = time.time()
dictgold = goldenSearch(f,xmin,xmax,returnDict=True)
inter = time.time()
dictgrad = scalarGradient(f,xinit,xmin,xmax,returnDict=True)
end = time.time()

xgold = dictgold["x"]
xgrad = dictgrad["x"]

fgold = dictgold['fmin']
fgrad = dictgrad['fmin']


print(f"Golden-Search {inter-start} s")
for s in dictgold :
    print(s," : ",dictgold[s])
print(f"\nGradient {end-inter} s")
for s in dictgrad :
    print(s," : ",dictgrad[s])

plt.figure(figsize=(9,3.5))

plt.plot(x,y,color='k',label="$f(x)$")
plt.plot(xgold,f(xgold),ls='',marker='D',markeredgecolor='k',label='Golden-Search')
plt.plot(xgrad,f(xgrad),ls='',marker='o',markeredgecolor='k',label='Gradient')
plt.title("Minimisation scalaire",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("f(x)",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("Minimisation1D.svg",dpi=150)
plt.show()