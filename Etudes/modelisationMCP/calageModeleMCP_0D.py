import numpy as np
import matplotlib.pyplot as plt
import time
from modeleMCP import simulationMCP_0D as fsim

import sys
plt.rc('font',family='Serif')

sys.path.append("../../Algorithmes_Analytiques")
from _minimize_scalar import *


hvec = np.linspace(50,300,200)
err = np.zeros_like(hvec)
xmin,xmax = 50,300
xinit = 0.5*(xmin+xmax)

start = time.time()
dictgold = goldenSearch(fsim,xmin,xmax,returnDict=True,tol=1e-6)
inter = time.time()
dictgrad = scalarGradient(fsim,xinit,xmin,xmax,
                gtol= 1e-3,
                tol = 1e-6,
                returnDict=True,
                deriveMethod="finite-difference",
                dh=1e-3,
                maxIter=25)
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


for i,hi in enumerate(hvec):
    ri = fsim(hi)
    err[i] = ri


plt.plot(hvec,err)
plt.plot(xgold,fgold,ls='',marker='D',markeredgecolor='k',label='Golden-Search')
plt.plot(xgrad,fgrad,ls='',marker='o',markeredgecolor='k',label='Gradient')
plt.title("Minimisation scalaire",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("f(x)",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("Minimisation1D.svg",dpi=150)
plt.show()