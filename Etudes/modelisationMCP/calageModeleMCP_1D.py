import numpy as np
import matplotlib.pyplot as plt
import time
from modeleMCP import simulationMCP_1D
import sys
plt.rc('font',family='Serif')

sys.path.append("../../Algorithmes_Analytiques")
from _minimize_scalar import *


hvec = np.linspace(50,300,50)
err = np.zeros_like(hvec)
xmin,xmax = 50,300
xinit = 0.5*(xmin+xmax)

simPCM1D = simulationMCP_1D(15)
fsim = simPCM1D.solveEquation
errfunc = simPCM1D.sqrtError

start = time.time()
dictgold = goldenSearch(errfunc,xmin,xmax,
                        returnDict=True,
                        precallfunc=fsim
                        ,tol=5e-3)
inter = time.time()
dictgrad = scalarGradient(errfunc,xinit,xmin,xmax,
                precallfunc=fsim,
                gtol= 1e-2,
                tol = 5e-3,
                returnDict=True,
                deriveMethod="finite-difference",
                dh = 5e-3,
                maxIter=50)

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
    fsim(hi)
    ri = errfunc(hi)
    err[i] = ri


plt.plot(hvec,err,'-o')
plt.plot(xgold,fgold,ls='',marker='D',markeredgecolor='k',label='Golden-Search')
plt.plot(xgrad,fgrad,ls='',marker='s',markeredgecolor='k',label='Gradient')
plt.title("Minimisation scalaire",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("f(x)",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("Minimisation1D.svg",dpi=150)
plt.show()