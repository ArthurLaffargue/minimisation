import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from networkSimulation import hydroNetwork

plt.rc('font',family='Serif')


sys.path.append("../../Algorithmes_Analytiques")
sys.path.append("../../Algorithmes_Genetiques")
from PyGen import optiBiAG

hydroSim = hydroNetwork()
minEconomicFactor = hydroSim.minEconomicFactor
maxEconomicFactor = hydroSim.maxEconomicFactor
Dmin = hydroSim.Dmin
Dmax = hydroSim.Dmax

front_size = 150
npop = front_size
ngen = 100000//npop

print("Nombre de générations : ",ngen)
print("Appels de la fonction :",ngen*npop)
f1 = lambda x : -hydroSim.energyCostFunc(x)
f2 = lambda x : -hydroSim.economicCostFunc(x)

minAg = optiBiAG(f1,f2,Dmin,Dmax)
xpop,front_f1,front_f2 = minAg.optimize(npop,ngen,nfront=front_size,verbose=True)

Dpop = (Dmax-Dmin)*xpop + Dmin

energyVector = -front_f1
economicVector = -front_f2
constraintViolation = np.zeros_like(energyVector,dtype=float)
iterationVector = np.ones_like(energyVector,dtype=float)*npop/len(energyVector)
funcCallsVector = np.ones_like(energyVector,dtype=float)*ngen*npop/len(energyVector)

plt.figure("Pareto front")
plt.plot(economicVector,energyVector,'o')
plt.ylabel("econommicVector")
plt.ylabel("energyVector")
plt.grid(True)


plt.show()


resultOptimization = np.array([energyVector,
                               economicVector,
                               constraintViolation,
                               iterationVector,
                               funcCallsVector]).T
method = "NSGA-II"
header = "energyFactor\t"
header += "econcomicFactor\t"
header += "constraintViolation\t"
header += "iteration\t"
header += "functionCalls"

np.savetxt("paretoFront/"+method+".txt",
            resultOptimization,
            header=header)


np.savetxt("solutionNSGA/solution.txt",Dpop)