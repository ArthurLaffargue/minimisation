import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
plt.rc('font',family='Serif')

sys.path.append("../../Algorithmes_Analytiques")
from _minimize_NelderMead import *


from heatPumpSimulation import heatPump
## Optimisation
Taeo = 273.15 + 0
Te = 273.15 - 30.00
Tc = 273.15 + 60.00

xmin = np.array([0,-30,30])+273.15
xmax = np.array([20,-2.0,60.0])+273.15
xinit = [Taeo,Te,Tc]
# xinit = 0.5*(xmin+xmax)


sim = heatPump()
cons = [{'type': 'ineq', 'fun': sim.contrainte1},
        {'type': 'ineq', 'fun': sim.contrainte2},
        {'type': 'ineq', 'fun': sim.contrainte3},
        {'type': 'ineq', 'fun': sim.contrainte4},
        {'type': 'ineq', 'fun': sim.contrainte5},
        {'type': 'ineq', 'fun': sim.contrainte6}]


dictmin = NelderMead(sim.cost,xinit,xmin,xmax,
                            returnDict=True,
                            constraints=cons,
                            tol=1e-9,
                            precallfunc=sim.simulateHeatPump,
                            maxIter=500,
                            penalityFactor=10,
                            storeIterValues=True)

sim.printDictSim(dictmin["x"])

print("\n")
for s in dictmin :
    if not( s.endswith("History") ):
        print(s," : ",dictmin[s])


##
plt.figure(figsize=(8,4))
plt.plot(dictmin["fHistory"][1:],
                label='fmin',
                marker='o',
                ls='--',
                markeredgecolor='k',
                markerfacecolor="b",
                color='grey')
plt.grid(True)
plt.xlabel("Nombre d'itérations")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)

plt.show()
