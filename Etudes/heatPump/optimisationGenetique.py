import numpy as np
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')

sys.path.append("../../Algorithmes_Genetiques")
from PyGen import optiMonoAG


from heatPumpSimulation import heatPump
## Optimisation
Taeo = 273.15 + 5.00
Te = 273.15 - 5.00
Tc = 273.15 + 40.00

xmin = np.array([0,-30,30])+273.15
xmax = np.array([20,-2.0,60.0])+273.15
xinit = [Taeo,Te,Tc]



sim = heatPump()
costFunction = lambda x : -sim.cost(x)

cons = [{'type': 'ineq', 'fun': sim.contrainte1},
        {'type': 'ineq', 'fun': sim.contrainte2},
        {'type': 'ineq', 'fun': sim.contrainte3},
        {'type': 'ineq', 'fun': sim.contrainte4},
        {'type': 'ineq', 'fun': sim.contrainte5},
        {'type': 'ineq', 'fun': sim.contrainte6}]


npop = 20*3
ngen = 3*npop
minAg = optiMonoAG(costFunction,xmin,xmax,constraints=cons,preprocess_function=sim.simulateHeatPump)
minAg.setConstraintMethod("penality")
minAg.setSelectionMethod("tournament")
minAg.setElitisme(True)

Xag,Yag = minAg.optimize(npop,ngen,verbose=True)
fitnessArray = minAg.getStatOptimisation()

sim.printDictSim(Xag)

print(Xag)
print(-Yag)

plt.figure(figsize=(8,4))
plt.plot(-fitnessArray,label='fmin',marker='o',ls='--',markeredgecolor='k',markerfacecolor="y",color='grey')
plt.grid(True)
plt.xlabel("Nombre de générations")
plt.ylabel("Fonction objectif")
plt.title("Convergence de la solution")
plt.legend(loc=0)
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)

plt.show()