import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from networkSimulation import hydroNetwork

plt.rc('font',family='Serif')


hydroSim = hydroNetwork()
minEconomicFactor = hydroSim.minEconomicFactor
maxEconomicFactor = hydroSim.maxEconomicFactor
Dmin = hydroSim.Dmin
Dmax = hydroSim.Dmax

niteration = 150
economicConstraint = np.linspace(minEconomicFactor,
                                 maxEconomicFactor,
                                 niteration+2)[1:-1:]/maxEconomicFactor

DijVector = []
energyVector = []
economicVector = []
constraintViolation = []
iterationVector = []
funcCallsVector = []
Dk = Dmin
Dbounds = np.array([Dmin,Dmax]).T
options = {"maxiter":2000}
# figNetwork = plt.figure()
method = "SLSQP"

nfev = 0
def func(x):
    global nfev
    nfev += 1
    return hydroSim.energyCostFunc(x)

for k,ecoCons in enumerate(economicConstraint) :

    # figNetwork.clf()

    print("iteration ",k+1,"/",niteration)
    cons = [{"type":"ineq",
             "fun":lambda Dij : ecoCons - hydroSim.economicCostFunc(Dij),
             "jac":lambda Dij : -hydroSim.economicCostGrad(Dij) }]

    resMin = minimize(func,
                    Dk,
                    jac = hydroSim.energyCostGrad,
                    bounds = Dbounds,
                    constraints=cons,
                    options = options,
                    method=method)
    print(resMin)
    Dk = resMin.x

    phi_eco = hydroSim.economicCostFunc(Dk)
    phi_hydro = resMin.fun

    DijVector.append(Dk)
    energyVector.append(phi_hydro)
    economicVector.append(phi_eco)
    normViol = 0.0
    constraintViolation.append(normViol)
    iterationVector.append(resMin.nit)
    funcCallsVector.append(resMin.nfev)

    print(resMin)
    print("#"*50)
    print("nfev = ",nfev)
    print("#"*50)
    print("\n")



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
method = "scipy " + method
header = "energyFactor\t"
header += "econcomicFactor\t"
header += "constraintViolation\t"
header += "iteration\t"
header += "functionCalls"

np.savetxt("paretoFront/"+method+".txt",
            resultOptimization,
            header=header)


