import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from networkSimulation import hydroNetwork

plt.rc('font',family='Serif')


sys.path.append("../../Algorithmes_Analytiques")
from _minimize_gradient import *
from _minimize_BFGS import *
from _minimize_SQP_BFGS import *

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

# figNetwork = plt.figure()

penalityFactor = 10000
for k,ecoCons in enumerate(economicConstraint) :

    # figNetwork.clf()

    print("iteration ",k+1,"/",niteration)
    cons = [{"type":"ineq",
             "fun":lambda Dij : ecoCons - hydroSim.economicCostFunc(Dij),
             "jac":lambda Dij : -hydroSim.economicCostGrad(Dij) }]

    dictMin = conjugateGradient(hydroSim.energyCostFunc,
                        Dk,
                        Dmin,
                        Dmax,
                        gf = hydroSim.energyCostGrad,
                        returnDict=True,
                        constraints=cons,
                        penalityFactor=penalityFactor,
                        penalInnerIter = 1,
                        tol=1e-4,
                        gtol=1e-4,
                        maxIter=250)

    Dk = dictMin["x"]

    phi_eco = hydroSim.economicCostFunc(Dk)
    phi_hydro = dictMin["fmin"]

    DijVector.append(Dk)
    energyVector.append(phi_hydro)
    economicVector.append(phi_eco)
    normViol = np.linalg.norm(dictMin["constrViolation"])
    constraintViolation.append(normViol)
    iterationVector.append(dictMin["iterations"])
    funcCallsVector.append(dictMin["functionCalls"])

    for s in dictMin:
        if not( s.endswith("History") ):
            print(s," : ",dictMin[s])
    print("#"*50)
    print("\n")

    penalityFactor = max(100,0.9*penalityFactor)

    # title = r'Diamètres $\phi_{hydro} =$ %.3f and $\phi_{eco} = %.3f'%(phi_hydro,phi_eco)
    # hydroSim.plotUfieldBranch(Dk,
    #                         fig=figNetwork,
    #                         title=title,
    #                         nodeLabel=False,
    #                         cbarLabel='Diamètre [m]')
    # figNetwork.savefig("plotsnetwork/fig_no"+str(k)+".png")


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
method = dictMin["method"]
header = "energyFactor\t"
header += "econcomicFactor\t"
header += "constraintViolation\t"
header += "iteration\t"
header += "functionCalls"

np.savetxt("paretoFront/"+method+".txt",
            resultOptimization,
            header=header)


