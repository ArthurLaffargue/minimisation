import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from plotterHeatNetwork import *



class hydroNetwork :

    def __init__(self)  :

        self.maxEnergyDrop = 1.0
        self.maxEconomicFactor = 1.0
        self.minEconomicFactor = 1.0

        self.maxVelocity = 2.5 #m/s
        self.minVelocity = 0.1 #m/s

        self.coefCoude = 1.2
        self.coefSortie = 1.5
        self.coefJonction = 3.5

        self.rho = 1000.00
        self.aeco = 1.2

        # -1 Ouverture du fichier

        self.__nodesTopology = pd.read_excel("networkTopology.xlsx",sheet_name="nodes")
        self.__branchTopology = pd.read_excel("networkTopology.xlsx",sheet_name="branch")
        self.__branchTopology.fillna(0,inplace=True)

        self.__nbranch = self.__branchTopology.shape[0]
        self.__nnodes = self.__nodesTopology.shape[0]

        self.__frictionCoef = self.__branchTopology["frictionFactor"].to_numpy(dtype=float)
        self.__betaLocal = self.coefCoude*self.__branchTopology["coude"].to_numpy(dtype=float) +\
                    self.coefSortie*self.__branchTopology["sortie"].to_numpy(dtype=float) +\
                    self.coefJonction*self.__branchTopology["jonction"].to_numpy(dtype=float)
        self.__pipeLength = self.__branchTopology["length"].to_numpy(dtype=float)
        self.__massFlowRate = self.__branchTopology["massFlowRateClosedLoops"].to_numpy(dtype=float)

        # -3 Diametre max et min
        self.Dmax = 2*np.sqrt(np.abs(self.__massFlowRate)/(self.rho*np.pi*self.minVelocity))
        self.Dmin = 2*np.sqrt(np.abs(self.__massFlowRate)/(self.rho*np.pi*self.maxVelocity))

        # -4 Initialisation
        self.maxEnergyDrop = self.energyDrop(self.Dmin)
        self.maxEconomicFactor = self.economicFactor(self.Dmax)
        self.minEconomicFactor = self.economicFactor(self.Dmin)


    def Rhydro(self,Dij):
        Sij = np.pi*Dij**2/4
        Rij = 1/2*(self.__frictionCoef*self.__pipeLength/Dij+self.__betaLocal)/(self.rho*Sij**2)
        return Rij


    def pressureDrop(self,Dij):
        deltaP = self.Rhydro(Dij)*self.__massFlowRate*np.abs(self.__massFlowRate)
        return deltaP

    def energyDrop(self,Dij):
        deltaE = np.sum(self.__massFlowRate*self.pressureDrop(Dij)/self.rho)
        return deltaE


    def economicFactor(self,Dij):
        return np.sum(Dij**self.aeco*self.__pipeLength)



    def energyCostFunc(self,Dij):
        return self.energyDrop(Dij)/self.maxEnergyDrop


    def economicCostFunc(self,Dij):
        return self.economicFactor(Dij)/self.maxEconomicFactor



    def energyCostGrad(self,Dij):
        dRhydro = -8/(np.pi**2*self.rho)*\
                  (5*self.__frictionCoef*self.__pipeLength/Dij**6+\
                   4*self.__betaLocal/Dij**5)
        grad = dRhydro*self.__massFlowRate**2*np.abs(self.__massFlowRate)/self.rho
        return grad/self.maxEnergyDrop


    def economicCostGrad(self,Dij):

        return self.aeco*Dij**(self.aeco-1)*self.__pipeLength/self.maxEconomicFactor



    def plotNetwork(self,**kwargs):
        # Affichage reseau
        plotNetworkTop(self.__branchTopology,self.__nodesTopology,**kwargs)

    def plotMassFlowRate(self,**kwargs):
        plotBranchField(self.__branchTopology,
                        self.__nodesTopology,
                        np.abs(self.__massFlowRate),
                        **kwargs)


    def plotUfieldBranch(self,U,**kwargs):
        plotBranchField(self.__branchTopology,
                        self.__nodesTopology,
                        U,
                        **kwargs)

if __name__ == "__main__" :
    import matplotlib.pyplot as plt
    figReseau = plt.figure(figsize=(7,4))
    hydroSim = hydroNetwork()
    hydroSim.plotNetwork(fig=figReseau,nodeLabel=False)
    figReseau.savefig("reseau.svg",dpi=150)


    hydroSim.plotMassFlowRate()
    plt.show()