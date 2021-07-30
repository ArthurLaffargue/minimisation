import numpy as np
import numpy.random as rd
import time

# __all__ = ["minimize_simulatedAnnealing"]

class simulatedAnnealing :

    def __init__(self,f,xmin,xmax,maxIter=1000,constraints=[],preprocess_function=None,perturbationRatio=None):

        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)

        self.__ndof = len(self.__xmin)

        self.__preProcess = preprocess_function

        self.__constraints = constraints
        self.__function = f

        self.__maxIter = maxIter
        self.__penalityFactor = 1e3
        self.__initialTemp = 1e6
        self.__tempDecreaseRatio = 0.99
        if perturbationRatio is None :
            self.__perturbationRatio = 1/(self.__ndof+1)
        else :   
            self.__perturbationRatio = perturbationRatio

        self.__constraintAbsTol = 1e-3
        self.statMinimize = []
    
    def setTemperatureParams(self,initialTemp=None,decreaseRate=None):
        if decreaseRate is not None : 
            self.__tempDecreaseRatio = decreaseRate
        if initialTemp is not None :
            self.__initialTemp = initialTemp
    

    def setPenalityFactor(self,penalityFactor=1e3) : 
        self.__penalityFactor = penalityFactor

    def setConstraintTol(self,ctol=1e-3):
        self.__constraintAbsTol = ctol 

    def __evaluateFitness(self,x):
        if self.__preProcess is not None :
            self.__preProcess(x)


        objective = self.__function(x)
        feasibility = True
        penality = 0.0
        constrViol = []
        for c in self.__constraints :
            type = c["type"]
            g = c['fun']
            gi = g(x)

            if type == 'ineq' :
                feasibility &= gi>=0
                cviol = np.minimum(gi,0.0)
                penality += cviol**2
                constrViol.append(np.abs(cviol))

            if type == 'eq'  :
                feasibility &= np.abs(gi)<=self.__constraintAbsTol
                penality += gi**2
                constrViol.append(np.abs(gi))

        objective = objective + self.__penalityFactor*penality
        return objective,feasibility,constrViol


    def __localPerturbation(self,x):
        r0 = 2*self.__perturbationRatio*(rd.random(self.__ndof)-0.5)
        xnew = np.maximum(np.minimum(x + r0,1),0)
        return xnew

    def minimize(self,verbose=False,returnDict=False,storeIterValues=False):

        xmin = self.__xmin
        xmax = self.__xmax
        ndof = self.__ndof


        self.__optiObj = None
        self.__optiX = None
        self.statMinimize = []
        self.xHistory = []

        startTime = time.time()

        s0 = rd.sample(ndof)
        x0 = s0*(xmax-xmin) + xmin
        f0,feasibility,_ = self.__evaluateFitness(x0)
        theta = self.__initialTemp

        self.__optiObj = f0
        self.__optiX = x0
        self.__optiFeasible = feasibility

        for iter in range(self.__maxIter) :

            s1 = self.__localPerturbation(s0)
            x1 = s1*(xmax-xmin) + xmin
            f1,feasibility,_ = self.__evaluateFitness(x1)

            df = f1-f0
            if df <= 0 :
                s0 = s1
                x0 = x1
                f0 = f1

                if f0<self.__optiObj  :
                    self.__optiObj = f0
                    self.__optiX = x0
                    self.__optiFeasible = feasibility

            elif rd.random() < np.exp(-df/theta) :
                s0 = s1
                x0 = x1
                f0 = f1


            theta = theta*self.__tempDecreaseRatio

            if verbose : 
                message = "iteration %i - temperature %.3e - solution change %.3e"%(iter,theta,max(df,0.0))
                print(message)


            self.statMinimize.append([self.__optiObj,theta])
            self.xHistory.append(self.__optiX)

        self.statMinimize = np.array(self.statMinimize,dtype=float)
        self.xHistory = np.array(self.xHistory,dtype=float)

        endTime = time.time()
        duration = endTime-startTime
        if verbose : 
            print('\n'*2+'#'*60+'\n')
            print('Simulated annealing iterations completed')
            print('Sucess : ',self.__optiFeasible)
            print('Number of iteration : ',self.__maxIter)
            print('Elapsed time : %.3f s'%duration)

            print('#'*60+'\n')

        _,_,constrViolation = self.__evaluateFitness(self.__optiX)

        if returnDict : 

            minDict = { "method":"simulatedAnnealing",
                        "success":self.__optiFeasible,
                        "x":self.__optiX,
                        'f':self.__optiObj,
                        "constrViolation":constrViolation,
                        "temperatureDecreaseRate":self.__tempDecreaseRatio,
                        "initialTemperature":self.__initialTemp,
                        "finalTemperature":theta
                        }
            
            if storeIterValues : 
                minDict["xHistory"] = self.xHistory
                minDict["fHistory"] = self.statMinimize[:,0]
                minDict["tempHistory"] = self.statMinimize[:,1]
            
            return minDict

        return self.__optiX


    def autoSetup(self,npermutations,config="highTemp",verbose=False):
        deltaF = np.zeros(npermutations)
        xmin = self.__xmin
        xmax = self.__xmax
        ndof = self.__ndof

        s0 = rd.sample(ndof)
        x0 = s0*(xmax-xmin) + xmin
        f0,feasibility,_ = self.__evaluateFitness(x0)
        for i in range(npermutations):
            s1 = self.__localPerturbation(s0)
            x1 = s1*(xmax-xmin) + xmin
            f1,feasible,_= self.__evaluateFitness(x1)
            df = f1-f0
            f0 = f1
            x0 = x1
            s0 = s1
            deltaF[i] = df

        dfMean = deltaF[deltaF>0].mean()

        if config == "highTemp":
            self.__initialTemp = -dfMean/np.log(0.60)
        elif config == "lowTemp" :
            self.__initialTemp = -dfMean/np.log(0.20)
        else :
            self.__initialTemp = -dfMean/np.log(0.85)


        self.__finalTemp = -dfMean/np.log(0.001)
        self.__tempDecreaseRatio = (self.__finalTemp/self.__initialTemp)**(1/(0.5*self.__maxIter))

        if verbose : 
            print('\n'*2+'#'*60+'\n')
            print(f"# - initial temperature {config} {self.__initialTemp}")
            print(f"# - decrease ratio {self.__tempDecreaseRatio} ")




def minimize_simulatedAnnealing(func,
                xmin,
                xmax,
                maxIter=1000,
                constraints=[],
                preprocess_function=None,
                perturbationRatio=0.33,
                initialTemperature=None,
                temperatureDecreaseRate=None,
                autoSetUpIter=100,
                config='highTemp',
                penalityFactor = 1e3, 
                constraintAbsTol = 1e-3,
                verbose = False,
                returnDict=False,
                storeIterValues=False
                ) : 
    """
    
    """
    sa_instance = simulatedAnnealing(func,
                                     xmin,
                                     xmax,
                                     maxIter=maxIter,
                                     constraints=constraints,
                                     preprocess_function=preprocess_function,
                                     perturbationRatio=perturbationRatio)

    sa_instance.setTemperatureParams(initialTemp=initialTemperature,
                                     decreaseRate=temperatureDecreaseRate)

    sa_instance.setConstraintTol(ctol=constraintAbsTol)
    sa_instance.setPenalityFactor(penalityFactor=penalityFactor)
    
    if autoSetUpIter > 0 : 
        sa_instance.autoSetup(autoSetUpIter,config=config,verbose=verbose)
    
    res = sa_instance.minimize(verbose=verbose,
                            returnDict=returnDict,
                            storeIterValues=storeIterValues)
    
    return res



