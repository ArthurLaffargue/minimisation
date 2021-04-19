
import numpy as np
import numpy.random as rd
import time


class simulatedAnnealing :

    def __init__(self,f,xmin,xmax,maxIter=1000,constraints=[],preprocess_function=None):

        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)

        self.__ndof = len(self.__xmin)

        self.__preProcess = preprocess_function

        self.__constraints = constraints
        self.__function = f

        self.__maxIter = maxIter
        self.__penalityFactor = 1e6
        self.__initialTemp = 1e6
        self.__tempDecreaseRatio = 0.99
        self.__tempStepIter = 1
        self.__tempStepAcceptRate = 1
        self.__perturbationRation = 1/(self.__ndof+1)


        self.__constraintAbsTol = 1e-3
        self.statMinimize = []



    def __evaluateFitness(self,x):
        if self.__preProcess is not None :
            self.__preProcess(x)


        objective = self.__function(x)
        feasibility = True
        penality = 0.0
        for c in self.__constraints :
            type = c["type"]
            g = c['fun']
            gi = g(x)

            if type == 'strictIneq'  :
                feasibility &= gi>0
                penality += (gi*(gi<=0.0))**2

            if type == 'ineq' :
                feasibility &= gi>=0
                penality += np.minimum(gi,0.0)**2

            if type == 'eq'  :
                feasibility &= np.abs(gi)<=self.__constraintAbsTol
                penality += gi**2
        objective = objective + self.__penalityFactor*penality
        return objective,feasibility


    def __localPerturbation(self,x):
        r0 = 2*self.__perturbationRation*(rd.random(self.__ndof)-0.5)
        xnew = np.maximum(np.minimum(x + r0,1),0)
        return xnew

    def minimize(self):

        xmin = self.__xmin
        xmax = self.__xmax
        ndof = self.__ndof


        self.__optiObj = None
        self.__optiX = None
        self.statMinimize = []

        startTime = time.time()

        s0 = rd.sample(ndof)
        x0 = s0*(xmax-xmin) + xmin
        f0,feasibility = self.__evaluateFitness(x0)
        theta = self.__initialTemp
        acceptRate = 0.0
        stepIter = 0

        self.__optiObj = f0
        self.__optiX = x0
        self.__optiFeasible = feasibility

        for iter in range(self.__maxIter) :
            stepIter += 1

            s1 = self.__localPerturbation(s0)
            x1 = s1*(xmax-xmin) + xmin
            f1,feasibility = self.__evaluateFitness(x1)

            df = f1-f0
            if df <= 0 :
                s0 = s1
                x0 = x1
                f0 = f1
                acceptRate += 1/self.__tempStepIter

                if f0<self.__optiObj  :
                    self.__optiObj = f0
                    self.__optiX = x0
                    self.__optiFeasible = feasibility

            elif rd.random() < np.exp(-df/theta) :
                s0 = s1
                x0 = x1
                f0 = f1
                acceptRate += 1/self.__tempStepIter


            if (stepIter == self.__tempStepIter) or (acceptRate>= self.__tempStepAcceptRate) :
                acceptRate = 0.0
                stepIter = 0
                theta = theta*self.__tempDecreaseRatio


            self.statMinimize.append([self.__optiObj,theta,max(df,0)])

        self.statMinimize = np.array(self.statMinimize,dtype=float)
        self.statMinimize[:,2] = np.exp(-self.statMinimize[:,2]/self.statMinimize[:,1])

        endTime = time.time()
        duration = endTime-startTime
        print('\n'*2+'#'*60+'\n')
        print('Simulated annealing iterations completed')
        print('Sucess : ',self.__optiFeasible)
        print('Number of iteration : ',self.__maxIter)
        print('Elapsed time : %.3f s'%duration)

        print('#'*60+'\n')


        return self.__optiX


    def autoSetup(self,npermutations,config="highTemp"):
        deltaF = np.zeros(npermutations)
        xmin = self.__xmin
        xmax = self.__xmax
        ndof = self.__ndof

        s0 = rd.sample(ndof)
        x0 = s0*(xmax-xmin) + xmin
        f0,feasibility = self.__evaluateFitness(x0)
        for i in range(npermutations):
            s1 = self.__localPerturbation(s0)
            x1 = s1*(xmax-xmin) + xmin
            f1,feasible = self.__evaluateFitness(x1)
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

        print('\n'*2+'#'*60+'\n')
        print(f"# - initial temperature {config} {self.__initialTemp}")
        print(f"# - decrease ration {self.__tempDecreaseRatio} ")

