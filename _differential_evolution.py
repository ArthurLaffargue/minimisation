import numpy as np
import numpy.random as rd
from _utils import uniform_init_population_lhs,uniform_init_population_random

__all__ = ["differential_evolution"]

def differential_evolution(func,xmin,xmax,constraints=[],preprocess_function=None,
                           strategy='best1bin',
                           maxIter=1000, popsize=15, tol=0.01, atol=0.,
                           mutation=(0.5, 1.), recombination=0.7,
                           returnDict=False, storeIterValues=False, verbose=False,
                           init='LHS',constraintAbsTol=1e-3,
                           penalityFactor = 1e3):
    
    DE_instance = differentialEvolutionSolver(func,
                            xmin,
                            xmax,
                            constraints=constraints,
                            preprocess_function=preprocess_function,
                            strategy=strategy,
                            maxiter=maxIter, 
                            popsize=popsize, 
                            tol=tol, atol=atol,
                            mutation=mutation, 
                            recombination=recombination,
                            returnDict=returnDict, 
                            storeIterValues=storeIterValues, 
                            verbose=verbose,
                            init=init,
                            constraintAbsTol=constraintAbsTol,
                            penalityFactor = penalityFactor)

    DE_instance.solve(reset=False)
    result = DE_instance.result
    return result


class differentialEvolutionSolver :

    __binomial = {'best1bin': '_best1',
                  'best2bin': '_best2',
                  'rand1bin': '_rand1',
                  'rand2bin': '_rand2',
                  'currenttobest1bin' : "_currenttobest1",
                  '_currenttorand1bin' : "_currenttorand1"}

    __exponential = {'best1exp': '_best1',
                     'best2exp': '_best2',
                     'rand1exp': '_rand1',
                     'rand2exp': '_rand2',
                     'currenttobest1exp' : "_currenttobest1",
                     '_currenttorand1exp' : "_currenttorand1"}

    __init_population = {'LHS' : '_latinhypercube_init',
                         'random' : "_random_init"}

    def __init__(self,func,xmin,xmax,constraints=[],preprocess_function=None,
                           strategy='best1bin',
                           maxiter=1000, popsize=15, tol=1e-3, atol=0.,
                           mutation=(0.5, 1.0), recombination=0.7,
                           returnDict=False, storeIterValues=False, verbose=False,
                           init='LHS',constraintAbsTol=1e-3,
                           penalityFactor = 1e3):


        # Bornes
        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)


        #Vecteur population
        self.__ndof = len(self.__xmin)
        self.__maxiter = maxiter
        self.__popsize = popsize
        self.__popshape = ((self.__popsize,self.__ndof))


        #Fonctions et contraintes
        self.__preProcess = preprocess_function
        self.__function = func
        self.__constraints = constraints


        #Contraintes et penalisation
        self.__constraintAbsTol = constraintAbsTol
        self.__penalityFactor = penalityFactor


        #Operateurs
        if len(mutation) == 2 :
            self.__mutation = np.sort(mutation)
        else :
            raise ValueError("Please select correct mutation option")
        self.__mutation_rate = recombination


        if strategy in self.__binomial :
            self.__strategy = strategy
            self.__mutation_operator = getattr(self, self.__binomial[self.__strategy])
        elif strategy in self.__exponential :
            self.__strategy = strategy
            self.__mutation_operator = getattr(self, self.__exponential[self.__strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")

        #Initialisation de population
        if init in self.__init_population :
            self.__init_population = getattr(self, self.__init_population[init])
        else:
            raise ValueError("Please select a valid initialization method")


        # relative and absolute tolerances for convergence
        self.__tol, self.__atol = tol, atol

        self.__reset_solver()


        self.__returnDict = returnDict
        self.__storeIterValues = storeIterValues
        self.__verbose = verbose




    def __reset_solver(self) :


        #Vecteur population
        self.__population = self.__init_population()
        self.__scaled_population = self.__scale(self.__population)
        self.__population_func = np.full(self.__popsize,np.inf,dtype=float)
        self.__population_pobj = np.full(self.__popsize,np.inf,dtype=float)
        self.__population_feasible = np.full(self.__popsize,False,dtype=bool)


        #Solution
        self.__best_solution = None
        self.__best_obj = None
        self.__best_penal_obj = None
        self.__best_feasible = False

        self.__constrViolation = []
        self.__trialVectors = []
        self.__trialObjectives = []

        self.__result = None




    def _latinhypercube_init(self) :
        return uniform_init_population_lhs(self.__popshape)


    def _uniform_init(self) :
        return uniform_init_population_random(self.__popshape)


    def __converged(self):
        converged = (np.std(self.__population_pobj) <=
                    self.__atol +
                    self.__tol * np.abs(np.mean(self.__population_pobj)))

        return converged


    def __scale(self,x):
        return x*(self.__xmax-self.__xmin) + self.__xmin

    def __evaluate_function_and_constraints(self,xi):
        """
        Evaluate the problem on xi point
        """
        if self.__preProcess is not None :
            self.__preProcess(xi)
        objective = self.__function(xi)
        constrViolation = []
        feasibility = True
        penality = 0.0
        for c in self.__constraints :
            type = c["type"]
            g = c['fun']
            gi = g(xi)

            if type == 'strictIneq'  :
                feasibility &= gi>0
                constrViol = np.minimum(gi,0.0)
                constrViolation.append(abs(constrViol))
                penality += constrViol**2

            if type == 'ineq' :
                feasibility &= gi>=0
                constrViol = np.minimum(gi,0.0)
                constrViolation.append(abs(constrViol))
                penality += constrViol**2

            if type == 'eq'  :
                constrViol = np.abs(gi)
                feasibility &= constrViol<=self.__constraintAbsTol
                constrViolation.append(abs(constrViol))
                penality += self.__penalityFactor*constrViol**2
        penalObjective = objective + penality
        return objective,penalObjective,feasibility,penality,constrViolation




    def __evaluate_all_population(self) :
        """
        Evaluation de la fonction objectif et des contraintes
        """
        best_obj = None 
        best_index = None
        for i,xi, in enumerate(self.__scaled_population) :
            (obj_i,
            pobj_i,
            feas_i,
            _,
            cviol_i) = self.__evaluate_function_and_constraints(xi)

            self.__population_func[i] = obj_i
            self.__population_pobj[i] = pobj_i
            self.__population_feasible[i] = feas_i

            if best_obj is None : 
                best_obj = pobj_i
                best_index = i 
                self.__constrViolation = cviol_i
            elif best_obj > pobj_i : 
                best_obj = pobj_i 
                best_index = i 
                self.__constrViolation = cviol_i


        self.__promote_best_solution(best_index)



    def __promote_best_solution(self,best):
        self.__population_func[[0, best]] = self.__population_func[[best, 0]]
        self.__population_pobj[[0, best]] = self.__population_pobj[[best, 0]]
        self.__population_feasible[[0, best]] = self.__population_feasible[[best, 0]]

        self.__population[[0, best], :] = self.__population[[best, 0], :]
        self.__scaled_population[[0, best], :] = self.__scaled_population[[best, 0], :]


        if (self.__best_penal_obj is None)  :
            self.__best_penal_obj = self.__population_pobj[0]
            self.__best_obj = self.__population_func[0]
            self.__best_solution = self.__scaled_population[0,:]
            self.__best_feasible = self.__population_feasible[0]
        elif self.__best_penal_obj > self.__population_pobj[0] :
            self.__best_penal_obj = self.__population_pobj[0]
            self.__best_obj = self.__population_func[0]
            self.__best_solution = self.__scaled_population[0,:]
            self.__best_feasible = self.__population_feasible[0]



    def __crossover(self,x_orig,x_prime):
        

        #Binomial
        if self.__strategy in self.__binomial : 
            mutation_flag = rd.sample(size=self.__ndof) <= self.__mutation_rate
            forced_point = rd.choice(self.__ndof)   
            mutation_flag[forced_point] = True
            
        
        #Exponential 
        if self.__strategy in self.__exponential : 
            crossovers = rd.sample(size=self.__ndof) <= self.__mutation_rate
            mutation_flag = np.zeros(self.__ndof,dtype=bool)
            cross_point = rd.choice(self.__ndof)
            count = 0
            while (count<self.__ndof) and (crossovers[count]) : 
                mutation_flag[cross_point] = True
                cross_point = (cross_point+1) % self.__ndof
                count += 1


        v_trial = np.where(mutation_flag,x_prime,x_orig)
        return v_trial

    def solve(self,reset=False):

        if reset :
            self.__reset_solver()


        if np.isinf(self.__population_func).all() :
            self.__evaluate_all_population()


        for iteration in range(1,self.__maxiter+1):
            self.__next_iteration()

            if self.__converged() :
                break

            if self.__verbose :
                message = "Differential_evolution iteration %d: penal objective = %.3e"%\
                              (iteration,
                               self.__best_penal_obj)

                print(message)


        self.__result = self.__complete_solver(iteration)



    def __next_iteration(self):


        scale_factor = rd.random()*(self.__mutation[1]-self.__mutation[0]) + \
                                    self.__mutation[0]


        for index in range(self.__popsize) :

            

            x_orig = self.__population[index,:]

            #Mutation
            x_prime = self.__mutation_operator(scale_factor,index)
            x_prime = np.maximum(0.,np.minimum(1.,x_prime))

            #Crossover
            v_trial = self.__crossover(x_orig,x_prime)
            u_trial = self.__scale(v_trial)

            #Evaluation
            (obj_trial,
            pobj_trial,
            feasible_trial,
            _,
            cviol_trial) = self.__evaluate_function_and_constraints(u_trial)

            

            #Selection 
            if pobj_trial < self.__population_pobj[index] : 

                self.__population[index,:] = v_trial
                self.__scaled_population[index,:] = u_trial

                self.__population_func[index] = obj_trial
                self.__population_pobj[index] = pobj_trial
                self.__population_feasible[index] = feasible_trial

                

                #Elitisme
                if pobj_trial < self.__best_penal_obj : 
                    self.__promote_best_solution(index)
                    self.__constrViolation = cviol_trial 

            if self.__storeIterValues : 
                #Archive 
                self.__archive(self.__best_solution,self.__best_penal_obj)


    def __archive(self,trial_vector,trial_objective):
        self.__trialVectors.append(trial_vector)
        self.__trialObjectives.append(trial_objective)

    def __complete_solver(self,last_iter):

        if self.__returnDict : 

            minDict = { "method":"Differential evolution",
                        "success":self.__best_feasible,
                        "x":self.__best_solution,
                        'f':self.__best_obj,
                        'penal_func':self.__best_penal_obj,
                        "constrViolation":self.__constrViolation,
                        "ierations":last_iter
                        }
            
            if self.__storeIterValues : 
                minDict["xHistory"] = np.array(self.__trialVectors)
                minDict["fHistory"] = np.array(self.__trialObjectives)

            return minDict
        else : 
            return self.__best_solution


    @property
    def result(self):
        return self.__result


    ## Mutations :

    def __select_samples(self, current_index,size):

        index_list = list(range(self.__popsize))
        index_list.remove(current_index)
        rd.shuffle(index_list)
        sample = index_list[:size]
        return sample

    def _best1(self,scale_factor,index):
        r1,r2 = self.__select_samples(index,2)
        y = self.__population[0,:] + \
            scale_factor*(self.__population[r1,:]-self.__population[r2,:])
        return y


    def _best2(self,scale_factor,index):
        r1,r2,r3,r4 = self.__select_samples(index,4)
        y = self.__population[0,:] + \
            scale_factor*(self.__population[r1,:]+self.__population[r2,:]-self.__population[r3,:]-self.__population[r4,:])
        return y
    

    def _rand1(self,scale_factor,index):
        r0,r1,r2 = self.__select_samples(index,3)
        y = self.__population[r0,:] + \
            scale_factor*(self.__population[r1,:]-self.__population[r2,:])
        return y
    
    def _rand2(self,scale_factor,index):
        r0,r1,r2,r3,r4 = self.__select_samples(index,5)
        y = self.__population[r0,:] + \
            scale_factor*(self.__population[r1,:]+self.__population[r2,:] -\
                        self.__population[r3,:]-self.__population[r4,:])
        return y
    

    def _currenttobest1(self,scale_factor,index):
        r1,r2 = self.__select_samples(index,2)
        y = self.__population[index,:] + \
            scale_factor*(self.__population[0,:]-self.__population[index,:]) +\
            scale_factor*(self.__population[r1,:]-self.__population[r2,:])
        return y 
    


    def _currenttorand1(self,scale_factor,index):
        r0,r1,r2 = self.__select_samples(index,3)
        y = self.__population[index,:] + \
            scale_factor*(self.__population[r0,:]-self.__population[index,:]) +\
            scale_factor*(self.__population[r1,:]-self.__population[r2,:])
        return y 



















if __name__ == "__main__" :

    resDE = differential_evolution(lambda x : x[0]**2 + 100.0*x[1]**2,
                                            [-1,-1],
                                            [1,1],verbose=False,returnDict=True)
    
    for ri in resDE : 
         print(ri," : ",resDE[ri])

