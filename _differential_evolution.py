import numpy as np
import numpy.random as rd
from _utils import uniform_init_population_lhs,uniform_init_population_random

class differentialEvolutionSolver :

    __strategy = {'best1': '_best1'}

    __init_population = {'LHS' : '_latinhypercube_init',
                         'random' : "_random_init"}

    def __init__(self,func,xmin,xmax,constraints=[],preprocess_function=None,
                           strategy='best1',
                           maxiter=1000, popsize=15, tol=0.01, atol=0,
                           mutation=(0.5, 1), recombination=0.7,
                           returnDict=False, storIterValues=False, verbose=False,
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


        if strategy in self.__strategy:
            self.__mutation_operator = getattr(self, self.__strategy[strategy])
            self.__strategy = strategy
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


        self.__returnDict=returnDict
        self.__storIterValues=storIterValues
        self.__verbose=verbose




    def __reset_solver(self) :


        #Vecteur population
        self.__population = self.__init_population()
        self.__scaled_population = self.__scale(self.__population)
        self.__population_func = np.full(self.__popsize,np.inf)
        self.__population_pobj = np.full(self.__popsize,np.inf)
        self.__population_feasible = np.full(self.__popsize,False)


        #Solution
        self.__best_solution = None
        self.__best_func = None


        self.__success = False
        self.__lastPop = None
        self.__constrViolation = []
        self.__statOptimisation = None

        #Solution
        self.__best_solution = None
        self.__best_func = None


        self.__success = False
        self.__lastPop = None
        self.__constrViolation = []
        self.__statOptimisation = None



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
        penalObjective = objective - penality
        return objective,penalObjective,feasibility,penality,constrViolation




    def __evaluate_all_population(self) :
        """
        Evaluation de la fonction objectif et des contraintes
        """

        for i,xi, in enumerate(self.__scaled_population) :
            (obj_i,
            pobj_i,
            feas_i,
            _,_) = self.__evaluate_function_and_constraints(xi)

            self.__population_func[i] = obj_i
            self.__population_pobj[i] = pobj_i
            self.__population_feasible[i] = feas_i


    def __evaluate_candidate(self,index):

        xi = self.__scaled_population[index,:]

        (obj_i,
        pobj_i,
        feas_i,
        _,_) = self.__evaluate_function_and_constraints(xi)

        self.__population_func[index] = obj_i
        self.__population_pobj[index] = pobj_i
        self.__population_feasible[index] = feas_i


    def __sort_best_solution(self):
        best = np.argmin(self.__population_pobj)

        self.__population_func[[0, best]] = self.__population_func[[best, 0]]
        self.__population_pobj[[0, best]] = self.__population_pobj[[best, 0]]
        self.__population_feasible[[0, best]] = self.__population_feasible[[best, 0]]

        self.__population[[0, best], :] = self.__population[[best, 0], :]
        self.__scaled_population[[0, best], :] = self.__scaled_population[[best, 0], :]


        if (self.__best_func is None)  :
            self.__best_func = self.__population_pobj[0]
            self.__best_solution = self.__scaled_population[0,:]
        elif self.__best_func > self.__population_pobj[0] :
            self.__best_func = self.__population_pobj[0]
            self.__best_solution = self.__scaled_population[0,:]



    def solve(self,reset=False):

        if reset :
            self.__reset_solver()


        if np.isinf(self.__population_func).all() :
            self.__evaluate_all_population()
            self.__sort_best_solution()


        for iteration in range(1,self.__maxiter+1):
            self.__next_iteration()

            if self.__converged() :
                break

            if self.__verbose :
                message = "Differential_evolution iteration %d: penal objective = %.3e"%\
                              (iteration,
                               self.__best_func)

                print(message)






    def __next_iteration(self):


        scale_factor = rd.random()*(self.__mutation[1]-self.__mutation[0]) + \
                                    self.__mutation[0]


        for index in range(self.__popsize) :

            mutation_flag = rd.sample(size=self.__ndof) <= self.__mutation_rate

            x_prime = self.__mutation_operator(scale_factor,index)
            x_prime = np.maximum(0.,np.minimum(1.,x_prime))

            self.__population[index,mutation_flag] = x_prime[mutation_flag]
            x_prime = self.__population[index,:]
            self.__scaled_population[index,:] = self.__scale(x_prime)

            self.__evaluate_candidate(index)

            objective = self.__population_pobj[index]

            if objective < self.__best_func :
                self.__sort_best_solution()





    ## Mutations :

    def __select_samples(self, current_index,size):

        index_list = list(range(self.__popsize))
        index_list.remove(current_index)
        rd.shuffle(index_list)
        sample = index_list[:size]
        return sample

    def _best1(self,scale_factor,index):
        r1,r2 = self.__select_samples(index,2)
        y = self.__population[index,:] + \
            scale_factor*(self.__population[r1,:]-self.__population[r2,:])
        return y





















if __name__ == "__main__" :

    DE_instance = differentialEvolutionSolver(lambda x : x[0]**2 + 100.0*x[1]**2,
                                            [-1,-1],
                                            [1,1],verbose=True)


    DE_instance.solve()