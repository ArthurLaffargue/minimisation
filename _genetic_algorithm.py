# coding: utf-8

import numpy as np
import numpy.random as rd
import time

from _utils import uniform_init_population_lhs,uniform_init_population_random


class genetic_operator : 
    _mutation = {"normal" : "_normalMutation",
                 "uniform" : "_uniformMutation"}

    _crossover = {"SBX" : "_SBXcrossover",
                  "uniform" : "_uniformCrossover"}

    def __init__(self,ndof,mutation_method,mutation_rate,mutation_step,
                    crossover_method,eta_cross,
                    sharing_distance) : 
        self.__ndof = ndof

        #Mutation
        self.__rateMut = mutation_rate
        if mutation_method in self._mutation : 
            self.__mutationMethod = mutation_method
            self.mutationFunction = getattr(self, self._mutation[self.__mutationMethod])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.__stepMut = mutation_step

        #Crossover
        if crossover_method in self._crossover : 
            self.__crossoverMethod = crossover_method
            self.crossoverFunction = getattr(self, self._crossover[self.__crossoverMethod])
        else:
            raise ValueError("Please select a valid crossover strategy")
        #croissement uniforme       
        self.__crossFactor = 1+1/(1+eta_cross)
        #croissement SBX
        self.__etac = eta_cross
        self.__alphac = 1/(self.__etac+1)

        #Sharing
        self.__sharingDist = sharing_distance


    def _uniformCrossover(self,selection,npop):
        """
        Operateur de croisement barycentrique
        """

        couples = np.zeros((npop//2,2,self.__ndof))
        children = np.zeros((npop,self.__ndof))
        alphaCross = rd.sample((npop//2,self.__ndof))*self.__crossFactor
        for i in range(npop//2):
            k = i
            while k == i :
                k = rd.randint(0,npop//2-1)
            couples[i] = [selection[i],selection[k]]
        children[:npop//2] = alphaCross*couples[:,0] + (1-alphaCross)*couples[:,1]
        children[npop//2:] = alphaCross*couples[:,1] + (1-alphaCross)*couples[:,0]

        return children
    
    def _SBXcrossover(self,selection,npop):
        couples = np.zeros((npop//2,2,self.__ndof))
        children = np.zeros((npop,self.__ndof))
        uCross = rd.sample((npop//2,self.__ndof))

        betaCross = np.zeros_like(uCross)
        uinf_filter = uCross<=0.5
        betaCross[uinf_filter] = (2*uCross[uinf_filter])**(self.__alphac)
        betaCross[~uinf_filter] = (2*(1-uCross[~uinf_filter]))**(-self.__alphac)

        for i in range(npop//2):
            k = i
            while k == i :
                k = rd.randint(0,npop//2-1)
            couples[i] = [selection[i],selection[k]]
        
        x1 = couples[:,0]
        x2 = couples[:,1]

        children[:npop//2] = 0.5*( (1-betaCross)*x1 + (1+betaCross)*x2 )
        children[npop//2:] = 0.5*( (1+betaCross)*x1 + (1-betaCross)*x2 )

        return children

    def sharingFunction(self,fitness,population):
        """
        Operateur de diversite de solution
        """
        dShare = self.__sharingDist

        if dShare is None : 
            return fitness
        elif dShare < 0 : 
            return fitness
        else : 
            distance = np.array([np.sqrt(np.sum((population - xj)**2,axis=1)) for xj in population])
            sharing = (1-distance/dShare)*(distance<dShare)
            sharingFactor = np.sum(sharing,axis=1)
            fitness = fitness/sharingFactor


            fmin = fitness.min()
            fmax = fitness.max()
            fitness = (fitness-fmin)/(fmax-fmin)
            return fitness



    def _uniformMutation(self,population,npop) :
        """
        Operateur de mutation
        """
        probaMutation = rd.sample((npop,self.__ndof))
        deltaX = self.__stepMut*(rd.sample((npop,self.__ndof))-0.5)
        population = population + deltaX*(probaMutation<=self.__rateMut)

        return population

    def _normalMutation(self,population,npop) :
        """
        Operateur de mutation
        """
        probaMutation = rd.sample((npop,self.__ndof))
        deltaX = self.__stepMut*( rd.normal(size=(npop,self.__ndof)) )
        population = population + deltaX*(probaMutation<=self.__rateMut)

        return population

class continousSingleObjectiveGA(genetic_operator) : 

    _init_population = {'LHS' : '_latinhypercube_init',
                         'random' : "_random_init"}

    _constraint_handling = ["penality","feasibility","mixed"]

    def __init__(self,func,xmin,xmax,constraints=[],preprocess_function=None,
                    mutation_method="normal",mutation_rate=0.10,mutation_step=0.10,
                    crossover_method="SBX",eta_cross=1.0,
                    sharing_distance=None,
                    atol = 0.,tol = 1e-3,stagThreshold = None,
                    eqcons_atol = 0.1,penalityFactor = 1e3,constraintMethod = "penality",
                    elitisme = True,elite_maxsize=None,
                    init="LHS") :
        """
        Instance de continousSingleObjectiveGA : 
        
        Algorithme genetique d'optimisation de fonction mono-objectif à 
        variables reelles. Recherche d'un optimum global de la fonction f sur
        les bornes xmin-xmax. 

        Parameters : 
        
            - func (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est 
                l'argument de la fonction de forme scalaire ou array et renvoie 
                un scalaire ou array.

            - xmin (array like) : 
                Borne inferieure des variables d'optimisation. 

            - xmax (array like) : 
                Borne supérieure des variables d'optimisation. Doit être de 
                même dimension que xmin. 

            - constraints (List of dict) option : 
                Definition des contraintes dans une liste de dictionnaires. 
                Chaque dictionnaire contient les champs suivants 
                    type : str
                        Contraintes de type egalite 'eq' ou inegalite 'ineq' ; 
                    fun : callable
                        La fonction contrainte de la meme forme que func ; 

            - preprocess_function (callable or None) option : 
                Definition d'une fonction sans renvoie de valeur à executer 
                avant chaque evaluation de la fonction objectif func ou des 
                contraintes.
        


        Example : 

            func = lambda x : x[0]**2 + x[1]**2
            xmin = [-1,-1]
            xmax = [1,1]
            ga_instance = continousSingleObjectiveGA(func,
                                                    xmin,
                                                    xmax)


            resGA = ga_instance.minimize(20,100,verbose=False,returnDict=True)
        
            Ouputs : 
                ############################################################

                AG iterations completed     
                Success :  True
                Number of generations :  100
                Population size :  20
                Elapsed time : 0.162 s
                ############################################################

                resGA = {
                    method  :  Continous Single Objective Genetic Algorithm
                    optimization  :  minimization
                    success  :  True
                    x  :  [-0.00222156  0.00380852] #may vary
                    f  :  1.9440156259914855e-05    #may vary
                    constrViolation  :  []
                    }

        """

        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)

        self.__ndof = len(self.__xmin)

        self.__preProcess = preprocess_function
        self.__function = func
        self.__constraints = constraints
        self.__num_cons = len(self.__constraints)

        #Operateurs genetiques
        super().__init__(self.__ndof,
                        mutation_method,mutation_rate,mutation_step,
                        crossover_method,eta_cross,
                        sharing_distance)
        self.selectionFunction = self.__selection_tournament

        #convergence 
        self.__atol = atol
        self.__tol = tol
        self.__stagThreshold = stagThreshold

        #Contraintes 
        if constraintMethod in self._constraint_handling : 
            self.__constraintMethod = constraintMethod
        else : 
            raise ValueError("Please select a valid constraint handlling strategy.")
        self.__constraintAbsTol = eqcons_atol
        self.__penalityFactor = penalityFactor

        self.__elitisme = elitisme
        if elite_maxsize is None :
            self.__elite_maxsize = np.inf
        else : 
            self.__elite_maxsize = elite_maxsize

        #Initialisation de population
        if init in self._init_population :
            self.__initial_population_selector = getattr(self, self._init_population[init])
            self.__initial_population = None 
        else:
            raise ValueError("Please select a valid initialization method")


        #init solver
        self.__sign_factor = 1.0
        self.__init_solver()



    def __init_solver(self):
        
        self.__best_xscale = None
        self.__best_x = None
        self.__best_obj = None
        self.__best_penality = None
        self.__best_feasibility = False
        self.__best_constrViolation = []

        self.__success = False
        self.__lastPop = None
        self.__statOptimisation = None

    def _latinhypercube_init(self,shape) :
        return uniform_init_population_lhs(shape)

    def _uniform_init(self) :
        return uniform_init_population_random(self.__popshape)

    def define_initial_population(self,xstart=None,selector='LHS'):
        """
        Definition d'une population initiale. 

        Parameter : 

            - xstart (array(npop,ndof)) or None option : 
                xstart est la solution initiale. Ses dimensions doivent etre de
                (npop,ndof). 
                npop la taille de la population et ndof le nombre de variable
                (degrees of freedom).
            
            - selector (string) option : 
                si pas de xstart definit, selection aléatoire : 
                    'LHS' : Latin Hypercube Selector, variante meilleure que uniforme ; 
                    'random' : loi uniforme ; 
                    sinon 'LHS' ; 
        """
        if xstart is not None : 
            x = np.array(xstart)
            npop,ndof = x.shape
            if ndof != self.__ndof : 
                raise ValueError("The size of initial population is not corresponding to bounds size")

            xpop = np.maximum(np.minimum((x-self.__xmin)/(self.__xmax-self.__xmin),1.0),0.0)
            if npop%2 != 0 : 
                xpop = np.insert(xpop,0,0.5,axis=0)
            
            self.__initial_population = xpop
        
        else : 
            if selector in self._init_population :
                self.__initial_population_selector = getattr(self, self._init_population[selector])
                self.__initial_population = None 
            else:
                raise ValueError("Please select a valid initialization method")

    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         OPERATEURS
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    def __evaluate_function_and_constraints(self,xi):
        """
        Evaluate the problem on xi point
        """
        if self.__preProcess is not None :
            self.__preProcess(xi)
        objective = self.__function(xi)*self.__sign_factor
        constrViolation = []
        feasibility = True
        penality = 0.0
        for c in self.__constraints : 
            type = c["type"]
            g = c['fun']
            gi = g(xi)

            if type == 'ineq' :
                feasibility &= gi>=0
                constrViol = np.minimum(gi,0.0)
                constrViolation.append(abs(constrViol))
                penality += constrViol**2

            if type == 'eq'  :
                constrViol = np.abs(gi)
                feasibility &= constrViol<=self.__constraintAbsTol
                constrViolation.append(abs(constrViol))
                penality += constrViol**2

        return objective,feasibility,penality,constrViolation

    def __fitness_and_feasibility(self,Xarray,npop) :
        """
        Evaluation de la fonction objectif et des contraintes
        """

        objective = np.ones(npop,dtype=float)
        feasibility = np.ones(npop,dtype=bool)
        penality = np.zeros(npop,dtype=float)
        constraintViol = []

        for i,xi, in enumerate(Xarray) :
            (obj_i,
            feas_i,
            penal_i,
            cviol_i) = self.__evaluate_function_and_constraints(xi)

            objective[i] = obj_i
            feasibility[i] = feas_i
            penality[i] = penal_i
            constraintViol.append(cviol_i)

        

        if self.__constraintMethod in ['penality', "mixed"] : 
            penalObjective = objective - self.__penalityFactor*penality
            omin = penalObjective.min()
            omax = penalObjective.max()
            fitness = (penalObjective-omin)/(omax-omin)

        if self.__constraintMethod == "feasibility" :
            omin = objective.min()
            omax = objective.max()
            fitness = (objective-omin)/(omax-omin)
        
        constraintViol = np.array(constraintViol)
        return objective,fitness,feasibility,penality,constraintViol


    def __selection_tournament(self,population,npop,fitness,feasibility,penality):
        """
        Operateur de selection par tournois
        """
        selection = np.zeros((npop//2,self.__ndof))
        for i in range(npop//2):
            indices = rd.choice(npop-1,2)

            if self.__constraintMethod == 'penality' or feasibility[indices].all() : 
                selection[i] = population[indices[np.argmax(fitness[indices])]]
            else : 
                r1,r2 = indices
                if penality[r1] < penality[r2] : 
                    selection[i] = population[r1]
                elif penality[r2] < penality[r1] : 
                    selection[i] = population[r2]
                else : 
                    selection[i] = population[indices[np.argmax(fitness[indices])]]
        return selection

    def __selection_elitisme(self,elite_pop,elite_obj,elite_penality,elite_feasibility,children_pop,children_obj,children_penal,children_feasibility) :
        """
        Operateur elitisme
        """
        npop = len(children_obj)
        nelite = len(elite_obj)

        #Assemblage 
        population = np.concatenate( (children_pop,elite_pop), axis=0)
        objective = np.concatenate( (children_obj,elite_obj) )
        penality = np.concatenate( (children_penal, elite_penality) )
        feasibility = np.concatenate( (children_feasibility, elite_feasibility) )
        penal_objective = objective-self.__penalityFactor*penality

        if (self.__constraintMethod == "penality") and (self.__num_cons > 0) : 
            sorted_index = np.argsort(-penal_objective)
        elif (self.__constraintMethod in ["feasibility","mixed"]) and (self.__num_cons > 0) : 
            rank = -1*feasibility
            dtype = [ ("rank", int) , ("penality", float), ("objective", float) ]
            values = np.array( list(zip(rank, penality,-objective)), dtype=dtype)
            sorted_index = np.argsort(values, order=["rank",'penality','objective'])
        else : 
            sorted_index = np.argsort(-objective)

        new_population = population[sorted_index][:npop]
        new_objective = objective[sorted_index][:npop]
        new_penal_objective = penal_objective[sorted_index][:npop]
        new_penality = penality[sorted_index][:npop]
        new_feasibility = feasibility[sorted_index][:npop]
        

        if self.__constraintMethod == "feasibility" : 
            
            elite_feasibility = feasibility[sorted_index]
            elite_pop = population[sorted_index][elite_feasibility]
            elite_obj = objective[sorted_index][elite_feasibility]
            nelite = len(elite_obj)
            elite_feasibility = np.full(nelite,True,dtype=bool)
            elite_penality = np.full(nelite,0.0,dtype=float)
        else : 
            elite_pop = population[sorted_index]
            elite_obj = objective[sorted_index]
            elite_feasibility = feasibility[sorted_index]
            elite_penality = penality[sorted_index]


        if nelite > self.__elite_maxsize : 
            elite_pop = elite_pop[:self.__elite_maxsize]
            elite_obj = elite_obj[:self.__elite_maxsize]


        if (self.__constraintMethod in ["penality","mixed"]) and (self.__num_cons>0) : 
            omin,omax = new_penal_objective.min(),new_penal_objective.max()
            fitness = (new_penal_objective-omin)/(omax-omin)
        if (self.__constraintMethod == "feasibility") or (self.__num_cons==0) : 
            omin,omax = new_objective.min(),new_objective.max()
            fitness = (new_objective-omin)/(omax-omin)

        return (new_population,new_objective,new_penality,new_feasibility,
                elite_pop,elite_obj,elite_penality,elite_feasibility,fitness)

        

    def __archive_solution(self,population,objective,penality,constrViol,feasibility):
        """
        Archive la solition de meilleur objectif
        """
        penal_objective = (objective-self.__penalityFactor*penality)
        index = np.argmax(penal_objective)
        better_solution = False
        xpenal_obj = penal_objective[index]
        xfeasible = feasibility[index]
        if (self.__best_obj is None)  : 
            better_solution = True
            self.__best_xscale = population[index]
            self.__best_x = self.__best_xscale*(self.__xmax-self.__xmin) + self.__xmin
            self.__best_obj = objective[index]
            self.__best_penality = penality[index]
            self.__best_feasibility = feasibility[index]
            self.__best_constrViolation = constrViol[index]
            return better_solution

        best_penal_obj = self.__best_obj - self.__penalityFactor*self.__best_penality
        if (xpenal_obj > best_penal_obj ) : 
            better_solution = True
            self.__best_xscale = population[index]
            self.__best_x = self.__best_xscale*(self.__xmax-self.__xmin) + self.__xmin
            self.__best_obj = objective[index]
            self.__best_penality = penality[index]
            self.__best_feasibility = feasibility[index]
            self.__best_constrViolation = constrViol[index]

        return better_solution

    def __archive_details(self,generation):
        """
        Archive une liste de la meilleur solution par iteration
        """
        if self.__best_obj is not None : 
            self.__statOptimisation[generation] = self.__best_obj*self.__sign_factor
        else : 
            self.__statOptimisation[generation] = None



    def __checkConvergenceState(self,generation,last_improvement,objective) : 
        
        c1 = False
        if self.__stagThreshold is not None : 
            c1 = (generation - last_improvement) > self.__stagThreshold
        
        c2 = ( np.std(objective) )<= ( self.__atol + self.__tol*np.abs(np.mean(objective)) )
            
        converged = (c2 or c1) and self.__best_feasibility

        return converged
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         ALGORITHME
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###

    def __runOptimization(self,npop,ngen,verbose=True):
        """
        Optimisation
        """

        ## Initialisation

        if npop%2 != 0 :
            npop += 1
        ndof = self.__ndof

        if self.__initial_population is None : 
            population = self.__initial_population_selector((npop,ndof)) #population initiale
        else : 
            population = self.__initial_population
            npop = len(population)

        xmin = self.__xmin
        xmax = self.__xmax
        
        self.__init_solver()
        last_improvement = 0

        self.__statOptimisation  = np.full(ngen,None,dtype=float)
        objective = np.zeros(npop)


        startTime = time.time()

        Xarray = population*(xmax-xmin) + xmin
        (objective,
        fitness,
        feasibility,
        penality,
        constrViol) = self.__fitness_and_feasibility(Xarray,npop)
        elite_pop = population[feasibility]
        elite_obj = objective[feasibility]
        elite_feasibility = feasibility[feasibility]
        elite_penality = penality[feasibility]
        for generation in range(ngen):

            #Algorithme genetique
            fitness = self.sharingFunction(fitness,population)
                        
            selection = self.selectionFunction(population,npop,fitness,feasibility,penality)

            children_pop = self.crossoverFunction(selection,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            children_pop = self.mutationFunction(children_pop,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            Xarray = children_pop*(xmax-xmin) + xmin
            (children_obj,
            fitness,
            children_feasibility,
            children_penality,
            children_constrViol) = self.__fitness_and_feasibility(Xarray,npop)

            #Elistisme
            if self.__elitisme :
                (population,
                objective,
                penality,
                feasibility,
                elite_pop,
                elite_obj,
                elite_penality,
                elite_feasibility,
                fitness) = self.__selection_elitisme(elite_pop,
                                                    elite_obj,
                                                    elite_penality,
                                                    elite_feasibility,
                                                    children_pop,
                                                    children_obj,
                                                    children_penality,
                                                    children_feasibility)
            else :
                population = children_pop[:]
                objective = children_obj[:]
                penality = children_penality[:]
                feasibility = children_feasibility[:]



            better_sol = self.__archive_solution(children_pop,
                                                 children_obj,
                                                 children_penality,
                                                 children_constrViol,
                                                 children_feasibility)
            self.__archive_details(generation)

            if better_sol  : 
                last_improvement = generation
            
            converged = self.__checkConvergenceState(generation,
                                                     last_improvement,
                                                     objective)

            if verbose :
                print('Iteration ',generation+1)


            if converged : 
                print("SOLUTION CONVERGED")
                break 

    
        Xarray = population*(xmax-xmin) + xmin
        endTime = time.time()
        duration = endTime-startTime

        ##MESSAGES
        if self.__best_x is not None : 
            self.__success = self.__best_feasibility

        print('\n'*2+'#'*60+'\n')
        print('AG iterations completed')
        print("Success : ", self.__success)
        print('Number of generations : ',generation+1)
        print('Population size : ',npop)
        print('Elapsed time : %.3f s'%duration)

        print('#'*60+'\n')

        self.__lastPop = Xarray
    

    def minimize(self,npop,ngen,verbose=True,returnDict=False):
        """
        Algorithme de minimisation de la fonction objectif sous contrainte.

        Parameters : 

            - npop (int) : 
                Taille de la population. Si npop est impair, l'algorithm 
                l'augmente de 1. Usuellement pour un probleme sans contrainte 
                une population efficace est situee entre 5 et 20 fois le nombre 
                de variable. Si les contraintes sont fortes, il sera utile 
                d'augmenter la population. Ce parametre n'est pas pris en compte
                si une population initiale a ete definie.

            - ngen (int) : 
                Nombre de generation. Usuellement une bonne pratique est de 
                prendre 2 à 10 fois la taille de la population. 

            - verbose (bool) option : 
                Affiche l'etat de la recherche pour chaque iteration. Peut 
                ralentir l'execution.

            - returnDict (bool) option : 
                Si l'option est True alors l'algorithme retourne un 
                dictionnaire. 
            
        Returns : 

            Si (returnDict = False) : 
                tuple : xsolution, objective_solution (array(ndof), array(1)) 
                            ou (None, None)
                    - xsolution est la meilleure solution x historisee. 
                      Sa dimension correspond a ndof, la taille du probleme 
                      initial.
                    - objective_solution est la fonction objectif evaluee à 
                      xsolution. 
                    Si la solution n'a pas convergee et les contraintes jamais 
                    validee, l'algorithme retourne (None, None)
            
            Si (returnDict = False) : 
                dict :
                    "method" (str) : algorithm utilise.
                    "optimization" (str) : minimisation ou maximisation.
                    "success" (bool) : True si l'algorithm a converge.
                    "x" (array or None) : Solution ou None si success = False.
                    "f" (array or None) : Minimum ou None si success = False. 
        """
        self.__sign_factor = -1.0
        self.__runOptimization(npop, ngen, verbose=verbose)

        if returnDict : 
            result = {"method":"Continous Single Objective Genetic Algorithm",
                      "optimization" : "minimization",
                      "success":self.__success,
                      "x" : self.__best_x,
                      "f" : self.__best_obj*self.__sign_factor,
                      "constrViolation" : self.__best_constrViolation
                      }
            return result
        else : 
            return self.__best_x,self.__best_obj*self.__sign_factor
    
    def maximize(self,npop,ngen,verbose=True,returnDict=False):
        """
        Algorithme de maximisation de la fonction objectif sous contrainte.

        Parameters : 

            - npop (int) : 
                Taille de la population. Si npop est impair, l'algorithm 
                l'augmente de 1. Usuellement pour un probleme sans contrainte 
                une population efficace est situee entre 5 et 20 fois le nombre 
                de variable. Si les contraintes sont fortes, il sera utile 
                d'augmenter la population. Ce parametre n'est pas pris en compte
                si une population initiale a ete definie.

            - ngen (int) : 
                Nombre de generation. Usuellement une bonne pratique est de 
                prendre 2 à 10 fois la taille de la population. 

            - verbose (bool) option : 
                Affiche l'etat de la recherche pour chaque iteration. Peut 
                ralentir l'execution.

            - returnDict (bool) option : 
                Si l'option est True alors l'algorithme retourne un 
                dictionnaire. 
            
        Returns : 

            Si (returnDict = False) : 
                tuple : xsolution, objective_solution (array(ndof), array(1)) 
                            ou (None, None)
                    - xsolution est la meilleure solution x historisee. 
                      Sa dimension correspond a ndof, la taille du probleme 
                      initial.
                    - objective_solution est la fonction objectif evaluee à 
                      xsolution. 
                    Si la solution n'a pas convergee et les contraintes jamais 
                    validee, l'algorithme retourne (None, None)
            
            Si (returnDict = False) : 
                dict :
                    "method" (str) : algorithm utilise.
                    "optimization" (str) : minimisation ou maximisation.
                    "success" (bool) : True si l'algorithm a converge.
                    "x" (array or None) : Solution ou None si success = False.
                    "f" (array or None) : Maximum ou None si success = False. 
        """
        self.__sign_factor = 1.0
        self.__runOptimization(npop, ngen, verbose=verbose)

        if returnDict : 
            result = {"method":"Continous Single Objective Genetic Algorithm",
                      "optimization" : "maximization",
                      "success":self.__success,
                      "x" : self.__best_x,
                      "f" : self.__best_obj*self.__sign_factor,
                      "constrViolation" : self.__best_constrViolation
                      }
            return result
        else : 
            return self.__best_x,self.__best_obj*self.__sign_factor
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         RENVOIS
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###

    def getLastPopulation(self) : return self.__lastPop

    def getStatOptimisation(self): return self.__statOptimisation











class continousBiObjective_NSGA(genetic_operator):
    _init_population = {'LHS' : '_latinhypercube_init',
                         'random' : "_random_init"}

    _constraint_handling = ["penality","feasibility","mixed"]

    def __init__(self,func1,func2,xmin,xmax,constraints=[],preprocess_function=None,
                    func1_criterion="min",func2_criterion="min",
                    mutation_method="normal",mutation_rate=0.10,mutation_step=0.10,
                    crossover_method="SBX",eta_cross=1.0,
                    sharing_distance=None,
                    eqcons_atol = 0.1,penalityFactor = 1e3,constraintMethod = "feasibility",
                    init="LHS") :
        """
        continousBiObjective_NSGA : 
        
        Algorithme genetique d'optimisation bi-objectif à variables reelles. 
        Recherche du front de Pareto de fonctions scalaires à variables 
        continues sur l'intervalle [xmin;xmax].
        ---------------------------------------------------------------------------------------------------------
        Source : 
        A Fast and Elitist Multiobjective Genetic Algorithm : NSGA-II
        Kalyanmoy Deb, Associate Member, IEEE, Amrit Pratap, Sameer Agarwal, 
        and T. Meyarivan
        Parameters : 
            - func1 (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est 
                l'argument de la fonction de forme scalaire ou array et 
                renvoie un scalaire ou array.
            - func2 (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est 
                l'argument de la fonction de forme scalaire ou array et 
                renvoie un scalaire ou array.
            - xmin (array like) : 
                Borne inferieure des variables d'optimisation. 
            - xmax (array like) : 
                Borne supérieure des variables d'optimisation. Doit être de 
                même dimension que xmin. 
            - constraints (List of dict) option : 
                Definition des contraintes dans une liste de dictionnaires. 
                Chaque dictionnaire contient les champs suivants 
                    type : str
                        Contraintes de type egalite 'eq' ou inegalite 'ineq' ; 
                    fun : callable
                        La fonction contrainte de la meme forme que func ; 
            - preprocess_function (callable or None) option : 
                Definition d'une fonction sans renvoie de valeur à executer 
                avant chaque evaluation de la fonction objectif func ou 
                des contraintes.
            - func1_criterion (string) option : 
                Definition du critere du premier objectif. 
                  >>'min' = minimisation
                  >>'max' = maximisation
                  >> else : maximisation
            - func2_criterion (string) option : 
                Definition du critere du deuxieme objectif. 
                  >>'min' = minimisation
                  >>'max' = maximisation
                  >> else : maximisation
        Example : 
            import numpy as np 
            xmin,xmax = [-2],[2]
            x = np.linspace(xmin[0],xmax[0],150)
            f1 = lambda x : (0.5*x**2+x)/4
            f2 = lambda x : (0.5*x**2-x)/4
            c = lambda x : np.sin(x)
            cons = [{"type":'ineq','fun':c}]
            nsga_instance = continousBiObjective_NSGA(f1,
                                                    f2,
                                                    xmin,
                                                    xmax,
                                                    constraints=cons,
                                                    func1_criterion='min',
                                                    func2_criterion='min')
                                                    
            xfront,f1front,f2front = nsga_instance.optimize(20,100)
        """
        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)

        self.__ndof = len(self.__xmin)

        self.__preProcess = preprocess_function
        self.__function1 = func1
        self.__function2 = func2
        self.__constraints = constraints
        self.__num_cons = len(self.__constraints)

        #Operateurs genetiques
        super().__init__(self.__ndof,
                        mutation_method,mutation_rate,mutation_step,
                        crossover_method,eta_cross,
                        sharing_distance)
        self.selectionFunction = self.__selection_tournament

        #Contraintes 
        if constraintMethod in self._constraint_handling : 
            self.__constraintMethod = constraintMethod
        else : 
            raise ValueError("Please select a valid constraint handlling strategy.")
        self.__constraintAbsTol = eqcons_atol
        self.__penalityFactor = penalityFactor

        #Initialisation de population
        if init in self._init_population :
            self.__initial_population_selector = getattr(self, self._init_population[init])
            self.__initial_population = None 
        else:
            raise ValueError("Please select a valid initialization method")


        #init solver
        self.__sign_factor = [1.0,1.0]
        if func1_criterion == 'min' : 
            self.__sign_factor[0] = -1.0
        if func2_criterion == 'min' : 
            self.__sign_factor[1] = -1.0
        self.__init_solver()

        
    def __init_solver(self):

        self.__success = False
        self.__xfront = None
        self.__f1front = None
        self.__f2front = None
        self.__feasibility_front = None
        self.__cviol_front = None
        self.__lastPop = None

    def _latinhypercube_init(self,shape) :
        return uniform_init_population_lhs(shape)

    def _uniform_init(self) :
        return uniform_init_population_random(self.__popshape)

    def define_initial_population(self,xstart=None,selector='LHS'):
        """
        Definition d'une population initiale. 

        Parameter : 

            - xstart (array(npop,ndof)) or None option : 
                xstart est la solution initiale. Ses dimensions doivent etre de
                (npop,ndof). 
                npop la taille de la population et ndof le nombre de variable
                (degrees of freedom).
            
            - selector (string) option : 
                si pas de xstart definit, selection aléatoire : 
                    'LHS' : Latin Hypercube Selector, variante meilleure que uniforme ; 
                    'random' : loi uniforme ; 
                    sinon 'LHS' ; 
        """
        if xstart is not None : 
            x = np.array(xstart)
            npop,ndof = x.shape
            if ndof != self.__ndof : 
                raise ValueError("The size of initial population is not corresponding to bounds size")

            xpop = np.maximum(np.minimum((x-self.__xmin)/(self.__xmax-self.__xmin),1.0),0.0)
            if npop%2 != 0 : 
                xpop = np.insert(xpop,0,0.5,axis=0)
            
            self.__initial_population = xpop
        
        else : 
            if selector in self._init_population :
                self.__initial_population_selector = getattr(self, self._init_population[selector])
                self.__initial_population = None 
            else:
                raise ValueError("Please select a valid initialization method")


    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         OPERATEURS
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    
    def __fast_non_dominated_sort(self,x1,x2,selection_size=None):
        size_pop = len(x1)
        if selection_size is None :
            selection_size = size_pop
        index_R =  np.array(range(size_pop),dtype=int)
        Fi = []
        list_Sr = []
        list_nr = []
        rank_R = np.ones_like(index_R,dtype=int)*-1

        for r,(x1r,x2r) in enumerate(zip(x1,x2)) :

            dominated_test = (x1r<x1)&(x2r<x2)
            dominance_test = (x1r>x1)&(x2r>x2)


            nr = np.count_nonzero(dominated_test) #number of dominated solutions
            Sr = [q for q in range(size_pop) if dominance_test[q]]

            list_Sr.append(Sr)
            list_nr.append(nr)

            if nr == 0 :
                rank_R[r] = 1
                Fi.append(r)


        i = 1
        size = len(Fi)
        while size > 0 and size<selection_size:
            Q = []
            for r in Fi :
                Sr = list_Sr[r]
                for q in Sr :
                    nq = list_nr[q]
                    nq = nq - 1
                    list_nr[q] = nq
                    if nq == 0 :
                        rank_R[q] = i+1
                        Q.append(q)
            i += 1
            Fi = Q
            size += len(Fi)

        non_ranked_sol = rank_R < 1
        rank_R[non_ranked_sol] = max(rank_R)+1

        return rank_R

    def __crowning_distance(self,x1,x2,rank_R):
        size = len(x1)
        max_rank = max(rank_R)
        crowning_dist = np.zeros_like(x1,dtype=float)
        x1min,x1max = x1.min(),x1.max()
        x2min,x2max = x2.min(),x2.max()
        index_array = np.array(list(range(size)))

        for rank_k in range(1,max_rank+1):
            index_sol = index_array[rank_R == rank_k]
            x1_k = x1[index_sol]
            x2_k = x2[index_sol]
            distance_k = np.zeros_like(x1_k,dtype=float)

            sortindex = np.argsort(x1_k)
            x1_k = x1_k[sortindex]
            x2_k = x2_k[sortindex]
            index_sol = index_sol[sortindex]
            distance_k = distance_k[sortindex]
            distance_k[1:-1:] = (x1_k[2:] - x1_k[:-2])/(x1max-x1min)
            distance_k[0] = 1.0
            distance_k[-1] = 1.0

            sortindex = np.argsort(x2_k)
            x2_k = x2_k[sortindex]
            index_sol = index_sol[sortindex]
            distance_k = distance_k[sortindex]
            distance_k[1:-1:] += (x2_k[2:] - x2_k[:-2])/(x2max-x2min)
            distance_k[0] += 1.0
            distance_k[-1] += 1.0

            crowning_dist[index_sol] = distance_k

        dist_min,dist_max = crowning_dist.min(),crowning_dist.max()
        return (crowning_dist-dist_min)/(dist_max-dist_min)

    def __evaluate_function_and_constraints(self,xi):
        """
        Evaluate the problem on xi point
        """
        if self.__preProcess is not None :
            self.__preProcess(xi)
        objective1 = self.__function1(xi)*self.__sign_factor[0]
        objective2 = self.__function2(xi)*self.__sign_factor[1]
        constrViolation = []
        feasibility = True
        penality = 0.0
        for c in self.__constraints : 
            type = c["type"]
            g = c['fun']
            gi = g(xi)

            if type == 'ineq' :
                feasibility &= gi>=0
                constrViol = np.minimum(gi,0.0)
                constrViolation.append(abs(constrViol))
                penality += constrViol**2

            if type == 'eq'  :
                constrViol = np.abs(gi)
                feasibility &= constrViol<=self.__constraintAbsTol
                constrViolation.append(abs(constrViol))
                penality += constrViol**2

        return objective1,objective2,feasibility,penality,constrViolation

    def __fitness_and_feasibility(self,Xarray,npop) :
        """
        Evaluation de la fonction objectif et des contraintes
        """
        objective1 = np.zeros(npop,dtype=float)
        objective2 = np.zeros(npop,dtype=float)
        feasibility = np.ones(npop,dtype=bool)
        penality = np.zeros(npop,dtype=float)
        constrViolation = np.zeros((npop,self.__num_cons))
        for i,xi, in enumerate(Xarray) :
            (obj1_i,
            obj2_i,
            feas_i,
            penal_i,
            cviol_i) = self.__evaluate_function_and_constraints(xi)

            objective1[i] = obj1_i
            objective2[i] = obj2_i
            feasibility[i] = feas_i
            penality[i] = penal_i
            constrViolation[i] = cviol_i

        return objective1,objective2,penality,feasibility,constrViolation

    def __sort_population(self,objective1,objective2,penality,feasibility,selection_size=None) : 
        values1 = objective1[:]
        values2 = objective2[:]
        unfeasibility = np.logical_not(feasibility)

        if self.__constraintMethod in ["penality","mixed"] and unfeasibility.any() :
            values1 = objective1-self.__penalityFactor*penality
            values2 = objective2-self.__penalityFactor*penality
        
        elif self.__constraintMethod == "feasibility" and unfeasibility.any() :
            o1min,o2min = objective1.min(),objective2.min()
            values1[unfeasibility] = o1min - 1
            values2[unfeasibility] = o2min - 1
        
        rank_pop = self.__fast_non_dominated_sort(values1,values2,selection_size)
        return rank_pop

    def __selection_tournament(self,population,npop,rank_K,fitness):
        ndof = self.__ndof
        selection = np.zeros((npop//2,ndof))
        for i in range(npop//2):
            (index_1,index_2) = rd.choice(npop-1,2)
            if rank_K[index_1]<rank_K[index_2] :
                select = index_1
            elif rank_K[index_1]>rank_K[index_2] :
                select = index_2
            elif fitness[index_1] >= fitness[index_2] :
                select = index_1
            else :
                select = index_2
            selection[i] = population[select]
        return selection

    def __selection_elitisme(self,parents_pop,
                                  parents_obj1,
                                  parents_obj2,
                                  parents_penality,
                                  parents_feasibility,
                                  parents_cviol,
                                  children_pop,
                                  children_obj1,
                                  children_obj2,
                                  children_penality,
                                  children_feasibility,
                                  children_cviol) :
        
        npop = len(children_obj1)
        nparents = len(parents_obj1)
        nelite = npop + nparents

        population = np.concatenate( (parents_pop,children_pop), axis=0)
        objective1 = np.concatenate( (parents_obj1, children_obj1) )
        objective2 = np.concatenate( (parents_obj2, children_obj2) )
        penality = np.concatenate( (parents_penality, children_penality) )
        feasibility = np.concatenate( (parents_feasibility, children_feasibility) )
        cviol = np.concatenate( (parents_cviol,children_cviol), axis=0)

        rank_pop = self.__sort_population(objective1,objective2,penality,feasibility,nelite)
        fitness = self.__crowning_distance(objective1,objective2,rank_pop)


        dtype = [('rank',int),('crowning',float)]
        values = np.array( list(zip(rank_pop,-fitness)) ,dtype=dtype)
        sorted_index = np.argsort(values,order=["rank","crowning"])

        

        new_population = population[sorted_index][:npop]
        new_objective1 = objective1[sorted_index][:npop]
        new_objective2 = objective2[sorted_index][:npop]
        # new_penality = penality[sorted_index][:npop]
        # new_feasibility = feasibility[sorted_index][:npop]
        # new_cviol = cviol[sorted_index][:npop]
        rank_pop = rank_pop[sorted_index]
        feasibility = feasibility[sorted_index]
        new_rank_pop = rank_pop[:npop]
        new_fitness = self.__crowning_distance(new_objective1,new_objective2,new_rank_pop)

        rank1_filter = rank_pop == 1
        if self.__constraintMethod in ["feasibility","mixed"] :
            rank1_filter &= feasibility
        
        parents_pop = population[sorted_index][rank1_filter]
        parents_obj1 = objective1[sorted_index][rank1_filter]
        parents_obj2 = objective2[sorted_index][rank1_filter]
        parents_penality = penality[sorted_index][rank1_filter]
        parents_feasibility = feasibility[sorted_index][rank1_filter]
        parents_cviol = cviol[sorted_index][rank1_filter]

        if len(parents_pop) > self.__nfront :
            parents_pop = parents_pop[:self.__nfront]
            parents_obj1 = parents_obj1[:self.__nfront]
            parents_obj2 = parents_obj2[:self.__nfront]
            parents_penality = parents_penality[:self.__nfront]
            parents_feasibility = parents_feasibility[:self.__nfront]
            parents_cviol = parents_cviol[:self.__nfront]


        return  (new_population,
                new_objective1,
                new_objective2,
                new_rank_pop,
                new_fitness,
                parents_pop,
                parents_obj1,
                parents_obj2,
                parents_penality,
                parents_feasibility,
                parents_cviol)


    # def __selection_elitisme(self,parents_pop,
    #                               parents_obj1,
    #                               parents_obj2,
    #                               parents_penality,
    #                               parents_feasibility,
    #                               parents_cviol,
    #                               children_pop,
    #                               children_obj1,
    #                               children_obj2,
    #                               children_penality,
    #                               children_feasibility,
    #                               children_cviol) :

    #     child_filter = children_feasibility[:]

    #     feasible_pop = np.concatenate((parents_pop,children_pop[child_filter]),axis=0)
    #     feasible_obj1 = np.concatenate( (parents_obj1,children_obj1[child_filter]) )
    #     feasible_obj2 = np.concatenate( (parents_obj2,children_obj2[child_filter]) )

    #     npop = len(children_pop)
    #     nfeasible = len(feasible_pop)

    #     if (nfeasible >= npop) :

    #         rank_pop = self.__fast_non_dominated_sort(feasible_obj1,feasible_obj2,npop)
    #         fitness = self.__crowning_distance(feasible_obj1,feasible_obj2,rank_pop)



    #         values = list(zip(rank_pop,-fitness))
    #         dtype = [('rank',int),('crowning',float)]
    #         evaluation_array = np.array(values,dtype=dtype)
    #         sorted_index = np.argsort(evaluation_array,order=["rank","crowning"])

    #         population = feasible_pop[sorted_index][:npop]
    #         objective1 = feasible_obj1[sorted_index][:npop]
    #         objective2 = feasible_obj2[sorted_index][:npop]
    #         rank_pop = rank_pop[sorted_index]
    #         fitness = self.__crowning_distance(objective1,objective2,rank_pop[:npop])

    #         parents_pop = feasible_pop[sorted_index][rank_pop==1]
    #         parents_obj1 = feasible_obj1[sorted_index][rank_pop==1]
    #         parents_obj2 = feasible_obj2[sorted_index][rank_pop==1]

    #         if len(parents_pop) > self.__nfront :
    #             parents_pop = parents_pop[:self.__nfront]
    #             parents_obj1 = parents_obj1[:self.__nfront]
    #             parents_obj2 = parents_obj2[:self.__nfront]
    #         rank_pop = rank_pop[:npop]


    #         return  (population,
    #                 objective1,
    #                 objective2,
    #                 rank_pop,
    #                 fitness,
    #                 parents_pop,
    #                 parents_obj1,
    #                 parents_obj2,
    #                 parents_penality*0.0,
    #                 parents_feasibility*0.0,
    #                 parents_cviol*0.0)


    #     else :
    #         nextend = npop-nfeasible
    #         notfeasible = np.logical_not(children_feasibility)
    #         notfeasible_pop = children_pop[notfeasible]
    #         notfeasible_obj1 = children_obj1[notfeasible]
    #         notfeasible_obj2 = children_obj2[notfeasible]

    #         rank_pop = self.__fast_non_dominated_sort(feasible_obj1,feasible_obj2,nfeasible)
    #         max_rank = max(rank_pop)
    #         parents_pop = feasible_pop[rank_pop==1]
    #         parents_obj1 = feasible_obj1[rank_pop==1]
    #         parents_obj2 = feasible_obj2[rank_pop==1]

    #         population = np.concatenate( (feasible_pop, notfeasible_pop[:nextend]), axis=0 )
    #         objective1 = np.concatenate( (feasible_obj1, notfeasible_obj1[:nextend]) )
    #         objective2 = np.concatenate( (feasible_obj2, notfeasible_obj2[:nextend]) )
    #         rank_pop = np.concatenate( (rank_pop, [max_rank+1]*nextend) )

    #         fitness = self.__crowning_distance(objective1,objective2,rank_pop)



    #         return  (population,
    #                 objective1,
    #                 objective2,
    #                 rank_pop,
    #                 fitness,
    #                 parents_pop,
    #                 parents_obj1,
    #                 parents_obj2,
    #                 parents_penality*0.0,
    #                 parents_feasibility*0.0,
    #                 parents_cviol*0.0)


    def __runOptimization(self,npop,ngen,nfront=None,verbose=True):
        """
        Run optimization
        """

        ## Initialisation
        if npop%2 != 0 :
            npop += 1
        ndof = self.__ndof

        self.__success = False
        self.__lastPop = None
        self.__xfront = None
        self.__f1front = None
        self.__f2front = None
        self.__frontsize = 0

        if self.__initial_population is None : 
            population = self.__initial_population_selector((npop,ndof)) #population initiale
        else : 
            population = self.__initial_population
            npop = len(population)

        if nfront is None :
            nfront = 2*npop
        self.__nfront = nfront

        xmin = self.__xmin
        xmax = self.__xmax

        objective1 = np.zeros(npop)
        objective2 = np.zeros(npop)

        startTime = time.time()

        Xarray = population*(xmax-xmin) + xmin

        (objective1,
        objective2,
        penality,
        feasibility,
        constrViolation) = self.__fitness_and_feasibility(Xarray,npop)

        rank_pop = self.__sort_population(objective1,objective2,penality,feasibility)
        fitness = self.__crowning_distance(objective1,objective2,rank_pop)
        parents_pop = population[feasibility]
        parents_obj1 = objective1[feasibility]
        parents_obj2 = objective2[feasibility]
        parents_penality = penality[feasibility]
        parents_feasibility = feasibility[feasibility]
        parents_cviol = constrViolation[feasibility]

        for generation in range(ngen):

            #Algorithme genetique
            
            fitness = self.sharingFunction(fitness,population)

            selection = self.selectionFunction(population,npop,rank_pop,fitness)

            children_pop = self.crossoverFunction(selection,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            children_pop = self.mutationFunction(children_pop,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            Xarray = children_pop*(xmax-xmin) + xmin
            (children_obj1,
            children_obj2,
            children_penality,
            children_feasibility,
            children_cviol) = self.__fitness_and_feasibility(Xarray,npop)

            #Elistisme
            (population,
            objective1,
            objective2,
            rank_pop,
            fitness,
            parents_pop,
            parents_obj1,
            parents_obj2,
            parents_penality,
            parents_feasibility,
            parents_cviol) = self.__selection_elitisme(parents_pop,
                                                    parents_obj1,
                                                    parents_obj2,
                                                    parents_penality,
                                                    parents_feasibility,
                                                    parents_cviol,
                                                    children_pop,
                                                    children_obj1,
                                                    children_obj2,
                                                    children_penality,
                                                    children_feasibility,
                                                    children_cviol)


            self.__lastPop = population*(xmax-xmin) + xmin
            # self.__archive_solution(parents_pop,parents_obj1,parents_obj2)
            # self.__archive_details(generation)

            if verbose :
                print('Iteration ',generation+1)



        endTime = time.time()
        duration = endTime-startTime
        self.__frontsize = parents_pop.shape[0]

        self.__success = parents_feasibility.all()
        self.__xfront = parents_pop*(xmax-xmin) + xmin
        self.__f1front = parents_obj1*self.__sign_factor[0]
        self.__f2front = parents_obj2*self.__sign_factor[1]
        self.__feasibility_front = parents_feasibility
        self.__cviol_front = parents_cviol
            
        ##MESSAGES

        print('\n'*2+'#'*60+'\n')
        print('AG iterations completed')
        print('Number of generations : ',ngen)
        print('Population size : ',npop)
        print('Elapsed time : %.3f s'%duration)

        print('#'*60+'\n')







    def optimize(self,npop,ngen,nfront=None,verbose=False,returnDict=False):
        """
        Algorithme d'optimisation bi-objectifs sous contrainte. 
        Parameters : 
            - npop (int) : 
                Taille de la population. Si npop est impair, l'algorithm 
                l'augmente de 1. Usuellement pour un probleme sans contrainte 
                une population efficace est situee entre 5 et 20 fois le 
                nombre de variable. Si les contraintes sont fortes, il sera 
                utile d'augmenter la population. Ce parametre n'est pas pris 
                en compte si une population initiale a ete definie.
            - ngen (int) : 
                Nombre de generation. Usuellement une bonne pratique est de 
                prendre 2 à 10 fois la taille de la population. 
            - nfront (int or None) option : 
                Nombre de point maximum dans le front de Pareto. Si None, 
                alors nfront = 3*npop
            - verbose (bool) option : 
                Affiche l'etat de la recherche pour chaque iteration. 
                Peut ralentir l'execution.
            - returnDict (bool) option : 
                Si l'option est True alors l'algorithme retourne un 
                dictionnaire. 
            
        Returns : 
            Si (returnDict = False) : 
                tuple : (  xsolutions, objective1_solutions, 
                            objective2_solutions )
                        (array(ndof,nfront), array(nfront), array(nfront))
                        
                    - xsolutions sont les solutions composants le front 
                      de Pareto.
                    - objective1_solutions sont les points de la fonction 1 
                      dans le front de Pareto. 
                    - objective2_solutions sont les points de la fonction 2 
                      dans le front de Pareto.
            
            Si (returnDict = True) : 
                dict :
                    "method" (str) : algorithm utilise.
                    "optimization" (str) : minimisation ou maximisation.
                    "success" (bool) : True si l'algorithm a converge.
                    "x" (array or None) : solutions du front de Pareto.
                    "f1" (array or None) : front de Pareto f1.
                    "f2" (array or None) : front de Pareto f2.
        """
        self.__runOptimization(npop, ngen, nfront=nfront, verbose=verbose)

        if returnDict : 
            result = {"method":"Continous Bi-Objective NSGA",
                      "optimization" : "Pareto",
                      "success":self.__success,
                      "x" : self.__xfront,
                      "f1" : self.__f1front,
                      "f2" : self.__f2front
                      }
            return result
        else : 
            return self.__xfront, self.__f1front, self.__f2front
    
    def getLastPopulation(self) : return self.__lastPop
    
    def getConstraintViolation(self) : return self.__cviol_front


if __name__ == '__main__' : 

    
    import numpy as np 
    import matplotlib.pyplot as plt 
    
    ### GA exemple 
    ga_instance = continousSingleObjectiveGA(lambda x : x[0]**2 + x[1]**2,
                                            [-1,-1],
                                            [1,1])


    resGA = ga_instance.minimize(20,100,verbose=False,returnDict=True)

    for ri in resGA : 
         print(ri," : ",resGA[ri])

    ### NSGA exemple
    xmin,xmax = [-2],[2]
    x = np.linspace(xmin[0],xmax[0],150)
    f1 = lambda x : (0.5*x**2+x)/4
    f2 = lambda x : (0.5*x**2-x)/4
    c = lambda x : np.sin(x)
    cons = [{"type":'ineq','fun':c}]
    nsga_instance = continousBiObjective_NSGA(f1,
                                              f2,
                                              xmin,
                                              xmax,
                                              constraints=cons,
                                              func1_criterion='min',
                                              func2_criterion='min')
                                              
    xfront,f1front,f2front = nsga_instance.optimize(20,100,verbose=False)

    plt.figure(1,figsize=(7,4))

    plt.subplot(121)
    plt.plot(x,f1(x),label='f1(x)',color='r')
    plt.plot(x,f2(x),label='f2(x)',color='b')
    plt.plot(x,c(x),label='Constraint',color='k',ls='--')
    plt.plot(xfront,f1front,label='front f1',marker='o',ls='',markerfacecolor="None",markeredgecolor='r')
    plt.plot(xfront,f2front,label='front f2',marker='o',ls='',markerfacecolor="None",markeredgecolor='b')
    plt.grid(True)
    plt.legend()
    plt.xlabel("Variable x")
    plt.ylabel("Function values")

    feasible = c(x)>=0.0

    plt.subplot(122)
    plt.plot(f1(x)[feasible],f2(x)[feasible],label='feasible',marker='.',ls='',color='k')
    plt.plot(f1(x)[~feasible],f2(x)[~feasible],label='not feasible',marker='x',ls='',color='grey')
    plt.plot(f1front,f2front,label='Pareto front',marker='o',ls='',markerfacecolor="orange",markeredgecolor='k',alpha=0.9)
    plt.grid(True)
    plt.legend()
    plt.xlabel("f1(x)")
    plt.ylabel("f2(x)")

    plt.tight_layout()

    plt.show()