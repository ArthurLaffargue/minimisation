# coding: utf-8

import numpy as np
import numpy.random as rd
from numpy.linalg import norm
import time

from _utils import uniform_init_population_lhs,uniform_init_population_random


class genetic_operator : 
    _mutation = {"normal" : "_normalMutation",
                 "uniform" : "_uniformMutation"}

    _crossover = {"SBX" : "_SBXcrossover",
                  "uniform" : "_uniformCrossover"}
    
    _constraint_handling = ["penality","feasibility","mixed"]

    def __init__(self,ndof,mutation_method,mutation_rate,mutation_step,
                    crossover_method,eta_cross,
                    sharing_distance,
                    constraintMethod) : 
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

        #Contraintes 
        if constraintMethod in self._constraint_handling : 
            self.__constraintMethod = constraintMethod
        else : 
            raise ValueError("Please select a valid constraint handlling strategy.")


    def _uniformCrossover(self,selection,population,npop):
        """
        Operateur de croisement barycentrique
        """

        couples = np.zeros((npop//2,2,self.__ndof))
        children = np.zeros_like(population)
        alphaCross = rd.sample((npop//2,self.__ndof))*self.__crossFactor
        for i in range(npop//2):
            k = i
            while k == i :
                k = rd.randint(0,npop//2-1)
            couples[i] = [selection[i],selection[k]]
        children[:npop//2] = alphaCross*couples[:,0] + (1-alphaCross)*couples[:,1]
        children[npop//2:] = alphaCross*couples[:,1] + (1-alphaCross)*couples[:,0]

        return children
    
    def _SBXcrossover(self,selection,population,npop):
        couples = np.zeros((npop//2,2,self.__ndof))
        children = np.zeros_like(population)
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

    def __init__(self,func,xmin,xmax,constraints=[],preprocess_function=None,
                    mutation_method="normal",mutation_rate=0.10,mutation_step=0.10,
                    crossover_method="SBX",eta_cross=1.0,
                    sharing_distance=None,
                    atol = 0.,tol = 1e-3,stagThreshold = None,
                    eqcons_atol = 1e-3,penalityFactor = 1e3,constraintMethod = "penality",
                    elitisme = True,
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

        #Operateurs genetiques
        super().__init__(self.__ndof,
                        mutation_method,mutation_rate,mutation_step,
                        crossover_method,eta_cross,
                        sharing_distance,
                        constraintMethod)
        self.selectionFunction = self.__selection_tournament

        #convergence 
        self.__atol = atol
        self.__tol = tol
        self.__stagThreshold = stagThreshold

        self.__constraintMethod = self._genetic_operator__constraintMethod
        self.__constraintAbsTol = eqcons_atol
        self.__penalityFactor = penalityFactor

        self.__elitisme = elitisme


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
        
        self.__optiObj = None
        self.__optiX = None
        self.__optiX_scaled = None
        self.__optiFeasible = False
        self.__optiPenality = None

        self.__success = False
        self.__lastPop = None
        self.__constrViolation = []
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
                penality += constrViol**2
        penalObjective = objective - self.__penalityFactor*penality
        return objective,penalObjective,feasibility,penality,constrViolation

    def __fitness_and_feasibility(self,Xarray,objective,npop) :
        """
        Evaluation de la fonction objectif et des contraintes
        """

        feasibility = np.ones(npop,dtype=bool)
        penality = np.zeros(npop,dtype=float)
        penalObjective = np.zeros(npop,dtype=float)
        constraintViol = []

        for i,xi, in enumerate(Xarray) :
            (obj_i,
            pobj_i,
            feas_i,
            penal_i,
            cviol_i) = self.__evaluate_function_and_constraints(xi)

            objective[i] = obj_i
            penalObjective[i] = pobj_i
            feasibility[i] = feas_i
            penality[i] = penal_i
            constraintViol.append(cviol_i)

        if self.__constraintMethod == 'penality' : 
            omin = penalObjective.min()
            omax = penalObjective.max()
            fitness = (penalObjective-omin)/(omax-omin)

        if self.__constraintMethod == "feasibility" :
            omin = objective.min()
            omax = objective.max()
            fitness = (objective-omin)/(omax-omin)*feasibility

        if self.__constraintMethod == "mixed" : 
            omin = penalObjective.min()
            omax = penalObjective.max()
            fitness = (penalObjective-omin)/(omax-omin)
        
        if not( feasibility.all() ) :
            constraintViol = np.array(constraintViol)
            cmax = constraintViol.max(axis=0)
            cmax[cmax==0.0] = 1.0
            distviol = norm(constraintViol/cmax,axis=1)
        else : 
            distviol = np.zeros(npop,dtype=float)
        return objective,fitness,feasibility,penality,distviol

    def __constraints_ranking(self,fitness,distviol,feasibility):
        if feasibility.all() or self.__constraintMethod == "penality" or self.__constraints == []: 
            return np.argsort(fitness).argsort()
        else : 
            ranking_values = np.array( list(zip(fitness,-distviol)) ,
                                        dtype=[('fitness', float), ('dviol', float)])
            ranking = np.argsort(ranking_values,order=("dviol","fitness")).argsort()
            return ranking

    def __selection_tournament(self,population,npop,fitness,distviol,feasibility):
        """
        Operateur de selection par tournois
        """
        rank = self.__constraints_ranking(fitness,distviol,feasibility)
        selection = np.zeros((npop//2,self.__ndof))
        for i in range(npop//2):
            indices = rd.choice(npop-1,2)
            selection[i] = population[indices[np.argmax(rank[indices])]]
        return selection

    def __selection_elitisme(self,elite_pop,elite_obj,children_pop,children_obj,children_dviol,children_feasible,npop) :
        """
        Operateur elitisme
        """
        nelite = elite_obj.shape[0]
        elite_feasible = np.full(nelite,True,dtype=bool)
        elite_distviol = np.full(nelite,0.0,dtype=float)

        if self.__optiX is None :
            pop = np.concatenate((elite_pop,children_pop), axis=0)
            obj = np.concatenate( (elite_obj,children_obj) )
            feasible = np.concatenate( (elite_feasible,children_feasible) )
            distviol = np.concatenate( (elite_distviol,children_dviol) )
        else :
            pop =  np.concatenate( (elite_pop,children_pop,[self.__optiX_scaled]), axis=0)
            obj = np.concatenate( (elite_obj,children_obj,[self.__optiObj]) )
            feasible = np.concatenate( (elite_feasible,children_feasible,[self.__optiFeasible]) )
            distviol = np.concatenate( (elite_distviol,children_dviol,[0.0]) )

        if self.__constraintMethod in ['feasibility',"mixed"] :
            rank = self.__constraints_ranking(obj,distviol,feasible)
            
        elif self.__constraintMethod == 'penality' :
            rank = np.argsort(obj).argsort()

        select = (rank.shape[0] - rank - 1)<npop
        elit_select =  feasible & ( (rank.shape[0] - rank - 1)<2*npop )
        population = pop[select]
        objective = obj[select]
        elite_pop = pop[elit_select]
        elite_obj = obj[elit_select]

        omin,omax = objective.min(),objective.max()
        fitness = (objective-omin)/(omax-omin)

        return population,objective,elite_pop,elite_obj,fitness

    def __archive_solution(self,population,objective,penality,feasibility):
        """
        Archive la solition de meilleur objectif
        """
        better_solution = False
        if self.__optiObj is None :
            selection = np.ones_like(feasibility,dtype=bool)
        else :
            selection = (penality <= self.__optiPenality)&(objective > self.__optiObj) 
            

        if selection.any() : 
            indexmax = np.argmax(objective[selection])

            self.__optiX_scaled = population[selection][indexmax]
            self.__optiX = self.__optiX_scaled*(self.__xmax-self.__xmin)+self.__xmin
            self.__optiObj = objective[selection][indexmax]
            self.__optiFeasible = feasibility[selection][indexmax]
            self.__optiPenality = penality[selection][indexmax]

            better_solution = True
            
        return better_solution

    def __archive_details(self,generation):
        """
        Archive une liste de la meilleur solution par iteration
        """
        if self.__optiObj is not None : 
            self.__statOptimisation[generation] = self.__optiObj*self.__sign_factor
        else : 
            self.__statOptimisation[generation] = None

    def __checkConvergenceState(self,generation,last_improvement,objective) : 
        
        c1 = False
        if self.__stagThreshold is not None : 
            c1 = (generation - last_improvement) > self.__stagThreshold
        
        c2 = ( np.std(objective) ) <= ( self.__atol + self.__tol*np.abs(np.mean(objective)) )
            
        converged = (c2 or c1) and self.__optiFeasible

        return converged
    


    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         ALGORITHME
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###

    def __runOptimization(self,npop,ngen,verbose=True):
        """
        Optimisation
        """

        ## Initialisation
        

        # if not(self.__elitisme) :
        #     self.__constraintMethod = "penality"


        if npop%2 != 0 :
            npop += 1
        ndof = self.__ndof

        if self.__initial_population is None : 
            population = self.__initial_population_selector((npop,ndof)) #population initiale
        else : 
            population = self.__initial_population
            npop = len(population)

        # if self.__sharingDist is None :
        #     self.__sharingDist = 1/npop

        xmin = self.__xmin
        xmax = self.__xmax
        
        self.__optiObj = None
        self.__optiX = None
        self.__optiX_scaled = None
        self.__optiFeasible = False
        self.__success = False
        self.__lastPop = None
        last_improvement = 0

        self.__statOptimisation  = np.full(ngen,None,dtype=float)
        objective = np.zeros(npop)


        startTime = time.time()

        Xarray = population*(xmax-xmin) + xmin
        objective,fitness,feasibility,penality,distviol = self.__fitness_and_feasibility(Xarray,objective,npop)
        parents_pop = population[feasibility]
        parents_obj = objective[feasibility]
        for generation in range(ngen):

            #Algorithme genetique
            fitness = self.sharingFunction(fitness,population)
            
            selection = self.selectionFunction(population,npop,fitness,distviol,feasibility)

            children_pop = self.crossoverFunction(selection,population,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            children_pop = self.mutationFunction(children_pop,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            Xarray = children_pop*(xmax-xmin) + xmin
            children_obj,fitness,feasibility,penality,distviol = self.__fitness_and_feasibility(Xarray,objective,npop)

            #Elistisme
            if self.__elitisme :
                (population,
                objective,
                parents_pop,
                parents_obj,
                fitness) = self.__selection_elitisme(parents_pop,
                                                    parents_obj,
                                                    children_pop,
                                                    children_obj,
                                                    distviol,
                                                    feasibility,
                                                    npop)
            else :
                population = children_pop[:]
                objective = children_obj[:]

            better_sol = self.__archive_solution(children_pop,children_obj,penality,feasibility)
            self.__archive_details(generation)

            if better_sol : 
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
        if self.__optiX is not None : 
            self.__success = self.__optiFeasible

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
                      "x" : self.__optiX,
                      "f" : self.__optiObj*self.__sign_factor
                      }
            return result
        else : 
            return self.__optiX,self.__optiObj*self.__sign_factor
    
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
                      "x" : self.__optiX,
                      "f" : self.__optiObj*self.__sign_factor
                      }
            return result
        else : 
            return self.__optiX,self.__optiObj*self.__sign_factor
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         RENVOIS
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###

    def getLastPopulation(self) : return self.__lastPop

    def getStatOptimisation(self): return self.__statOptimisation



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