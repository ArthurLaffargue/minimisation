import numpy as np
from numpy.lib.utils import _set_function_name
import numpy.random as rd
import time

class continousSingleObjectiveGA : 

    def __init__(self,func,xmin,xmax,constraints=[],preprocess_function=None) :
        """
        Instance de continousSingleObjectiveGA : algorithme genetique d'optimisation de fonction mono-objectif à variables reelles
        Recherche d'un optimum global de la fonction f sur les bornes [xmin;xmax]. 
        Parameters : 
            - func (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est l'argument de la fonction de forme scalaire ou array et renvoie un scalaire ou array.

            - xmin (array like) : 
                Borne inferieure des variables d'optimisation. 

            - xmax (array like) : 
                Borne supérieure des variables d'optimisation. Doit être de même dimension que xmin. 

            - constraints (List of dict) option : 
                Definition des contraintes dans une liste de dictionnaires. Chaque dictionnaire contient les champs suivants 
                    type : str
                        Contraintes de type egalite 'eq' ou inegalite 'ineq' ; 
                    fun : callable
                        La fonction contrainte de la meme forme que func ; 

            - preprocess_function (callable or None) option : 
                Definition d'une fonction sans renvoie de valeur à executer avant chaque evaluation de la fonction objectif func ou des contraintes.
        """

        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)

        self.__ndof = len(self.__xmin)

        self.__preProcess = preprocess_function
        self.__function = func
        self.__constraints = constraints

        self.__nPreSelected = 2 #Nombre d'individus preselectionne pour tournois
        self.__stepMut = 0.15
        self.__rateMut = 0.25
        self.__crossFactor = 1.25
        self.__constraintAbsTol = 1e-3
        self.__penalityFactor = 1e3
        self.__penalityGrowth = 1.0
        self.__sharingDist = None
        self.__constraintMethod = "penality" #"feasibility"
        self.__elitisme = True

        self.__initial_population = None 

        self.__optiObj = None
        self.__optiX = None
        self.__success = False
        self.__lastPop = None
        self.__constrViolation = []
        self.__statOptimisation = None

        self.__sign_factor = 1.0

        self.__selection_function = self.__selection_tournament


    def setPreSelectNumber(self,nbr_preselected=2):
        """
        Change le nombre d'individus pre-selectionnes dans un mode de selection par tournois. 
        Parameter : 
            - nbr_preselected (int) option : nombre d'individus participants à chaque tournois. Superieur a 1.
        """
        self.__nPreSelected = nbr_preselected

    def setMutationParameters(self,mutation_step=0.15,mutation_rate=0.25) : 
        """
        Change les parametres de mutation. 
        Parameters : 
            - mutation_step (float) option : limite de deplacement par mutation. Doit etre compris entre 0 et 1.

            - mutation_rate (float) option : probabilite de mutation. Doit etre compris entre 0 et 1.
        """

        self.__stepMut = mutation_rate
        self.__rateMut = mutation_rate


    def setCrossFactor(self,crossover_factor=1.25):
        """
        Change la limite de barycentre dans l'operateur de reproduction. Si egale a 1, les enfants seront strictement entre les parents. 
        Si superieur a 1, les enfants peuvent etre a l'exterieur du segment parent. 
        Parameter : 
            - crossover_factor (float) option : facteur de melange des individus parents ; 
        """
        self.__crossFactor = crossover_factor

    def setSharingDist(self,sharingDist=None) :
        """
        Change le rayon de diversite des solutions.
        Parameter : 
            - sharingDict (float or None) option : Rayon de diversite de solution. Si None, le parametre est initialise a 1/(taille population).
        """
        self.__sharingDist = sharingDist

    def setSelectionMethod(self,method="tournament"):
        """
        Change la methode de selection. 
        Parameter : 
            - method (str) option : Definition de la methode de selection. 
                Si method = 'tournament' : selection par tournois. 
                Si method = 'SRWRS' : selection par "Stochastic remainder without replacement selection" [Golberg]
        
        [Golberg]    D.E Goldberg. Genetic Algorithms in Search, Optimization and Machine Learning. Reading MA Addison Wesley, 1989.
        """
        if method == "SRWRS" :
            self.__selection_function = self.__selection_SRWRS
        if method == "tournament" :
            self.__selection_function = self.__selection_tournament

    def setConstraintMethod(self,method="penality"):
        """
        Change la methode de prise en compte des contraintes. 
        Parameter : 
            method (str) option : Definition de la methode de gestion des contraintes. 
                Si method = 'penality' utilisation d'une penalisation quadratique. 
                Si method = 'feasibility' les solutions non satisfaisante sont rejetees.
        """
        if method == "feasibility" or method == "penality" :
            self.__constraintMethod = method

    def setPenalityParams(self,constraintAbsTol=1e-3,penalityFactor=1e3,penalityGrowth=1.00):
        """
        Change le parametrage de la penalisation de contrainte. 
        Parameters : 
            - contraintAbsTol (float) option : tolerance des contraintes d'egalite. Si la contrainte i ||ci(xk)|| <= contraintAbsTol, la solution est viable. 

            - penalityFactor (float) option : facteur de penalisation. La nouvelle fonction objectif est evaluee par la forme suivante :
                                                    penal_objective_func = objective_func + sum(ci**2 if ci not feasible)*penalityFactor
                                                    objectif_penalise = objectif + somme( ci**2 si ci non faisable)*facteur_penalite
                                            
            - penalityGrowth (float) option : pour chaque iteration de l'algorithme le facteur de penalite peut croitre d'un facteur penalityGrowth
                                                    penalityFactor_(k+1) = penalityFactor_(k)*penalityGrowth

        """
        self.__constraintAbsTol = constraintAbsTol
        self.__penalityFactor = penalityFactor
        self.__penalityGrowth = penalityGrowth
        self.__constraintMethod = "penality"

    def setElitisme(self,elitisme=True):
        """
        Booleen d'activation d'un operateur d'elitsime herite de la methode NSGA-II. 
        L'elitisme melange les populations parents et enfants pour en extraire les meilleurs individus. 
        Si cette option est desactivee, la methode de contrainte par faisabilite est impossible. L'algorithme utilisera une penalite.
        Parameter : 
            - elitisme (bool) option : actif (True) ou inactif (False)

        """
        self.__elitisme = elitisme

    def redefine_objective(self,func):
        """
        Permet de redefinir la fonction objectif. 
        Parameter : 
            - func (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est l'argument de la fonction de forme scalaire ou array et renvoie un scalaire ou array.
        """
        self.__function = func

    def redefine_constraints(self,constraints=[]):
        """
        Permet de redefinir les contraintes du probleme. 
        Parameter : 
            - constraints (List of dict) option : 
                    Definition des contraintes dans une liste de dictionnaires. Chaque dictionnaire contient les champs suivants 
                        type : str
                            Contraintes de type egalite 'eq' ou inegalite 'ineq' ; 
                        fun : callable
                            La fonction contrainte de la meme forme que func ; 
        """
        self.__constraints = constraints
    

    def redefine_preprocess_func(self,preprocess_function=None):
        """
        Permet de redefinir la fonction de preprocessing. 
        Parameter
            - preprocess_function (callable or None) option : 
                Definition d'une fonction sans renvoie de valeur à executer avant chaque evaluation de la fonction objectif func ou des contraintes.
        """
        self.__preProcess = preprocess_function


    def define_initial_population(self,xstart):
        """
        Definition d'une population initiale. 
        Parameter : 
            - xstart (array(npop,ndof)) : 
                xstart est la solution initiale. Ses dimensions doivent etre de (npop,ndof). 
                npop la taille de la population et ndof le nombre de variable (degrees of freedom).
        """

        x = np.array(xstart)
        npop,ndof = x.shape
        if ndof != self.__ndof : 
            raise ValueError("The size of initial population is not corresponding to bounds size")

        xpop = np.maximum(np.minimum((x-self.__xmin)/(self.__xmax-self.__xmin),1.0),0.0)
        if npop%2 != 0 : 
            xpop = np.insert(xpop,0,0.5,axis=0)

        self.__initial_population = xpop



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
                penality += self.__penalityFactor*constrViol**2
        penalObjective = objective - penality
        return objective,penalObjective,feasibility,penality,constrViolation

    def __fitness_and_feasibility(self,Xarray,objective,npop) :
        """
        Evaluation de la fonction objectif et des contraintes
        """

        feasibility = np.ones(npop,dtype=bool)
        penality = np.zeros(npop,dtype=float)
        penalObjective = np.zeros(npop,dtype=float)
        for i,xi, in enumerate(Xarray) :
            (obj_i,
            pobj_i,
            feas_i,
            penal_i,_) = self.__evaluate_function_and_constraints(xi)

            objective[i] = obj_i
            penalObjective[i] = pobj_i
            feasibility[i] = feas_i
            penality[i] = penal_i

        if self.__constraintMethod == 'penality' : 
            omin = penalObjective.min()
            omax = penalObjective.max()
            fitness = (penalObjective-omin)/(omax-omin)
        if self.__constraintMethod == "feasibility" :
            omin = objective.min()
            omax = objective.max()
            fitness = (objective-omin)/(omax-omin)
            fitness *= feasibility
        self.__penalityFactor *= self.__penalityGrowth
        return objective,fitness,feasibility,penality

    def __barycenterCrossover(self,selection,population,npop):
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

        return population

    def __sharingFunction(self,fitness,population):
        """
        Operateur de diversite de solution
        """
        dShare = self.__sharingDist

        distance = np.array([np.sqrt(np.sum((population - xj)**2,axis=1)) for xj in population])
        sharing = (1-distance/dShare)*(distance<dShare)
        sharingFactor = np.sum(sharing,axis=1)
        fitness = fitness/sharingFactor


        fmin = fitness.min()
        fmax = fitness.max()
        fitness = (fitness-fmin)/(fmax-fmin)
        return fitness

    def __selection_tournament(self,population,npop,fitness):
        """
        Operateur de selection par tournois
        """
        ndof = self.__ndof
        selection = np.zeros((npop//2,ndof))
        for i in range(npop//2):
            indices = rd.choice(npop-1,self.__nPreSelected)
            selection[i] = population[indices[np.argmax(fitness[indices])]]
        return selection

    def __selection_SRWRS(self,population,npop,fitness) :
        """
        Operateur de selection SRWRS
        """
        ndof = self.__ndof
        r_array = fitness/fitness.mean()
        index_list = []
        prob_list = []
        for i,ri in enumerate(r_array):
            eri = int(ri)
            index_list += [i]*(eri+1)
            prob_list += [ri-eri]*(eri+1)

        prob_list = np.array(prob_list)/np.sum(prob_list)
        index_list = np.array(index_list,dtype=int)
        index_select = rd.choice(index_list,npop//2,p=prob_list)
        selection = population[index_select]
        return selection

    def __mutation_delta(self,population,npop) :
        """
        Operateur de mutation
        """
        probaMutation = rd.sample((npop,self.__ndof))
        deltaX = self.__stepMut*(rd.sample((npop,self.__ndof))-0.5)
        population = population + deltaX*(probaMutation<=self.__rateMut)
        return population

    def __selection_elitisme(self,parents_pop,parents_obj,children_pop,children_obj,children_penal,feasibility) :
        """
        Operateur elitisme
        """

        if self.__optiX is None :
            feasible_pop = np.array([pi for pi in parents_pop]+[ci for ci in children_pop[feasibility]])
            feasible_obj = np.array([pi for pi in parents_obj]+[ci for ci in children_obj[feasibility]])
        else :
            feasible_pop = np.array([pi for pi in parents_pop]+[ci for ci in children_pop[feasibility]]+[self.__optiX])
            feasible_obj = np.array([pi for pi in parents_obj]+[ci for ci in children_obj[feasibility]]+[self.__optiObj])

        npop = len(children_pop)
        nfeasible = len(feasible_pop)


        if nfeasible >= npop :
            omin,omax = feasible_obj.min(),feasible_obj.max()
            fitness = (feasible_obj-omin)/(omax-omin)
            fitness = self.__sharingFunction(fitness,feasible_pop)
            sorted_index = np.argsort(fitness)[::-1]
            population = feasible_pop[sorted_index[:npop]]
            objective = feasible_obj[sorted_index[:npop]]


            omin = objective.min()
            omax = objective.max()
            fitness = (objective-omin)/(omax-omin)

            return population,objective,population,objective,fitness


        else :
            nextend = npop-nfeasible
            notfeasible = np.logical_not(feasibility)
            notfeasible_pop = children_pop[notfeasible]
            notfeasible_obj = children_obj[notfeasible]
            notfeasible_penal = children_penal[notfeasible]
            penalObj = notfeasible_obj - notfeasible_penal
            sortedIndex = np.argsort(penalObj)[::-1]
            sortedIndex = sortedIndex[:nextend]

            population = np.array([xi for xi in feasible_pop]+[xi for xi in notfeasible_pop[sortedIndex]])
            objective = np.array([xi for xi in feasible_obj]+[xi for xi in notfeasible_obj[sortedIndex]])

            if self.__constraintMethod == 'penality' : 
                penality = np.array([0.0 for xi in feasible_obj]+[xi for xi in notfeasible_penal[sortedIndex]])
                penalObj = objective - penality
                omin = penalObj.min()
                omax = penalObj.max()
                fitness = (penalObj-omin)/(omax-omin)
            else : 
                omin = objective.min()
                omax = objective.max()
                fitness = (objective-omin)/(omax-omin)

            parents_obj = feasible_obj[:]
            parents_pop = feasible_pop[:]
            return population,objective,feasible_pop,feasible_obj,fitness

    def __archive_solution(self,parents_pop,parents_obj):
        """
        Archive la solition de meilleur objectif
        """
        if len(parents_pop) > 0 :
            indexmax = np.argmax(parents_obj)
            maxObj = parents_obj[indexmax]

            if self.__optiObj is None :
                self.__optiX = parents_pop[indexmax]*(self.__xmax-self.__xmin)+self.__xmin
                self.__optiObj = parents_obj[indexmax]

            else :
                if self.__optiObj < maxObj :
                    self.__optiX = parents_pop[indexmax]*(self.__xmax-self.__xmin)+self.__xmin
                    self.__optiObj = parents_obj[indexmax]

    def __archive_details(self,generation):
        """
        Archive une liste de la meilleur solution par iteration
        """
        if self.__optiObj is not None : 
            self.__statOptimisation[generation] = self.__optiObj*self.__sign_factor
        else : 
            self.__statOptimisation[generation] = None

    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         ALGORITHME
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###

    def __runOptimization(self,npop,ngen,verbose=True):
        """
        Optimisation
        """

        ## Initialisation
        

        if not(self.__elitisme) :
            self.__constraintMethod = "penality"


        if npop%2 != 0 :
            npop += 1
        ndof = self.__ndof

        if self.__initial_population is None : 
            population = rd.sample((npop,ndof)) #population initiale
        else : 
            population = self.__initial_population
            npop = len(population)

        if self.__sharingDist is None :
            self.__sharingDist = 1/npop

        xmin = self.__xmin
        xmax = self.__xmax
        
        self.__optiObj = None
        self.__optiX = None
        self.__success = False
        self.__constrViolation = []
        self.__lastPop = None

        self.__statOptimisation  = np.zeros(ngen)
        objective = np.zeros(npop)


        startTime = time.time()

        Xarray = population*(xmax-xmin) + xmin
        objective,fitness,feasibility,penality = self.__fitness_and_feasibility(Xarray,objective,npop)
        parents_pop = population[feasibility]
        parents_obj = objective[feasibility]
        for generation in range(ngen):

            #Algorithme genetique

            fitness = self.__sharingFunction(fitness,population)

            selection = self.__selection_function(population,npop,fitness)

            children_pop = self.__barycenterCrossover(selection,population,npop)

            children_pop = self.__mutation_delta(children_pop,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            Xarray = children_pop*(xmax-xmin) + xmin
            children_obj,fitness,feasibility,penality = self.__fitness_and_feasibility(Xarray,objective,npop)

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
                                                    penality,
                                                    feasibility)
            else :
                population = children_pop
                objective = children_obj



            self.__archive_solution(children_pop[feasibility],children_obj[feasibility])
            self.__archive_details(generation)

            if verbose :
                print('Iteration ',generation+1)





        Xarray = children_pop[feasibility]*(xmax-xmin) + xmin
        endTime = time.time()
        duration = endTime-startTime

        ##MESSAGES
        if self.__optiX is not None : 
            self.__success = True
            _,_,_,_,constrViolation = self.__evaluate_function_and_constraints(self.__optiX)
            self.__constrViolation = constrViolation

        print('\n'*2+'#'*60+'\n')
        print('AG iterations completed')
        print("Success : ", self.__success)
        print('Number of generations : ',ngen)
        print('Population size : ',npop)
        print('Elapsed time : %.3f s'%duration)

        print('#'*60+'\n')

        self.__lastPop = Xarray

        
    def minimize(self,npop,ngen,verbose=True,returnDict=False):
        """
        Algorithme de minimisation de la fonction objectif sous contrainte. 
        Parameters : 
            - npop (int) : 
                Taille de la population. Si npop est impair, l'algorithm l'augmente de 1. 
                Usuellement pour un probleme sans contrainte une population efficace est situee entre 5 et 20 fois le nombre de variable.
                Si les contraintes sont fortes, il sera utile d'augmenter la population.
                Ce parametre n'est pas pris en compte si une population initiale a ete definie.

            - ngen (int) : 
                Nombre de generation. Usuellement une bonne pratique est de prendre 2 à 10 fois la taille de la population. 

            - verbose (bool) option : 
                Affiche l'etat de la recherche pour chaque iteration. Peut ralentir l'execution.

            - returnDict (bool) option : 
                Si l'option est True alors l'algorithme retourne un dictionnaire. 
            
        Returns : 
            Si (returnDict = False) : 
                tuple : xsolution, objective_solution (array(ndof), array(1)) ou (None, None)
                    - xsolution est la meilleure solution x historisee. Sa dimension correspond a ndof, la taille du probleme initial.
                    - objective_solution est la fonction objectif evaluee à xsolution. 
                    Si la solution n'a pas convergee et les contraintes jamais validee, l'algorithme return (None, None)
            
            Si (returnDict = False) : 
                dict :
                    "method" (str) : algorithm utilise.
                    "optimization" (str) : minimisation ou maximisation.
                    "success" (bool) : True si l'algorithm a converge, False sinon. 
                    "x" (array or None) : meilleure solution ou None si success = False.
                    "f" (array or None) : meilleure objectif ou None si success = False. 
                    "constrViolation" (List of float) : violation des contraintes. Liste vide si aucune contrainte.
        """
        self.__sign_factor = -1.0
        self.__runOptimization(npop, ngen, verbose=verbose)

        if returnDict : 
            result = {"method":"Continous Single Objective Genetic Algorithm",
                      "optimization" : "minimization",
                      "success":self.__success,
                      "x" : self.__optiX,
                      "f" : self.__optiObj*self.__sign_factor,
                      "constrViolation" : self.__constrViolation
                      }
            return result
        else : 
            return self.__optiX,self.__optiObj*self.__sign_factor
    
    def maximize(self,npop,ngen,verbose=True,returnDict=False):
        """
        Algorithme de maximisation de la fonction objectif sous contrainte. 
        Parameters : 
            - npop (int) : 
                Taille de la population. Si npop est impair, l'algorithm l'augmente de 1. 
                Usuellement pour un probleme sans contrainte une population efficace est situee entre 5 et 20 fois le nombre de variable.
                Si les contraintes sont fortes, il sera utile d'augmenter la population.
                Ce parametre n'est pas pris en compte si une population initiale a ete definie.

            - ngen (int) : 
                Nombre de generation. Usuellement une bonne pratique est de prendre 2 à 10 fois la taille de la population. 

            - verbose (bool) option : 
                Affiche l'etat de la recherche pour chaque iteration. Peut ralentir l'execution.

            - returnDict (bool) option : 
                Si l'option est True alors l'algorithme retourne un dictionnaire. 
            
        Returns : 
            Si (returnDict = False) : 
                tuple : xsolution, objective_solution (array(ndof), array(1)) ou (None, None)
                    - xsolution est la meilleure solution x historisee. Sa dimension correspond a ndof, la taille du probleme initial.
                    - objective_solution est la fonction objectif evaluee à xsolution. 
                    Si la solution n'a pas convergee et les contraintes jamais validee, l'algorithme return (None, None)
            
            Si (returnDict = False) : 
                dict :
                    "method" (str) : algorithm utilise.
                    "optimization" (str) : minimisation ou maximisation.
                    "success" (bool) : True si l'algorithm a converge, False sinon. 
                    "x" (array or None) : meilleure solution ou None si success = False.
                    "f" (array or None) : meilleure objectif ou None si success = False. 
                    "constrViolation" (List of float) : violation des contraintes. Liste vide si aucune contrainte.
        """
        self.__sign_factor = 1.0
        self.__runOptimization(npop, ngen, verbose=verbose)

        if returnDict : 
            result = {"method":"Continous Single Objective Genetic Algorithm",
                      "optimization" : "maximization",
                      "success":self.__success,
                      "x" : self.__optiX,
                      "f" : self.__optiObj*self.__sign_factor,
                      "constrViolation" : self.__constrViolation
                      }
            return result
        else : 
            return self.__optiX,self.__optiObj*self.__sign_factor
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         RENVOIS
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###

    def getLastPopulation(self) : return self.__lastPop

    def getStatOptimisation(self): return self.__statOptimisation




class continousBiObjectiveGA():
    

    def __init__(self,func1,func2,xmin,xmax,constraints=[],preprocess_function=None,func1_criterion="max",func2_criterion="max") :
        """
        Instance de continousSingleObjectiveGA : algorithme genetique d'optimisation de fonction mono-objectif à variables reelles
        Recherche d'un optimum global de la fonction f sur les bornes [xmin;xmax]. 
        Parameters : 
            - func1 (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est l'argument de la fonction de forme scalaire ou array et renvoie un scalaire ou array.

            - func2 (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est l'argument de la fonction de forme scalaire ou array et renvoie un scalaire ou array.

            - xmin (array like) : 
                Borne inferieure des variables d'optimisation. 

            - xmax (array like) : 
                Borne supérieure des variables d'optimisation. Doit être de même dimension que xmin. 

            - constraints (List of dict) option : 
                Definition des contraintes dans une liste de dictionnaires. Chaque dictionnaire contient les champs suivants 
                    type : str
                        Contraintes de type egalite 'eq' ou inegalite 'ineq' ; 
                    fun : callable
                        La fonction contrainte de la meme forme que func ; 

            - preprocess_function (callable or None) option : 
                Definition d'une fonction sans renvoie de valeur à executer avant chaque evaluation de la fonction objectif func ou des contraintes.

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


        """
        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)

        self.__ndof = len(self.__xmin)

        self.__preProcess = preprocess_function
        self.__constraints = constraints

        self.__nPreSelected = 2 #Nombre d'individus preselectionne pour tournois
        self.__stepMut = 0.15
        self.__rateMut = 0.25
        self.__crossFactor = 1.25
        self.__constraintAbsTol = 1e-3
        self.__penalityFactor = 1e3
        self.__penalityGrowth = 1.0
        self.__sharingDist = None
        self.__constraintMethod = "penality" #"feasibility"


        self.__function1 = func1
        self.__function2 = func2

        self.__selection_function = self.__selection_tournament
        self.__constraintMethod = "penality"

        self.__initial_population = None 
        self.__success = False
        self.__lastPop = None
        self.__nfront = 0

        self.__xfront = None
        self.__f1front = None
        self.__f2front = None
        self.__frontsize = 0

        self.__sign_factor = [1.0,1.0]
        if func1_criterion == 'min' : 
            self.__sign_factor[0] = -1.0
        if func2_criterion == 'min' : 
            self.__sign_factor[1] = -1.0

    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         PARAMETRISATION
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###

    def setPreSelectNumber(self,nbr_preselected=2):
        """
        Change le nombre d'individus pre-selectionnes dans un mode de selection par tournois. 
        Parameter : 
            - nbr_preselected (int) option : nombre d'individus participants à chaque tournois. Superieur a 1.
        """
        self.__nPreSelected = nbr_preselected

    def setMutationParameters(self,mutation_step=0.15,mutation_rate=0.25) : 
        """
        Change les parametres de mutation. 
        Parameters : 
            - mutation_step (float) option : limite de deplacement par mutation. Doit etre compris entre 0 et 1.

            - mutation_rate (float) option : probabilite de mutation. Doit etre compris entre 0 et 1.
        """

        self.__stepMut = mutation_rate
        self.__rateMut = mutation_rate

    def setCrossFactor(self,crossover_factor=1.25):
        """
        Change la limite de barycentre dans l'operateur de reproduction. Si egale a 1, les enfants seront strictement entre les parents. 
        Si superieur a 1, les enfants peuvent etre a l'exterieur du segment parent. 
        Parameter : 
            - crossover_factor (float) option : facteur de melange des individus parents ; 
        """
        self.__crossFactor = crossover_factor

    def setSharingDist(self,sharingDist=None) :
        """
        Change le rayon de diversite des solutions.
        Parameter : 
            - sharingDict (float or None) option : Rayon de diversite de solution. Si None, le parametre est initialise a 1/(taille population).
        """
        self.__sharingDist = sharingDist

    def setConstraintMethod(self,method="penality"):
        """
        Change la methode de prise en compte des contraintes. 
        Parameter : 
            method (str) option : Definition de la methode de gestion des contraintes. 
                Si method = 'penality' utilisation d'une penalisation quadratique. 
                Si method = 'feasibility' les solutions non satisfaisante sont rejetees.
        """
        if method == "feasibility" or method == "penality" :
            self.__constraintMethod = method

    def setPenalityParams(self,constraintAbsTol=1e-3,penalityFactor=1e3,penalityGrowth=1.00):
        """
        Change le parametrage de la penalisation de contrainte. 
        Parameters : 
            - contraintAbsTol (float) option : tolerance des contraintes d'egalite. Si la contrainte i ||ci(xk)|| <= contraintAbsTol, la solution est viable. 

            - penalityFactor (float) option : facteur de penalisation. La nouvelle fonction objectif est evaluee par la forme suivante :
                                                    penal_objective_func = objective_func + sum(ci**2 if ci not feasible)*penalityFactor
                                                    objectif_penalise = objectif + somme( ci**2 si ci non faisable)*facteur_penalite
                                            
            - penalityGrowth (float) option : pour chaque iteration de l'algorithme le facteur de penalite peut croitre d'un facteur penalityGrowth
                                                    penalityFactor_(k+1) = penalityFactor_(k)*penalityGrowth

        """
        self.__constraintAbsTol = constraintAbsTol
        self.__penalityFactor = penalityFactor
        self.__penalityGrowth = penalityGrowth
        self.__constraintMethod = "penality"

    def redefine_constraints(self,constraints=[]):
        """
        Permet de redefinir les contraintes du probleme. 
        Parameter : 
            - constraints (List of dict) option : 
                    Definition des contraintes dans une liste de dictionnaires. Chaque dictionnaire contient les champs suivants 
                        type : str
                            Contraintes de type egalite 'eq' ou inegalite 'ineq' ; 
                        fun : callable
                            La fonction contrainte de la meme forme que func ; 
        """
        self.__constraints = constraints
    
    def redefine_preprocess_func(self,preprocess_function=None):
        """
        Permet de redefinir la fonction de preprocessing. 
        Parameter
            - preprocess_function (callable or None) option : 
                Definition d'une fonction sans renvoie de valeur à executer avant chaque evaluation de la fonction objectif func ou des contraintes.
        """
        self.__preProcess = preprocess_function

    def define_initial_population(self,xstart):
        """
        Definition d'une population initiale. 
        Parameter : 
            - xstart (array(npop,ndof)) : 
                xstart est la solution initiale. Ses dimensions doivent etre de (npop,ndof). 
                npop la taille de la population et ndof le nombre de variable (degrees of freedom).
        """

        x = np.array(xstart)
        npop,ndof = x.shape
        if ndof != self.__ndof : 
            raise ValueError("The size of initial population is not corresponding to bounds size")

        xpop = np.maximum(np.minimum((x-self.__xmin)/(self.__xmax-self.__xmin),1.0),0.0)
        if npop%2 != 0 : 
            xpop = np.insert(xpop,0,0.5,axis=0)

        self.__initial_population = xpop

    def setSelectionMethod(self,method="tournament"):
        """
        Change la methode de selection. 
        Parameter : 
            - method (str) option : Definition de la methode de selection. 
                Si method = 'tournament' : selection par tournois. 
        """
        if method == "tournament" :
            self.__selection_function = self.__selection_tournament

    def redefine_objective(self,func1,func2,func1_criterion="max",func2_criterion="max"):
        """
        Permet de redefinir les fonctions objectifs. 
        Parameters : 
            - func1 (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est l'argument de la fonction de forme scalaire ou array et renvoie un scalaire ou array.

            - func2 (callable) : 
                Fonction objectif a optimiser de la forme f(x) ou x est l'argument de la fonction de forme scalaire ou array et renvoie un scalaire ou array.


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
        """
        self.__function1 = func1
        self.__function2 = func2
        if func1_criterion == 'min' : 
            self.__sign_factor[0] = -1.0
        if func2_criterion == 'min' : 
            self.__sign_factor[1] = -1.0




    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    ###                         OPERATEURS
    ### ---------------------------------------------------------------------------------------------------------------------------------------- ###
    def __barycenterCrossover(self,selection,population,npop):
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

        return population

    def __sharingFunction(self,fitness,population):
        """
        Operateur de diversite de solution
        """
        dShare = self.__sharingDist

        distance = np.array([np.sqrt(np.sum((population - xj)**2,axis=1)) for xj in population])
        sharing = (1-distance/dShare)*(distance<dShare)
        sharingFactor = np.sum(sharing,axis=1)
        fitness = fitness/sharingFactor


        fmin = fitness.min()
        fmax = fitness.max()
        fitness = (fitness-fmin)/(fmax-fmin)
        return fitness

    def __mutation_delta(self,population,npop) :
        """
        Operateur de mutation
        """
        probaMutation = rd.sample((npop,self.__ndof))
        deltaX = self.__stepMut*(rd.sample((npop,self.__ndof))-0.5)
        population = population + deltaX*(probaMutation<=self.__rateMut)
        return population
    
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
        max_rank = max(rank_R)
        crowning_dist = np.zeros_like(x1,dtype=float)
        x1min,x1max = x1.min(),x1.max()
        x2min,x2max = x2.min(),x2.max()

        for rank_k in range(1,max_rank+1):
            index_sol = rank_R == rank_k
            x1_k = x1[index_sol]
            x2_k = x2[index_sol]
            distance_k = np.zeros_like(x1_k,dtype=float)

            sortindex = np.argsort(x1_k)
            x1_k = x1_k[sortindex]
            distance_k[sortindex[1:-1:]] = (x1_k[2:] - x1_k[:-2])/(x1max-x1min)
            distance_k[sortindex[0]] = 1.0
            distance_k[sortindex[-1]] = 1.0

            sortindex = np.argsort(x2_k)
            x2_k = x2_k[sortindex]
            distance_k[sortindex[1:-1:]] += (x2_k[2:] - x2_k[:-2])/(x2max-x2min)
            distance_k[sortindex[0]] += 1.0
            distance_k[sortindex[-1]] += 1.0

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
                constrViol = gi
                feasibility &= constrViol<=self.__constraintAbsTol
                constrViolation.append(abs(constrViol))
                penality += constrViol**2

        return objective1,objective2,feasibility,penality,constrViolation

    def __fitness_and_feasibility(self,Xarray,objective1,objective2,npop) :
        """
        Evaluation de la fonction objectif et des contraintes
        """

        feasibility = np.ones(npop,dtype=bool)
        penality = np.zeros(npop,dtype=float)
        for i,xi, in enumerate(Xarray) :
            (obj1_i,
            obj2_i,
            feas_i,
            penal_i,_) = self.__evaluate_function_and_constraints(xi)

            objective1[i] = obj1_i
            objective2[i] = obj2_i
            feasibility[i] = feas_i
            penality[i] = penal_i

        if self.__constraintMethod == "penality" :
            return objective1-penality,objective2-penality,feasibility
        
        if self.__constraintMethod == "feasibility" :
            o1min,o2min = objective1.min(),objective2.min()
            objective1[np.logical_not(feasibility)] = o1min - 1
            objective2[np.logical_not(feasibility)] = o2min - 1

            return objective1,objective2,feasibility


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
                                  children_pop,
                                  children_obj1,
                                  children_obj2,
                                  feasibility) :


        feasible_pop = np.array([pi for pi in parents_pop]+[ci for ci in children_pop[feasibility]])
        feasible_obj1 = np.array([pi for pi in parents_obj1]+[ci for ci in children_obj1[feasibility]])
        feasible_obj2 = np.array([pi for pi in parents_obj2]+[ci for ci in children_obj2[feasibility]])

        npop = len(children_pop)
        nfeasible = len(feasible_pop)
        # omin1,omax1 = feasible_obj1.min(),feasible_obj1.max()
        # omin2,omax2 = feasible_obj2.min(),feasible_obj2.max()




        if (nfeasible >= npop) :

            rank_pop = self.__fast_non_dominated_sort(feasible_obj1,feasible_obj2,npop)
            fitness = self.__crowning_distance(feasible_obj1,feasible_obj2,rank_pop)



            values = list(zip(rank_pop,-fitness))
            dtype = [('rank',int),('crowning',float)]
            evaluation_array = np.array(values,dtype=dtype)
            sorted_index = np.argsort(evaluation_array,order=["rank","crowning"])

            population = feasible_pop[sorted_index][:npop]
            objective1 = feasible_obj1[sorted_index][:npop]
            objective2 = feasible_obj2[sorted_index][:npop]
            rank_pop = rank_pop[sorted_index]
            fitness = self.__crowning_distance(objective1,objective2,rank_pop[:npop])

            parents_pop = feasible_pop[sorted_index][rank_pop==1]
            parents_obj1 = feasible_obj1[sorted_index][rank_pop==1]
            parents_obj2 = feasible_obj2[sorted_index][rank_pop==1]

            if len(parents_pop) > self.__nfront :
                parents_pop = parents_pop[:self.__nfront]
                parents_obj1 = parents_obj1[:self.__nfront]
                parents_obj2 = parents_obj2[:self.__nfront]
            rank_pop = rank_pop[:npop]


            return  (population,
                    objective1,
                    objective2,
                    rank_pop,
                    fitness,
                    parents_pop,
                    parents_obj1,
                    parents_obj2)


        else :
            nextend = npop-nfeasible
            notfeasible = np.logical_not(feasibility)
            notfeasible_pop = children_pop[notfeasible]
            notfeasible_obj1 = children_obj1[notfeasible]
            notfeasible_obj2 = children_obj2[notfeasible]

            rank_pop = self.__fast_non_dominated_sort(feasible_obj1,feasible_obj2,nfeasible)
            max_rank = max(rank_pop)
            parents_pop = feasible_pop[rank_pop==1]
            parents_obj1 = feasible_obj1[rank_pop==1]
            parents_obj2 = feasible_obj2[rank_pop==1]

            population = np.array([xi for xi in feasible_pop]+[xi for xi in notfeasible_pop[:nextend]])
            objective1 = np.array([xi for xi in feasible_obj1]+[xi for xi in notfeasible_obj1[:nextend]])
            objective2 = np.array([xi for xi in feasible_obj2]+[xi for xi in notfeasible_obj2[:nextend]])
            rank_pop = np.array([xi for xi in rank_pop]+[max_rank+1 for i in range(nextend)])
            fitness = self.__crowning_distance(objective1,objective2,rank_pop)



            return  (population,
                    objective1,
                    objective2,
                    rank_pop,
                    fitness,
                    parents_pop,
                    parents_obj1,
                    parents_obj2)


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
            population = rd.sample((npop,ndof)) #population initiale
        else : 
            population = self.__initial_population
            npop = len(population)

        if self.__sharingDist is None :
            self.__sharingDist = 1/npop

        if nfront is None :
            nfront = 3*npop
        self.__nfront = nfront

        xmin = self.__xmin
        xmax = self.__xmax

        objective1 = np.zeros(npop)
        objective2 = np.zeros(npop)

        startTime = time.time()

        population = rd.sample((npop,ndof)) #population initiale
        Xarray = population*(xmax-xmin) + xmin

        (objective1,
        objective2,
        feasibility) = self.__fitness_and_feasibility(Xarray,
                                                        objective1,
                                                        objective2,
                                                        npop)

        rank_pop = self.__fast_non_dominated_sort(objective1,objective2,None)
        fitness = self.__crowning_distance(objective1,objective2,rank_pop)
        parents_pop = population[feasibility]
        parents_obj1 = objective1[feasibility]
        parents_obj2 = objective2[feasibility]

        for generation in range(ngen):

            #Algorithme genetique

            fitness = self.__sharingFunction(fitness,population)

            selection = self.__selection_function(population,npop,rank_pop,fitness)

            children_pop = self.__barycenterCrossover(selection,population,npop)

            children_pop = self.__mutation_delta(children_pop,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            Xarray = children_pop*(xmax-xmin) + xmin
            (objective1,
            objective2,
            feasibility) = self.__fitness_and_feasibility(Xarray,
                                                            objective1,
                                                            objective2,
                                                            npop)

            #Elistisme
            (population,
            objective1,
            objective2,
            rank_pop,
            fitness,
            parents_pop,
            parents_obj1,
            parents_obj2) = self.__selection_elitisme(parents_pop,
                                                parents_obj1,
                                                parents_obj2,
                                                children_pop,
                                                objective1,
                                                objective2,
                                                feasibility)


            self.__lastPop = population*(xmax-xmin) + xmin
            # self.__archive_solution(parents_pop,parents_obj1,parents_obj2)
            # self.__archive_details(generation)

            if verbose :
                print('Iteration ',generation+1)





        Xarray = parents_pop*(xmax-xmin) + xmin
        endTime = time.time()
        duration = endTime-startTime
        self.__frontsize = Xarray.shape[0]

        if self.__frontsize > 0 : 
            self.__success = True
            self.__xfront = Xarray
            self.__f1front = parents_obj1*self.__sign_factor[0]
            self.__f2front = parents_obj2*self.__sign_factor[1]


        else : 
            self.__success = False
            self.__xfront = None
            self.__f1front = None
            self.__f2front = None
            
        ##MESSAGES

        print('\n'*2+'#'*60+'\n')
        print('AG iterations completed')
        print('Number of generations : ',ngen)
        print('Population size : ',npop)
        print('Elapsed time : %.3f s'%duration)

        print('#'*60+'\n')







    def optimize(self,npop,ngen,nfront=None,verbose=True,returnDict=False):
        """
        Algorithme d'optimisation bi-objectifs sous contrainte. 
        Parameters : 
            - npop (int) : 
                Taille de la population. Si npop est impair, l'algorithm l'augmente de 1. 
                Usuellement pour un probleme sans contrainte une population efficace est situee entre 5 et 20 fois le nombre de variable.
                Si les contraintes sont fortes, il sera utile d'augmenter la population.
                Ce parametre n'est pas pris en compte si une population initiale a ete definie.

            - ngen (int) : 
                Nombre de generation. Usuellement une bonne pratique est de prendre 2 à 10 fois la taille de la population. 

            - nfront (int or None) option : 
                Nombre de point maximum dans le front de Pareto. Si None, alors nfront = 3*npop

            - verbose (bool) option : 
                Affiche l'etat de la recherche pour chaque iteration. Peut ralentir l'execution.

            - returnDict (bool) option : 
                Si l'option est True alors l'algorithme retourne un dictionnaire. 
            
        Returns : 
            Si (returnDict = False) : 
                tuple : xsolutions, objective1_solutions, objective2_solutions (array(ndof,nfront), array(nfront), array(nfront))
                    - xsolutions sont les solutions composants le front de Pareto.
                    - objective1_solutions sont les points de la fonction 1 dans le front de Pareto. 
                    - objective2_solutions sont les points de la fonction 2 dans le front de Pareto.
            
            Si (returnDict = True) : 
                dict :
                    "method" (str) : algorithm utilise.
                    "optimization" (str) : minimisation ou maximisation.
                    "success" (bool) : True si l'algorithm a converge, False sinon. 
                    "x" (array or None) : solutions du front de Pareto.
                    "f1" (array or None) : front de Pareto f1.
                    "f2" (array or None) : front de Pareto f2.
        """
        self.__runOptimization(npop, ngen, nfront=nfront, verbose=verbose)

        if returnDict : 
            result = {"method":"Continous Single Objective Genetic Algorithm",
                      "optimization" : "minimization",
                      "success":self.__success,
                      "x" : self.__xfront,
                      "f1" : self.__f1front,
                      "f2" : self.__f2front
                      }
            return result
        else : 
            return self.__xfront, self.__f1front, self.__f2front

if __name__ == '__main__' : 
    # x = np.linspace(-1,1)
    # y = x**2

    # ga_instance = continousSingleObjectiveGA(lambda x : x[0]**2 + x[1]**2,
    #                                         [-1,-1],
    #                                         [1,1])


    # resGA = ga_instance.minimize(20,100,verbose=True,returnDict=True)

    # for ri in resGA : 
    #     print(ri," : ",resGA[ri])

    f1 = lambda x : x**2+x-3
    f2 = lambda x : x**2-x+2
    nsga_instance = continousBiObjectiveGA(f1,f2,[-2],[2])


    # nsga_instance.setElitisme(True)

    

    