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
        self.__penalityFactor = 100
        self.__penalityGrowth = 1.0
        self.__sharingDist = None
        self.__constraintMethod = "penality" #"feasibility"
        self.__elitisme = True

        self.__initial_population = None 

        self.__optiObj = None
        self.__optiX = None
        self.__success = False
        self.__lastPop = None

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

    def setPenalityParams(self,constraintAbsTol=1e-3,penalityFactor=100,penalityGrowth=1.00):
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

    def __fitness_and_feasibility(self,Xarray,objective,npop) :
        """
        Evaluation de la fonction objectif et des contraintes
        """

        feasibility = np.ones(npop,dtype=bool)
        penality = np.zeros(npop,dtype=float)
        for i,xi, in enumerate(Xarray) :

            if self.__preProcess is not None :
                self.__preProcess(xi)

            objective[i] = self.__function(xi)*self.__sign_factor

            for c in self.__constraints :
                type = c["type"]
                g = c['fun']
                gi = g(xi)

                if type == 'strictIneq'  :
                    feasibility[i] *= (gi>0)
                    penality[i] += (gi*(gi<=0.0))**2

                if type == 'ineq' :
                    feasibility[i] *= (gi>=0)
                    penality[i] += np.minimum(gi,0.0)**2

                if type == 'eq'  :
                    feasibility[i] *= (np.abs(gi)<=self.__constraintAbsTol)
                    penality[i] += gi**2


        if self.__constraintMethod == "constraint" :
            penality = self.__penalityFactor*penality
            penalObjective = objective - penality
        else :
            penalObjective = objective
        omin = penalObjective.min()
        omax = penalObjective.max()
        fitness = (penalObjective-omin)/(omax-omin)
        if self.__constraintMethod == "feasibility" :
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
            penality = np.array([0.0 for xi in feasible_obj]+[xi for xi in notfeasible_penal[sortedIndex]])
            penalObj = objective - penality
            omin = penalObj.min()
            omax = penalObj.max()
            fitness = (penalObj-omin)/(omax-omin)



            return population,penalObj,feasible_pop,feasible_obj,fitness

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
        self.__statOptimisation[generation] = self.__optiObj


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
                    "success" (bool) : True si l'algorithm a converge, False sinon. 
                    "x" (array or None) : meilleure solution ou None si success = False.
                    "f" (array or None) : meilleure objectif ou None si success = False. 
                    "constrViolation" (List of float) : violation des contraintes. List vide si aucune contrainte.
        """
        self.__sign_factor = -1.0
        self.__runOptimization(npop, ngen, verbose=verbose)

        return self.__optiX,self.__optiObj




if __name__ == '__main__' : 
    x = np.linspace(-1,1)
    y = x**2

    import matplotlib.pyplot as plt

    ga_instance = continousSingleObjectiveGA(lambda x : x[0]**2 + x[1]**2,
                                            [-1,-1],
                                            [1,1])


    xag,yag = ga_instance.minimize(20,100,verbose=True)


    print(xag)

    
