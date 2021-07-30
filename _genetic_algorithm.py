import numpy as np
import numpy.random as rd
import time

class optimizeMonoAG :

    def __init__(self,f,xmin,xmax,constraints=[],preprocess_function=None) :

        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)

        self.__ndof = len(self.__xmin)

        self.__preProcess = preprocess_function
        self.__function = f
        self.__constraints = constraints

        self.__nPreSelected = 2 #Nombre d'individus preselectionne pour tournois
        self.__stepMut = 0.15
        self.__rateMut = 0.25
        self.__crossFactor = 1.25
        self.__constraintAbsTol = 1e-6
        self.__penalityFactor = 1000
        self.__penalityGrowth = 1.0
        self.__sharingDist = None
        self.__constraintMethod = "penality" #"feasibility"
        self.__elitisme = True


        self.__selection_function = self.__selection_tournament

    def setPreSelectNumber(self,nbr_preselected):
        self.__nPreSelected = nbr_preselected

    def setStepMut(self,step_mutation):
        self.__stepMut = step_mutation

    def setCrossFactor(self,mutation_rate):
        self.__rateMut = mutation_rate

    def setCrossFactor(self,crossover_factor):
        self.__crossFactor = crossover_factor

    def setSharingDist(self,sharingDist) :
        self.__sharingDist = sharingDist

    def setSelectionMethod(self,method="tournament"):
        if method == "SRWRS" :
            self.__selection_function = self.__selection_SRWRS
        if method == "tournament" :
            self.__selection_function = self.__selection_tournament

    def setConstraintMethod(self,method="penality"):
        if method == "feasibility" or method == "penality" :
            self.__constraintMethod = method

    def setPenalityParams(self,constraintAbsTol=1e-6,penalityFactor=1,penalityGrowth=1.01):
        self.__constraintAbsTol = constraintAbsTol
        self.__penalityFactor = penalityFactor
        self.__penalityGrowth = penalityGrowth
        self.__constraintMethod = "penality"

    def redefine_objective(self,f):
        self.__function = f

    def redefine_constraints(self,constraints=[]):
        self.__constraints = constraints

    def setElitisme(self,elitisme=True):
        self.__elitisme = elitisme

    def optimize(self,npop,ngen,verbose=True):


        ## Initialisation
        if not(self.__elitisme) :
            self.__constraintMethod = "penality"

        if npop%2 != 0 :
            npop += 1

        if self.__sharingDist is None :
            self.__sharingDist = 1/npop

        xmin = self.__xmin
        xmax = self.__xmax
        ndof = self.__ndof


        self.__optiObj = None
        self.__optiX = None

        self.__statOptimisation  = np.zeros(ngen)
        objective = np.zeros(npop)


        startTime = time.time()

        population = rd.sample((npop,ndof)) #population initiale
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

        print('\n'*2+'#'*60+'\n')
        print('AG iterations completed')
        print('Number of generations : ',ngen)
        print('Population size : ',npop)
        print('Elapsed time : %.3f s'%duration)

        print('#'*60+'\n')

        self.__lastPop = Xarray


        return self.__optiX,self.__optiObj

# --------------------------------------------------------------------------- #
    def __fitness_and_feasibility(self,Xarray,objective,npop) :

        feasibility = np.ones(npop,dtype=bool)
        penality = np.zeros(npop,dtype=float)
        for i,xi, in enumerate(Xarray) :

            if self.__preProcess is not None :
                self.__preProcess(xi)

            objective[i] = self.__function(xi)

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


        if self.__constraintMethod == "penality" :
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
        ndof = self.__ndof
        selection = np.zeros((npop//2,ndof))
        for i in range(npop//2):
            indices = rd.choice(npop-1,self.__nPreSelected)
            selection[i] = population[indices[np.argmax(fitness[indices])]]
        return selection

    def __selection_SRWRS(self,population,npop,fitness) :
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

        probaMutation = rd.sample((npop,self.__ndof))
        deltaX = self.__stepMut*(rd.sample((npop,self.__ndof))-0.5)
        population = population + deltaX*(probaMutation<=self.__rateMut)
        return population


    def __selection_elitisme(self,parents_pop,parents_obj,children_pop,children_obj,children_penal,feasibility) :

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

        self.__statOptimisation[generation] = self.__optiObj


    def getLastPopulation(self) : return self.__lastPop

    def getStatOptimisation(self): return self.__statOptimisation





class optiBiAG :

    def __init__(self,f1,f2,xmin,xmax,constraints=[],preprocess_function=None) :

        self.__xmin = np.minimum(xmin,xmax)
        self.__xmax = np.maximum(xmin,xmax)

        self.__ndof = len(self.__xmin)

        self.__preProcess = preprocess_function
        self.__function1 = f1
        self.__function2 = f2
        self.__constraints = constraints

        self.__nPreSelected = 2 #Nombre d'individus preselectionne pour tournois
        self.__stepMut = 0.10
        self.__rateMut = 0.25
        self.__crossFactor = 1.35
        self.__sharingDist = None

        self.__constraintAbsTol = 1e-6
        self.__penalityFactor = 1e6
        self.__constraintMethod = "penality"

    def setPreSelectNumber(self,nbr_preselected):
        self.__nPreSelected = nbr_preselected

    def setStepMut(self,step_mutation):
        self.__stepMut = step_mutation

    def setCrossFactor(self,mutation_rate):
        self.__rateMut = mutation_rate

    def setCrossFactor(self,crossover_factor):
        self.__crossFactor = crossover_factor

    def setSharingDist(self,sharingDist) :
        self.__sharingDist = sharingDist

    def setConstraintMethod(self,method="feasibility"):
        if method == "feasibility" or method == "penality" :
            self.__constraintMethod = method

    def setPenalityParams(self,constraintAbsTol=1e-6,penalityFactor=1e6,penalityGrowth=1.0):
        self.__constraintAbsTol = constraintAbsTol
        self.__penalityFactor = penalityFactor
        self.__constraintMethod = "penality"

    def optimize(self,npop,ngen,nfront=None,verbose=True):


        ## Initialisation
        if npop%2 != 0 :
            npop += 1

        if self.__sharingDist is None :
            self.__sharingDist = 1/npop

        if nfront is None :
            nfront = 3*npop
        self.__nfront = nfront

        xmin = self.__xmin
        xmax = self.__xmax
        ndof = self.__ndof

        objective1 = np.zeros(npop)
        objective2 = np.zeros(npop)

        startTime = time.time()

        population = rd.sample((npop,ndof)) #population initiale
        Xarray = population*(xmax-xmin) + xmin

        (objective1,
        objective2,
        feasibility) = self.__evaluate_objectives_and_constraints(Xarray,
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

            selection = self.__selection_tournament(population,npop,rank_pop,fitness)

            children_pop = self.__barycenterCrossover(selection,population,npop)

            children_pop = self.__mutation_delta(children_pop,npop)

            children_pop = np.minimum(1.0,np.maximum(0.0,children_pop))

            Xarray = children_pop*(xmax-xmin) + xmin
            (objective1,
            objective2,
            feasibility) = self.__evaluate_objectives_and_constraints(Xarray,
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



            # self.__archive_solution(parents_pop,parents_obj1,parents_obj2)
            # self.__archive_details(generation)

            if verbose :
                print('Iteration ',generation+1)





        Xarray = parents_pop*(xmax-xmin) + xmin
        endTime = time.time()
        duration = endTime-startTime

        ##MESSAGES

        print('\n'*2+'#'*60+'\n')
        print('AG iterations completed')
        print('Number of generations : ',ngen)
        print('Population size : ',npop)
        print('Elapsed time : %.3f s'%duration)

        print('#'*60+'\n')

        self.__lastPop = Xarray


        return parents_pop,parents_obj1,parents_obj2

# --------------------------------------------------------------------------- #

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




    def __evaluate_objectives_and_constraints(self,Xarray,objective1,objective2,npop) :

        feasibility = np.ones(npop,dtype=bool)
        penality = np.zeros(npop,dtype=float)
        for i,xi, in enumerate(Xarray) :

            if self.__preProcess is not None :
                self.__preProcess(xi)

            objective1[i] = self.__function1(xi)
            objective2[i] = self.__function2(xi)

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


        if self.__constraintMethod == "penality" :
            penality = self.__penalityFactor*penality
            objective1 = objective1 - penality
            objective2 = objective2 - penality
        o1min,o2min = objective1.min(),objective2.min()
        if self.__constraintMethod == "feasibility" :
            objective1[np.logical_not(feasibility)] = o1min - 1
            objective2[np.logical_not(feasibility)] = o2min - 1

        return objective1,objective2,feasibility



    def __barycenterCrossover(self,selection,population,npop):

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
        dShare = self.__sharingDist

        distance = np.array([np.sqrt(np.sum((population - xj)**2,axis=1)) for xj in population])
        sharing = (1-distance/dShare)*(distance<dShare)
        sharingFactor = np.sum(sharing,axis=1)
        fitness = fitness/sharingFactor


        fmin = fitness.min()
        fmax = fitness.max()
        fitness = (fitness-fmin)/(fmax-fmin)
        return fitness

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


    def __mutation_delta(self,population,npop) :

        probaMutation = rd.sample((npop,self.__ndof))
        deltaX = self.__stepMut*(rd.sample((npop,self.__ndof))-0.5)
        population = population + deltaX*(probaMutation<=self.__rateMut)
        return population


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




        if nfeasible >= npop :

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



    def getLastPopulation(self) : return self.__lastPop

    def getStatOptimisation(self): return self.__statOptimisation