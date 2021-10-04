import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize,dual_annealing

plt.rc('font',family='Serif')
## Fonction objectif
f0 = lambda x : (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

xmin = [-75,-75]
xmax = [75,75]

## contraintes 
fc0= lambda x : -(0.001*x**3-x)
c0 = lambda x : x[1] - fc0(x[0])
epsilon = 80
c1 = lambda x : -x[0] + 2*x[1] + 0.01*x[1]**2 - 1 
c2 = lambda x : -c1(x) + 2*epsilon

cons = []
#OR
cons = [{'type': 'ineq', 'fun': c1},
       {'type': 'ineq', 'fun': c2}]

## Optimisation

import sys
sys.path.append("..")
from _simulated_annealing import minimize_simulatedAnnealing
from _genetic_algorithm import realSingleObjectiveGA
from _differential_evolution import differential_evolution

nloop = 10
ga_convergence = []
sa_convergence = []
de_convergence = []


maxIter = 2000
npop = 36
ngen = (maxIter+1)//npop

ga_instance = realSingleObjectiveGA(f0,xmin,xmax,constraints=cons,tol=-1.0,constraintMethod="penality")

for k in range(nloop):
    print("LOOP : ",k)
    print("")
    #genetic algorithm 
    ga_instance.minimize(npop,ngen,verbose=False)
    fitness_ga = ga_instance.getStatOptimisation()

    ga_convergence.append(fitness_ga)

    #simulated annealing
    mindict_sa = minimize_simulatedAnnealing(f0,xmin,xmax,maxIter=maxIter,constraints=cons,returnDict=True,storeIterValues=True)
    sa_convergence.append(mindict_sa["fHistory"])

    #differential evolution 
    mindict_de = differential_evolution(f0,xmin,xmax,maxIter=ngen,popsize=npop,constraints=cons,returnDict=True,storeIterValues=True,tol=-1)
    de_convergence.append(mindict_de["fHistory"])


# Optimisation locale
bounds = [(xi,xj) for xi,xj in zip(xmin,xmax)]
startX = np.mean(bounds,axis=1)
res = minimize(f0,mindict_sa["x"],bounds=bounds)
fscipy = res.fun


ga_convergence = np.log10((np.array(ga_convergence).T - fscipy)/abs(fscipy))
sa_convergence = np.log10((np.array(sa_convergence).T - fscipy)/abs(fscipy))
de_convergence = np.log10((np.array(de_convergence).T - fscipy)/abs(fscipy))


plt.figure(figsize=(8,4))

line_sa_mean = plt.plot( sa_convergence.mean(axis=1),
                        lw = 1.5,
                        ls ='-',
                        color='r')[0]

line_ga_mean = plt.plot(np.array(list(range(len(ga_convergence))))*npop,
                        ga_convergence.mean(axis=1),
                        lw = 1.5,
                        color='b')[0]

line_de_mean = plt.plot(de_convergence.mean(axis=1),
                        lw = 1.5,
                        color='g')[0]

plt.grid(True)
# plt.ylim(-8,0)
plt.xlabel("Evaluation de la fonction")
plt.ylabel("log10 erreur")
# plt.yscale('log')
plt.title("Convergence des algorithmes")
plt.legend([
            line_ga_mean,
            line_sa_mean,
            line_de_mean],
            [
            "Genetic algorithme (avg)",
            "Simulated annealing (avg)",
            "Differential evolution (avg)"])
plt.tight_layout()
plt.savefig("convergence.svg",dpi=300)

plt.show()