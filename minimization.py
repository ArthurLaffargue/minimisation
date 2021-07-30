# coding: utf-8
#__docformat__ = "restructuredtext en"

# --- Optimisation 1D
from _minimize_scalar import goldenSearch, scalarGradient

# --- Optimisation analytique
from _minimize_BFGS import BFGS
from _minimize_gradient import conjugateGradient
from _minimize_NelderMead import NelderMead
from _minimize_Powell import Powell

# --- Optimisation globale stochastique
from _simulated_annealing import simulatedAnnealing
from _genetic_algorithms import continousSingleObjectiveGA,continousBiObjective_NSGA

__all__ = ["minimize_scalar",
           "continousSingleObjectiveGA",
           "continousBiObjective_NSGA"]

method_minimizeScalar = ["goldenSearch",
                         "scalarGradient"]

def minimize_scalar(f,xmin,xmax,xinit=None,
                    method='goldenSearch',
                    tol=1e-6,
                    gtol=1e-6,
                    maxIter=500,
                    gf=None,
                    constraints=[],
                    penalityFactor=100,
                    returnDict=False,
                    storeIterValues=False,
                    deriveMethod="finite-difference",
                    dh=1e-9,
                    stop_tolRatio=1e-2,
                    stop_gtolRatio=1e-2,
                    precallfunc=None) :
    """
    minimize_scalar : 

    Algorithmes de minimisation de fonctions scalaires à une variable.

    Parameters : 

        - f (callable) : une fonction scalaire d'une variable scalaire réelle. 

        - xmin (float) : borne inférieure du problème de minimisation. 

        - xmax (float) : borne supérieure du problème de minimisation. 

        - xinit (float or None) option : point initiale uniquement pour les méthodes [scalarGradient], si None xinit = (xmin+xmax)/2. 

        - method (str) option : méthode de minimisation. 
            *"goldenSearch" : méthode dichotomique basée sur le nombre d'or sans évaluation du gradient. 
            *"scalarGradient" : méthode du gradient appliqué en dimension scalaire. Gradient approché ou exacte. 
        
        - tol (float) option : tolérance sur critère d'arrêt. L'agorithme s'arrête si (residu < tol)AND(gradient<gtol).

        - gtol (float) option : tolérance sur la valeur du gradient. 

        - maxIter (int) option : nombre limite d'itérations. 

        - gf (callable or None) : fonction de la dérivée. Fonction scalaire réelle. Si 'None' une approximation est utilisée. 

        - constraints (List of dict) option : 
                Definition des contraintes dans une liste de dictionnaires. Chaque dictionnaire contient les champs suivants 
                    type : str
                        Contraintes de type egalite 'eq' ou inegalite 'ineq' ; 
                    fun : callable
                        La fonction contrainte de la meme forme que func ; 
                    jac : callable 
                        Dérivée de la fonction contrainte de la même forme que gf ; 
        
        - penalityFactor (float) option : facteur de penalisation. La nouvelle fonction objectif est evaluee par la forme suivante :
                                                    penal_objective_func = objective_func + sum(ci**2 if ci not feasible)*penalityFactor
                                                    objectif_penalise = objectif + somme( ci**2 si ci non faisable)*facteur_penalite
        
        - returnDict (bool) option : 
                Si l'option est True alors l'algorithme retourne un dictionnaire. 

        - storeIterValues (bool) option :
                L'évolution des variables durant la résolution est stockée dans un numpy array retourné dans le dictionnaire si returnDict = True ; 
        
        - deriveMethod (string) option : méthode d'approximation de la derivée. 
                *"finite-difference" : df = ( f(x+dh)-f(x) )/dh
                *"complex" : df = imag(f(x+1j*dh)/dh))
        
        - dh (float) option : pas d'approximation de la dérivée. dh = max(abs(x)*dh,dh)

        - stop_tolRatio (float) option : tolerance minimale pour forcer l'arrêt sur le résidu si le gradient n'atteint pas la tolerance gtol. 
                La tolerance minimale est calculée par mintol = tol*stop_tolRatio.
        
        - stop_gtolRatio (float) option : tolerance minimale pour forcer l'arrêt sur le gradient si le résidu n'atteint pas la tolerance tol. 
                La tolerance minimale est calculée par mingtol = gtol*stop_gtolRatio.

        - precallfunc (callable or None) option : 
                Definition d'une fonction sans renvoie de valeur à executer avant chaque evaluation de la fonction objectif, des gradients ou des contraintes.

    Returns : 

        Si (returnDict = False) : 
            x (float) : solution de l'algorithme d'optimisation. 
        
        Si (returnDict = False) : 
            dict :
                "method" : algorithme utilisé.
                "success" : True si l'agorithm a convergé correctement.
                "x": solution.
                "fmin": valeur de la fonction objectif.
                "residual": résidu.
                "gradResidual": résidu du gradient.
                "gradNorm": norme du gradient.
                "constrViolation": list des violations des contraintes. Vide si pas de contrainte.
                "iterations": nombre d'itérations.
                "functionCalls": nombre d'appels de la fonction.
                "gradientCalls": nombre d'appels du gradient. 

                Si (storeIterValues = True) : 
                    "xHistory" : array des variables de recherche. 
                    "fHistory" : array des valeurs de la fonction objectif. 
                    "rHistory" : array des valeurs du résidu. 
    

    Example : 
    
        import numpy as np

        xmin,xmax = 2.7,7.5
        xinit = (xmax+xmin)/2

        f = lambda x : np.sin(x)*x + np.sin(10/3*x)
        df = lambda x : np.cos(x)*x + np.sin(x) + 10/3*np.cos(10/3*x)
        x = np.linspace(xmin,xmax,250)
        y = f(x)

        dictgold = minimize_scalar(f,xmin,xmax,returnDict=True,method="goldenSearch")
        dictgrad = minimize_scalar(f,xmin,xmax,xinit=xinit,returnDict=True,gf=df,method="scalarGradient")

        Outputs : 
            #Solution goldenSearch
            method  :  goldenSearch
            success  :  True
            x  :  5.094751312571761
            fmin  :  -5.6832744082498365
            constrViolation  :  []
            residual  :  8.696778973964855e-07
            iterations  :  29
            functionCalls  :  32
            gradientCalls  :  0

            #Solution scalarGradient
            method  :  gradient
            success  :  True
            x  :  5.094751436644642
            fmin  :  -5.683274408249959
            residual  :  2.6531499293843317e-06
            gradResidual  :  1.8482260166763353e-09
            gradNorm  :  1.8482260166763353e-09
            constrViolation  :  []
            iterations  :  2
            functionCalls  :  7
            gradientCalls  :  3
    """


    if method not in method_minimizeScalar : 
        raise ValueError("Method %s not implemented for minimize_scalar. Available methods are : %s"%\
                                (str(method),"-".join(method_minimizeScalar)))


    if xinit is None : 
        xinit = 0.5*(xmin+xmax)
    
    if method == "goldenSearch" : 
        res = goldenSearch(f,xmin,xmax,tol=tol,maxIter=maxIter,constraints=constraints,
                                                    penalityFactor=penalityFactor,
                                                    storeIterValues=storeIterValues,
                                                    returnDict=returnDict,
                                                    precallfunc=precallfunc)

    if method == "scalarGradient" : 
        res = scalarGradient(f,xinit,xmin,xmax,tol=tol,gtol=gtol,maxIter=maxIter,gf=gf,
                                                    constraints=constraints,
                                                    penalityFactor=penalityFactor,
                                                    storeIterValues=storeIterValues,
                                                    returnDict=returnDict,
                                                    deriveMethod=deriveMethod,
                                                    dh=dh,
                                                    stop_tolRatio=stop_tolRatio,
                                                    stop_gtolRatio=stop_gtolRatio,
                                                    precallfunc=precallfunc)

    return res



def localMinimize():
    """
    Minimization using local analytic optimization routines.

    """

    return



def globalMinimize():
    """
    Minimization using global stochastic optimization routines.

    """

    return