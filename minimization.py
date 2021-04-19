# coding: utf-8

# --- Optimisation 1D
from _minimize_scalar import goldenSearch, scalarGradient

# --- Optimisation analytique
from _minimize_BFGS import BFGS
from _minimize_gradient import conjugateGradient
from _minimize_NelderMead import NelderMead
from _minimize_Powell import Powell

# --- Optimisation globale stochastique
from _simulated_annealing import simulatedAnnealing
from _genetic_algorithm import optimizeMonoAG

all = ["optimizeMonoAG",
       "localMinimize"]


"""
Minimization :


"""

def minimize_scalar() :
    """
    Minimization using scalar search routines.

    """

    return



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