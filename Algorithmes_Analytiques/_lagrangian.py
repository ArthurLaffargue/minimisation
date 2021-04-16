
import numpy as np
from _numerical_deriv import multipleFuncAndGradEvaluator
from scipy.sparse import bmat
from scipy.sparse.linalg import cg
from numpy.linalg import norm,det
from scipy.optimize import fsolve

def lagrangianWrapper(func,
                      dim,
                      gf=None,
                      constraints=[],
                      precallfunc=None,
                      gradientMethod="finite-difference",
                      dh=1e-12) :




    ineqCons = []
    eqCons = []
    funcList = [func]
    gradientList = [gf]
    for i,ci in enumerate(constraints) :
        fc = ci["fun"]
        type = ci["type"]

        if "jac" in ci :
            gc = ci["jac"]
        else :
            gc = None

        funcList += [fc]
        gradientList += [gc]

        if type == "ineq" :
            ineqCons.append(i+1)
        if type == "eq" :
            eqCons.append(i+1)

    multiFuncGradEvals = multipleFuncAndGradEvaluator(
                    funcList,
                    gradientList,
                    dim,
                    dh=dh,
                    precallfunc=precallfunc)

    return ineqCons,eqCons,multiFuncGradEvals


def initLagrangianMulti(gf0,Jce,Jci):
    dim,ni = Jci.shape
    _,ne = Jce.shape

    A = np.zeros((dim,ni+ne))
    A[:,:ni] = Jci
    A[:,ni:] = Jce
    K = A.T.dot(A)
    mu,_ = cg(K,A.T.dot(-gf0))
    mui = mu[:ni]
    mue = mu[ni:]
    return mui,mue


def lagrangianQP(fk,
                gfk,
                Bk,
                ce,
                Jce,
                cineq,
                Jcineq,
                dk,
                muek,
                muineqk,
                tol=1e-6,
                maxCGiter=250):

    activeIneq = cineq<0.0
    ci = cineq[activeIneq]
    Jci = Jcineq[:,activeIneq]
    muik = muineqk[activeIneq]

    dim = len(dk)
    ne = len(ce)
    ni = len(ci)
    dimz = dim + ne + ni
    if dimz == dim :
        return dk,muineqk,muek


    z0 = np.zeros(dimz)
    z0[:dim] = dk
    z0[dim:dim+ne:] = muek
    z0[dim+ne:] = muik

    q = np.zeros(dimz)
    q[:dim] = -gfk
    q[dim:dim+ne:] = -ce
    q[dim+ne:] = -ci

    block = [[Bk,Jce,Jci],
             [Jce.T,None,None],
             [Jci.T,None,None]]

    Amat = bmat(block)
    detA = det(Amat.toarray())
    if np.abs(detA) < tol :
        return dk,muineqk,muek

    z0,_ = cg(Amat,q)
    dk = z0[:dim]
    muek = z0[dim:dim+ne:]
    muineqk[activeIneq] = z0[dim+ne:]

    return dk,muineqk,muek



