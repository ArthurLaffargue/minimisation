import numpy as np
from numpy.linalg import norm, solve,det
from _linesearch import linesearch
from _penalization import constraintsWrapper,penalizationFunction
from scipy.sparse.linalg import cg
from scipy.sparse import bmat
__all__ = ["SQP_BFGS"]


def SQP_BFGS(f,xinit,xmin,xmax,constraints,tol=1e-6,gtol=1e-6,ctol=None,
                            maxIter=500,
                            gf=None,
                            penalityFactor=100.0,
                            storeIterValues=False,
                            returnDict=False,
                            gradientMethod="finite-difference",
                            dh=1e-9,
                            stop_tolRatio=1e-2,
                            stop_gtolRatio=1e-2,
                            precallfunc=None) :

    if ctol is None :
        ctol = tol
    minTol = stop_tolRatio*tol
    minGradTol = stop_gtolRatio*gtol

    xmin,xmax=np.minimum(xmin,xmax),np.maximum(xmin,xmax)
    xk = np.array(xinit)
    dim = len(xk)


    ineqCons,eqCons,multiFunGrad = constraintsWrapper(f,
                                            dim,
                                            gf=gf,
                                            constraints=constraints,
                                            precallfunc=precallfunc,
                                            gradientMethod=gradientMethod,
                                            dh=dh)


    #Initialisation
    fvec,gfvec = multiFunGrad(xk)
    fk,gfk = fvec[0],gfvec[:,0]
    cik,Jik = fvec[ineqCons],gfvec[:,ineqCons]
    cek,Jek = fvec[eqCons],gfvec[:,eqCons]
    ni,ne = len(cik),len(cek)
    muik,muek = np.zeros(ni),np.zeros(ne)
    muik,muek,activeSet = lsqLagrangian(gfk,cik,Jik,cek,Jek,muik,muek)
    gLk = gfk - Jik[:,activeSet].dot(muik[activeSet]) - Jek.dot(muek)
    phik = fk + penalityFactor*np.sum(cek**2) + penalityFactor*np.sum(cik[activeSet]**2)
    dphik = gfk + 2*penalityFactor*Jek.dot(cek) + 2*penalityFactor*Jik[:,activeSet].dot(cik[activeSet])

    #etat des iteration
    constrViol = np.append(cek,cik*(cik<0.0))
    cnorm = norm(constrViol)
    success = cnorm < ctol
    fobj = fk
    gnorm = norm(gLk)
    gnorm0 = max(1,gnorm)
    xnorm0 = max(1,norm(xk))
    phik_old = fk + norm(gfk)/2 #Assume dx=1
    gfk_old = None
    gnorm_old = None
    pk = -gfk
    Imat = np.eye(dim)
    Hmatk = np.eye(dim)
    Bmatk = np.eye(dim)

    iter = 0
    funcCalls = 1
    gradCalls = 1
    res = 1.0
    gres = 1.0
    if storeIterValues :
        xHistory = [xk]
        fHistory = [fobj]
        rHistory = [res]

    innerIter = 0
    while iter < maxIter :

        pk,muek,muik = solveQP(gfk,Bmatk,cik,Jik,cek,Jek,muik)


        print(pk)








    minDict = {"method":"SQP_BFGS",
                "success":success,
                "x":xk,
                "fmin":fobj,
                "residual":res,
                "gradResidual":gres,
                "grad": gfk,
                "constrViolation":constrViol,
                "iterations":iter,
                "functionCalls":funcCalls,
                "gradientCalls":gradCalls}

    if storeIterValues :
        minDict["xHistory"] = np.array(xHistory)
        minDict["fHistory"] = np.array(fHistory)
        minDict["rHistory"] = np.array(rHistory)
        return minDict

    if returnDict :
        return minDict

    return xk




def hessianInverseBFGS(Hmatk,xk,xknew,gfk,gfknew,Imat=None):
    if Imat is None :
        dim = Hmatk.shape[0]
        Imat = np.eye(dim) #Matrice identité
    sk = (xknew-xk)[np.newaxis].T
    yk = (gfknew-gfk)[np.newaxis].T

    inv_rhok = np.dot(yk.T,sk)
    if inv_rhok == 0.0 :
        inv_rhok = 1e-3

    A1mat = np.dot(sk,yk.T)/inv_rhok
    A2mat = np.dot(yk,sk.T)/inv_rhok
    A3mat = np.dot(sk,sk.T)/inv_rhok
    Hknew = np.dot((Imat-A1mat),np.dot(Hmatk,(Imat-A2mat))) + A3mat
    return Hknew

def hessianBFGS(Bmatk,xk,xknew,gfk,gfknew,Imat=None):
    if Imat is None :
        dim = Bmatk.shape[0]
        Imat = np.eye(dim) #Matrice identité
    sk = (xknew-xk)[np.newaxis].T
    yk = (gfknew-gfk)[np.newaxis].T

    rho1 = np.dot(sk.T,Bmatk.dot(sk))
    if rho1 == 0.0 :
        rho1 = 1e-3
    rho2 = np.dot(yk.T,sk)
    if rho2 == 0.0 :
        rho2 = 1e-3
    Bknew = Bmatk - np.dot(Bmatk.dot(sk),sk.T.dot(Bmatk))/rho1 + yk.dot(yk.T)/rho2
    return Bknew



def lsqLagrangian(gfk,cik0,Jik0,cek,Jek,muik,muek):

    activeSet = cik0 < 0.0
    cik = cik0[activeSet]
    Jik = Jik0[:,activeSet]
    dim = len(gfk)
    ne,ni = len(cek),len(cik)
    if ne + ni < dim :
        Kmat = np.zeros((dim,ne+ni))
        Kmat[:,:ne] = Jek
        Kmat[:,ne:] = Jik

        muk = solve(Kmat.T.dot(Kmat),-Kmat.T.dot(gfk))
        muik[activeSet] = muk[ne:]
        muek = muk[:ne]
    return muik,muek,activeSet


def solveQP(gfk,Bk,cik0,Jik0,cek,Jek,muik):
    activeSet = cik0 < 0.0
    cik = cik0[activeSet]
    Jik = Jik0[:,activeSet]
    dim = len(gfk)
    ne,ni = len(cek),len(cik)
    blocks = [[Bk,-Jek,-Jik],
              [Jek.T,None,None],
              [Jik.T,None,None]]
    Amat = bmat(blocks)
    bvec = np.zeros(dim+ne+ni)
    bvec[:dim] = -gfk
    bvec[dim:dim+ne:] = -cek
    bvec[dim+ne:] = -cik

    sk,_ = cg(Amat,bvec)

    pk = sk[:dim]
    muek = sk[dim:dim+ne:]
    muik[activeSet] = sk[dim+ne:]

    return pk,muek,muik




