import numpy as np
from numpy.linalg import norm
from _penalization import penalizationFunction

__all__ = ["NelderMead"]


def NelderMead(f,xinit,xmin,xmax,tol=1e-6,maxIter=500,
                                    constraints=[],
                                    penalityFactor=100,
                                    storeIterValues=False,
                                    returnDict=False,
                                    precallfunc=None) :

    # alpha = 1.0
    # gamma = 2.0
    # rho = 0.5
    # sigma = 0.5
    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5

    xmin,xmax=np.minimum(xmin,xmax),np.maximum(xmin,xmax)
    dim = len(xinit)

    fpenal = penalizationFunction(f,constraints=constraints,
                                penalityFactor=penalityFactor,
                                precallfunc=precallfunc)

    xk = np.array(xinit)
    fk,success,constrViol,fobj = fpenal(xk)

    simplex,fsimplex = initSimplex(xk,fk,fpenal,xmin,xmax)
    index = np.argsort(fsimplex)
    simplex = simplex[index]
    fsimplex = fsimplex[index]

    df0 = np.max(np.abs(fsimplex[1:] - fsimplex[0]))
    dx0 = np.max(np.abs(simplex[1:] - simplex[0]))

    iter = 0
    funcCalls = dim+1
    res = 1.0
    if storeIterValues :
        xHistory = [xk]
        fHistory = [fk]
        rHistory = [res]

    while iter < maxIter :

        xbar = np.sum(simplex[:-1],axis=0)/dim
        xr = (1 + rho) * xbar - rho * simplex[-1]
        xr = np.minimum(np.maximum(xmin,xr),xmax)
        fr = fpenal(xr)[0]
        funcCalls += 1
        doshrink = 0

        if fr < fsimplex[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * simplex[-1]
            xe = np.minimum(np.maximum(xmin,xe),xmax)
            fe = fpenal(xe)[0]
            funcCalls += 1
            if fe < fr:
                simplex[-1] = xe
                fsimplex[-1] = fe
            else:
                simplex[-1] = xr
                fsimplex[-1] = fr

        else : #fr>=f0
            if fr < fsimplex[-2]:
                simplex[-1] = xr
                fsimplex[-1] = fr
            else:  # fr >= fn
                if fr < fsimplex[-1] :
                    # Contraction
                    xc = (1 + psi * rho) * xbar - psi * rho * simplex[-1]
                    xc = np.minimum(np.maximum(xmin,xc),xmax)
                    fc = fpenal(xc)[0]
                    funcCalls += 1
                    if fc < fsimplex[-1]:
                        simplex[-1] = xc
                        fsimplex[-1] = fc

                    else :
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * simplex[-1]
                    xcc = np.minimum(np.maximum(xmin,xcc),xmax)
                    fcc = fpenal(xcc)[0]
                    funcCalls += 1

                    if fcc < fsimplex[-1]:
                        simplex[-1] = xcc
                        fsimplex[-1] = fcc
                    else:
                        doshrink = 1

            if doshrink :
                x1 = simplex[0]
                for j,xj in enumerate(simplex[1:]):
                    xj = x1 + sigma * (xj - x1)
                    xj = np.minimum(np.maximum(xmin,xj),xmax)
                    simplex[j+1] = np.minimum(np.maximum(xmin,xj),xmax)
                    fsimplex[j+1] = fpenal(simplex[j+1])[0]
                    funcCalls += dim

        index = np.argsort(fsimplex)
        simplex = simplex[index]
        fsimplex = fsimplex[index]

        #update
        iter += 1
        dx = np.max(np.abs(simplex[1:] - simplex[0]))
        df = np.max(np.abs(fsimplex[0] - fsimplex[1:]))

        fopt = fsimplex[0]
        xopt = simplex[0]

        res = dx/dx0
        fres = df/df0

        #historisation
        if storeIterValues :
            rHistory.append(res)
            xHistory.append(xopt)
            fHistory.append(fopt)

        #Convergence
        if res<=tol or fres<=tol :
            break


    #Finalisation
    fk,success,constrViol,fobj = fpenal(xopt)
    minDict = {"method":"Nelder Mead",
                "success":success,
                "x":xopt,
                "fmin":fobj,
                "residual":res,
                "fresidual":fres,
                "constrViolation":constrViol,
                "iterations":iter,
                "functionCalls":funcCalls}

    if storeIterValues :
        minDict["xHistory"] = np.array(xHistory)
        minDict["fHistory"] = np.array(fHistory)
        minDict["rHistory"] = np.array(rHistory)
        return minDict

    if returnDict :
        return minDict

    return xopt



def initSimplex(x0,f0,fpenal,xmin,xmax,nonzdelt = 0.025,zdelt = 0.00025):

    N = len(x0)
    dmax = xmax-xmin

    sim = np.zeros((N + 1, N), dtype=float)
    sim[0] = x0
    y = np.zeros(N,dtype=float)
    for k in range(N):
        y[:] = x0[:]
        if y[k] != 0.0 :
            y[k] = (1.0 + nonzdelt)*y[k]
        else:
            y[k] = zdelt*dmax[k]

        y = np.maximum(np.minimum(xmax,y),xmin)
        sim[k+1] = y

    fsimplex = np.zeros((N+1),dtype=float)
    fsimplex[0] = f0
    for k,xk in enumerate(sim[1:]) :
        fsimplex[k+1] = fpenal(xk)[0]

    return sim,fsimplex


