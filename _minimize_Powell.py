import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from _penalization import penalizationFunction

__all__ = ["Powell"]


def Powell(f,xinit,xmin,xmax,tol=1e-6,maxIter=500,
                                    constraints=[],
                                    penalityFactor=100,
                                    storeIterValues=False,
                                    returnDict=False,
                                    precallfunc=None,
                                    updateDirectionMethod=None) :

    xmin,xmax=np.minimum(xmin,xmax),np.maximum(xmin,xmax)
    abs_x = np.abs(xinit)
    maxDeltaX = xmax-xmin
    dim = len(xinit)
    vecBase = np.eye(dim)
    yvec = np.zeros(dim)

    fpenal = penalizationFunction(f,constraints=constraints,
                                penalityFactor=penalityFactor,
                                precallfunc=precallfunc)

    xk = np.array(xinit)
    fk,success,constrViol,fobj = fpenal(xk)
    xnorm0 = norm(xk)+1e-20
    fnorm0 = max(abs(fk),1)

    iter = 0
    funcCalls = 1
    res = 1.0
    if storeIterValues :
        xHistory = [xk]
        fHistory = [fobj]
        rHistory = [res]
    while iter < maxIter :

        zj = xk
        maxdiff_ind = 0
        diff = 0.0
        fj = fk
        for j,pj in enumerate(vecBase) :

            (tj,
            ls_nfev,
            fknew,
            success,
            constrViol,
            fobj) = linesearchPowell(fpenal,zj,pj,xmin,xmax,tol=tol)

            zj = zj + tj*pj

            funcCalls += ls_nfev
            if (fj-fknew)>diff:
                maxdiff_ind = j
                diff = fj-fknew
            fj = fknew

        if updateDirectionMethod=="interpolation" :
            #Update set of directions
            f1 = fk.copy()
            f2 = fknew.copy()
            f3 = fpenal(2*zj-xk)[0]

            if (f3>=f1) or  ( (f1-2*f2+f3)*(f1-f2-diff)**2 >= 1/2*diff*(f1-f3)**2 ) :
                #keep same directions
                xknew = zj.copy()
                funcCalls += 1
            else :
                pn = zj-xk
                # pn = pn/(norm(pn)+1e-20)
                (tn,
                ls_nfev,
                fknew,
                success,
                constrViol,
                fobj) = linesearchPowell(fpenal,zj,pn,xmin,xmax,tol=tol)

                xknew = zj + tn*pn
                funcCalls += ls_nfev + 1

                #remove pm with m = maxdiff_ind
                #replace by pn
                vecBase[maxdiff_ind] = vecBase[-1]
                vecBase[-1] = pn

        else :
            xknew = zj.copy()
            funcCalls += 1
            pn = zj-xk
            # pn = pn/(norm(pn)+1e-20)
            vecBase[maxdiff_ind] = vecBase[-1]
            vecBase[-1] = pn


        #Update data
        fk_old = fk
        fk = fknew
        xk_old = xk
        xk = xknew
        dxnorm = norm(xk-xk_old)

        #residus
        res = dxnorm/xnorm0
        fres = 2.0 * abs(fk - fk_old)/(np.abs(fk) + np.abs(fk_old) + 1e-20)
        iter += 1



        if storeIterValues :
            rHistory.append(res)
            xHistory.append(xk)
            fHistory.append(fobj)

        #Convergence
        if res<=tol or fres<=tol :
            break





    minDict = {"method":"Powell",
                "success":success,
                "x":xk,
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

    return xk




def linesearchPowell(func,xk,pk,xmin,xmax,maxIter=100,tol=1e-6) :
    dim = len(pk)
    tmax = None
    for i,pki in enumerate(pk) :
        if pki < 0.0 :
            if tmax is None :
                tmax = (xmin[i]-xk[i])/pki
                tmin = (xmax[i]-xk[i])/pki
            else :
                tmax = min( (xmin[i]-xk[i])/pki,tmax )
                tmin = max( (xmax[i]-xk[i])/pki,tmin )
        if pki > 0.0 :
            if tmax is None :
                tmax = (xmax[i]-xk[i])/pki
                tmin = (xmin[i]-xk[i])/pki
            else :
                tmax = min( (xmax[i]-xk[i])/pki,tmax )
                tmin = max( (xmin[i]-xk[i])/pki,tmin )

    fresult = [None]
    xval = [None]
    def phi_func(t):
        fresult[:] = func(xk+t*pk)
        xval[0] = t
        return fresult[0]

    minres = minimize_scalar(phi_func,bounds=[tmin/2,tmax/2],
                                      method='bounded',
                                      options={"maxiter":maxIter})
    tknew = minres.x
    funcCalls = minres.nfev

    if tknew is None :
        funcCalls += 1
        fknew,success,constrViol,fobj = func(xk)
        return 0.0,funcCalls,fknew,success,constrViol,fobj

    if (xval[0] == xk + pk*tknew).all() :
        fknew,success,constrViol,fobj  = fresult[:]
    else :
        fknew,success,constrViol,fobj = func(xk + pk*tknew)
        funcCalls += 1


    return tknew,funcCalls,fknew,success,constrViol,fobj



