import numpy as np
from numpy.linalg import norm
from _linesearch import linesearch
from _penalization import constraintOptimizationWrapper

__all__ = ["conjugateGradient"]



def conjugateGradient(f,xinit,xmin,xmax,
                        tol=1e-6,
                        gtol=1e-6,
                        maxIter=500,
                        gf=None,
                        constraints=[],
                        penalityFactor=100.0,
                        penalInnerIter =None ,
                        storeIterValues=False,
                        returnDict=False,
                        gradientMethod="finite-difference",
                        linesearchMethod="wolfe",
                        dh=1e-9,
                        stop_tolRatio=1e-2,
                        stop_gtolRatio=1e-2,
                        precallfunc=None,
                        methodConjugate="PR") :

    minTol = stop_tolRatio*tol
    minGradTol = stop_gtolRatio*gtol

    xmin,xmax=np.minimum(xmin,xmax),np.maximum(xmin,xmax)
    abs_x = np.abs(xinit)
    dim = len(xinit)

    optiConstrWrapper = constraintOptimizationWrapper(f,
                                                    dim,
                                                    gf=gf,
                                                    constraints=constraints,
                                                    precallfunc=precallfunc,
                                                    penalityFactor=penalityFactor,
                                                    gradientMethod=gradientMethod,
                                                    dh=dh)


    if penalInnerIter is None :
        penalInnerIter = maxIter + 1
    optiConstrWrapper.initPenalizationMethod()
    fpenal = optiConstrWrapper.penalFunction
    gfpenal = optiConstrWrapper.penalGradient


    xk = np.array(xinit)
    fk,success,constrViol,fobj = fpenal(xk)
    gfk = gfpenal(xk)
    gnorm = norm(gfk)
    gnorm0 = max(1,gnorm)
    xnorm0 = max(1,norm(xk))
    fk_old = fk + norm(gfk)/2 #Assume dx=1
    gfk_old = None
    gnorm_old = None
    pk = -gfk


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

        (alpha,
        ls_nfev,
        ls_ngrad,
        fknew,
        success,
        constrViol,
        fobj)  = linesearch(fpenal,gfpenal,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = fk_old,
                            c1=0.0001,
                            c2=0.4,
                            linesearchMethod=linesearchMethod)


        #Update data
        fk_old = fk
        fk = fknew
        xk_old = xk
        xk = xk + pk*alpha
        gfk_old = gfk
        gnorm_old = gnorm
        gnorm = norm(gfk)
        gfk = gfpenal(xk)

        if ( (iter - innerIter) >= penalInnerIter ) :
            optiConstrWrapper.increasePenalization()
            optiConstrWrapper.initPenalizationMethod()
            innerIter = iter

        #residus
        res = norm(xk-xk_old)/xnorm0
        gres = gnorm/gnorm0

        #Conjugate gradient direction

        yk = gfk-gfk_old
        if methodConjugate == "PR" :
            beta_k = max(0.0, np.dot(gfk,yk) / np.dot(gfk_old,gfk_old) )

        elif methodConjugate == "FR" :
            beta_k = np.dot(gfk,gfk)/np.dot(gfk_old,gfk_old)

        elif methodConjugate == "HS" :
            beta_k = np.dot(gfk,gfk)/np.dot(pk,yk)

        elif methodConjugate == "DY" :
            beta_k = np.dot(gfk,gfk)/np.dot(pk,yk)

        else :
            beta_k = 0.0

        pk = -gfk + beta_k*pk



        iter += 1
        funcCalls += ls_nfev
        gradCalls += ls_ngrad + 1

        if storeIterValues :
            xHistory.append(xk)
            fHistory.append(fobj)
            rHistory.append(res)

        #Convergence
        if (res<=tol and gres<=gtol) or res<=minTol or gres<=minGradTol :
            break





    minDict = {"method":"conjugateGradient",
                "success":success,
                "x":xk,
                "fmin":fobj,
                "residual":res,
                "gradResidual":gres,
                "grad":gfk,
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




