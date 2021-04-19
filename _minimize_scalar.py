
import numpy as np
from _numerical_deriv import approxDerivative
from _penalization import penalizationFunction

__all__ = ["goldenSearch","scalarGradient"]



## CONSTRAINED ALGORITHMS


def goldenSearch(f,xmin,xmax,tol=1e-6,maxIter=500,constraints=[],
                                                    penalityFactor=100,
                                                    storeIterValues=False,
                                                    returnDict=False,
                                                    precallfunc=None) :
    alpha1 = (3-np.sqrt(5))/2
    alpha2 = 1 - alpha1
    a,b=min(xmin,xmax),max(xmin,xmax)
    dxMax = b-a

    fpenal = penalizationFunction(f,constraints=constraints,
                                penalityFactor=penalityFactor,
                                precallfunc=precallfunc)

    iter = 0
    funcCalls = 0
    gradCalls = 0
    r = 1.0

    x1 = a + alpha1*(b-a)
    x2 = a + alpha2*(b-a)
    f1,sucess1,constrViol1,fobj1 = fpenal(x1)
    f2,sucess2,constrViol2,fobj2 = fpenal(x2)


    if storeIterValues :
        xHistory = [x2,x1]
        fHistory = [f2,f1]
        rHistory = [1.0,1.0]

    funcCalls += 2

    h = b-a
    while r >tol and iter<maxIter :
        iter += 1

        if f1 < f2:
            b = x2
            x2 = x1
            f2,sucess2,constrViol2,fobj2 = f1,sucess1,constrViol1,fobj1
            h = alpha2 * h
            x1 = a + alpha1 * h
            f1,sucess1,constrViol1,fobj1 = fpenal(x1)
            if storeIterValues :
                xHistory.append(x2)
                fHistory.append(fobj2)
        else:
            a = x1
            x1 = x2
            f1 = f2
            f1,sucess1,constrViol1,fobj1 = f2,sucess2,constrViol2,fobj2
            h = alpha2 * h
            x2 = a + alpha2 * h
            f2,sucess2,constrViol2,fobj2 = fpenal(x2)
            if storeIterValues :
                xHistory.append(x1)
                fHistory.append(fobj1)

        funcCalls += 1
        r = h/dxMax
        if storeIterValues :
            rHistory.append(r)

    if f1 < f2:
        xmin = (a + x2)/2
    else:
        xmin = (b + x1)/2


    y,success,constrViol,fobj = fpenal(xmin)
    minDict = {"method":"goldenSearch",
                "success":success,
                "x":xmin,
                "fmin":fobj,
                "constrViolation":constrViol,
                "residual":r,
                "iterations":iter,
                "functionCalls":funcCalls+1,
                "derivfCalls":gradCalls}
    if storeIterValues :
        minDict["xHistory"] = np.array(xHistory)
        minDict["fHistory"] = np.array(fHistory)
        minDict["rHistory"] = np.array(rHistory)
        return minDict

    if returnDict :
        return minDict

    return xmin



def lineSearchArmijo(f,xk,fk,gk,pk,xmin=None,xmax=None,
                            old_fk=None,
                            c1=0.0001,
                            maxIter=10,
                            backtracking = False):

    if old_fk is not None :
        t0 = 2*(fk-old_fk)/(gk*pk)
        t0 = min(0.8*(xmax-xmin),t0)
    else :
        t0 = 0.8*(xmax-xmin)

    if t0 < 0 :
        t0 = 0.8*(xmax-xmin)

    x = np.maximum(np.minimum(xmax,xk+t0*pk),xmin)
    t0 = (x-xk)/pk

    phi_0 = fk
    phi_t0,success,constrViol_t0,fobj_t0 = f(xk+t0*pk)
    dphi_0 = gk*pk
    funcCalls = 1

    if phi_t0 <= (phi_0+t0*c1*dphi_0) :
        return t0,funcCalls,phi_t0,success,constrViol_t0,fobj_t0


    #Quadratique minimisation
    a = ( phi_t0-t0*dphi_0-phi_0)/t0**2
    b = dphi_0
    t1 =  -b/(2*a)
    if t1>0 and t1<t0 :
        phi_t1,success,constrViol,fobj  = f(xk+t1*pk)
    else :
        t1 = t0/2
        phi_t1,success,constrViol,fobj  = f(xk+t1*pk)
    funcCalls += 1

    if phi_t1 <= (phi_0+t1*c1*dphi_0) :
        return t1,funcCalls,phi_t1,success,constrViol,fobj


    iter = 0
    while iter < maxIter :
        #interpolation cubique
        factor = t0**2 * t1**2 * (t1-t0)
        a = t0**2 * (phi_t1 - phi_0 - dphi_0*t1) - \
            t1**2 * (phi_t0 - phi_0 - dphi_0*t0)
        a = a / factor
        b = -t0**3 * (phi_t1 - phi_0 - dphi_0*t1) + \
            t1**3 * (phi_t0 - phi_0 - dphi_0*t0)
        b = b / factor
        t2 = (-b + np.sqrt(abs(b**2 - 3 * a * dphi_0))) / (3.0*a)

        if t2<0 or t2 > t0 :
            t2 = t1/2

        phi_t2,sucess,constrViol,fobj = f(xk+t2*pk)
        funcCalls += 1

        if (phi_t2 <= phi_0 + c1*t2*dphi_0):
            return t2,funcCalls,phi_t2,success,constrViol,fobj
        t0 = t1
        t1 = t2
        phi_t0 = phi_t1
        phi_t1 = phi_t2

        iter += 1


    return t1,funcCalls,phi_t1,False,constrViol,fobj


def scalarGradient(f,xinit,xmin,xmax,tol=1e-6,gtol=1e-6,maxIter=500,gf=None,
                                                    constraints=[],
                                                    penalityFactor=10,
                                                    storeIterValues=False,
                                                    returnDict=False,
                                                    deriveMethod="finite-difference",
                                                    dh=1e-9,
                                                    stop_tolRatio=1e-2,
                                                    stop_gtolRatio=1e-2,
                                                    precallfunc=None,
                                                    backtracking=False) :

    minTol = stop_tolRatio*tol
    minGradTol = stop_gtolRatio*gtol

    xmin,xmax=min(xmin,xmax),max(xmin,xmax)
    dxMax = xmax-xmin
    abs_x = np.abs(xinit)

    fpenal = penalizationFunction(f,constraints=constraints,
                                penalityFactor=penalityFactor,
                                precallfunc=precallfunc)

    if gf is None or constraints :
        dh = max(abs_x*dh,dh)
        gf = approxDerivative(fpenal,method=deriveMethod,dh=dh)

    xk = np.array(xinit)
    fk,success,constrViol,fobj = fpenal(xk)
    gfk = gf(xk)
    gnorm0 = max(1,np.abs(gfk))
    xnorm0 = max(1,np.abs(xk))
    fk_old = None
    gfk_old = None


    iter = 0
    funcCalls = 1
    gradCalls = 1
    res = 1.0
    gres = 1.0
    if storeIterValues :
        xHistory = [xk]
        fHistory = [fobj]
        rHistory = [res]
    while iter < maxIter :

        pk = -gfk/np.abs(gfk)

        (alpha,
        ls_nfev,
        fknew,
        success,
        constrViol,
        fobj) = lineSearchArmijo(fpenal,xk,fk,gfk,pk,xmin,xmax,old_fk=fk_old)
        xknew = xk + alpha*pk

        res = np.abs(xknew-xk)/dxMax
        xk = xknew
        fk_old = fk
        fk = fknew
        gfk_old = gfk
        gfk = gf(xk)
        gres = np.abs(gfk)/gnorm0
        pkold = pk


        iter += 1
        funcCalls += ls_nfev
        gradCalls += 1

        if storeIterValues :
            xHistory.append(xk)
            fHistory.append(fobj)
            rHistory.append(res)

        #Convergence
        if (res<=tol and gres<=gtol) or res<=minTol or gres<=minGradTol :
            break

    minDict = {"method":"gradient",
                "success":success,
                "x":xk,
                "fmin":fobj,
                "residual":res,
                "gradResidual":gres,
                "gradNorm":abs(gfk),
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



