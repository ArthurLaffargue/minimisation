from scipy.optimize.linesearch import *
from scipy.optimize import minimize_scalar
import numpy as np


def linesearch(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = None,
                            c1=0.0001,
                            c2=0.9,
                            extra_condition=None,
                            linesearchMethod="wolfe",
                            t0 = None,
                            maxIter=10):


    if linesearchMethod == "wolfe" :
        return wolfeLineSearch(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = fk_old,
                            c1=c1,
                            c2=0.1,
                            extra_condition=extra_condition)


    if linesearchMethod == "armijo" :
        return armijoLineSearch(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = None,
                            c1=c1,
                            c2=0.4,
                            maxIter=maxIter,
                            t0=t0,
                            extra_condition=extra_condition)

    if linesearchMethod == "wolfe2" :
        return wolfeLineSearch2(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = fk_old,
                            c1=c1,
                            c2=c2,
                            t0=t0,
                            maxIter=maxIter,
                            extra_condition=extra_condition)


    if linesearchMethod == "exact" :
        return exactLineSearch(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = fk_old,
                            c1=c1,
                            c2=0.1,
                            extra_condition=extra_condition)

def wolfeLineSearch2(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = None,
                            c1=0.0001,
                            c2=0.4,
                            maxIter=10,
                            t0=None,
                            extra_condition=None):


    phi_func = lambda t : func(xk+t*pk)
    dphi_func = lambda t : np.dot(pk,gradf(xk+pk*t))
    phi_0 = fk
    dphi_0 = np.dot(gfk,pk)

    dim = len(pk)
    if t0 is None :
        if fk_old is not None :
            t0 = 2*(fk-fk_old)/(dphi_0)
            t0 = min(1.01*t0,1.0)

            if t0 <= 0 : t0 = 1.0
        else :
            t0 = 1.0

    #Restriction a xmin xmax
    x = np.minimum(xmax,np.maximum(xk + t0*pk,xmin))
    filtre = np.abs(pk)>0
    if filtre.any() :
        t0 = np.min( (x-xk)[filtre]/pk[filtre] )
    else :
        phi_t0,success,constrViol,fobj = phi_func(0.0)
        funcCalls,gradCalls = 1,0
        return 0.0,funcCalls,gradCalls,phi_t0,False,constrViol,fobj

    #Evaluation du point t0
    phi_t0,success,constrViol,fobj = phi_func(t0)
    dphi_t0 = dphi_func(t0)
    funcCalls = 1
    gradCalls = 1
    if phi_t0 <= (phi_0+t0*c1*dphi_0) and dphi_t0 >= c2*dphi_0 :
        return t0,funcCalls,gradCalls,phi_t0,success,constrViol,fobj


    #Evaluation du point t1
    a = ( phi_t0-t0*dphi_0-phi_0)/t0**2
    b = dphi_0
    t1 =  -b/(2*a)
    if t1>0 and t1<t0 :
        phi_t1,success,constrViol,fobj  = phi_func(t1)
    else :
        t1 = t0/2
        phi_t1,success,constrViol,fobj  = phi_func(t1)
    dphi_t1 = dphi_func(t1)
    funcCalls += 1
    gradCalls += 1


    if phi_t1 <= (phi_0+t1*c1*dphi_0) and dphi_t1 >= c2*dphi_0 :
        return t1,funcCalls,gradCalls,phi_t1,success,constrViol,fobj


    #Utilisation d'une boucle
    iter = 0
    ta = 0
    tb = t0
    while iter < maxIter :
        if phi_t1 > (phi_0+t1*c1*dphi_0) :
            tb = t1
            t1 = (ta+tb)/2

            phi_t1,success,constrViol,fobj
            dphi_t1 = dphi_func(t1)
            funcCalls += 1
            gradCalls += 1
        else  :
            ta = t1
            t1 = (ta+tb)/2

            phi_t1,success,constrViol,fobj
            dphi_t1 = dphi_func(t1)
            funcCalls += 1
            gradCalls += 1

        if phi_t1 <= (phi_0+t1*c1*dphi_0) and dphi_t1 >= c2*dphi_0 :
            return t1,funcCalls,gradCalls,phi_t1,success,constrViol,fobj

        iter += 1

    return t1,funcCalls,gradCalls,phi_t1,False,constrViol,fobj














def armijoLineSearch(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = None,
                            c1=0.0001,
                            c2=0.4,
                            maxIter=10,
                            t0=None,
                            extra_condition=None):

    phi_func = lambda t : func(xk+t*pk)
    phi_0 = fk
    dphi_0 = np.dot(gfk,pk)

    dim = len(pk)
    if t0 is None :
        if fk_old is not None :
            t0 = 2*(fk-fk_old)/(dphi_0)
            t0 = min(1.01*t0,1.0)

            if t0 <= 0 : t0 = 1.0
        else :
            t0 = 1.0

    #Restriction a xmin xmax
    x = np.minimum(xmax,np.maximum(xk + t0*pk,xmin))
    filtre = np.abs(pk)>0
    if filtre.any() :
        t0 = np.min( (x-xk)[filtre]/pk[filtre] )
    else :
        phi_t0,success,constrViol,fobj = phi_func(0.0)
        funcCalls,gradCalls = 1,0
        return 0.0,funcCalls,gradCalls,phi_t0,False,constrViol,fobj


    #Evaluation du point t0
    phi_t0,success,constrViol,fobj = phi_func(t0)
    funcCalls = 1
    gradCalls = 0
    if phi_t0 <= (phi_0+t0*c1*dphi_0) :
        return t0,funcCalls,gradCalls,phi_t0,success,constrViol,fobj

    #Evaluation du point t1
    a = ( phi_t0-t0*dphi_0-phi_0)/t0**2
    b = dphi_0
    t1 =  -b/(2*a)
    if t1>0 and t1<t0 :
        phi_t1,success,constrViol,fobj  = phi_func(t1)
    else :
        t1 = t0/2
        phi_t1,success,constrViol,fobj  = phi_func(t1)
    funcCalls += 1

    if phi_t1 <= (phi_0+t1*c1*dphi_0) :
        return t1,funcCalls,gradCalls,phi_t1,success,constrViol,fobj

    #Utilisation d'une boucle
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

        phi_t2,sucess,constrViol,fobj = phi_func(t2)
        funcCalls += 1

        if (phi_t2 <= phi_0 + c1*t2*dphi_0)  :
            return t2,funcCalls,gradCalls,phi_t2,success,constrViol,fobj
        t0 = t1
        t1 = t2
        phi_t0 = phi_t1
        phi_t1 = phi_t2

        iter += 1


    return t1,funcCalls,gradCalls,phi_t1,False,constrViol,fobj


def wolfeLineSearch(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = None,
                            c1=0.0001,
                            c2=0.4,
                            extra_condition=None):

    dim = len(pk)
    tmax = None
    for i,pki in enumerate(pk) :
        if pki < 0.0 :
            if tmax is None :
                tmax = (xmin[i]-xk[i])/pki
            else : tmax = min( (xmin[i]-xk[i])/pki,tmax )
        if pki > 0.0 :
            if tmax is None :
                tmax = (xmax[i]-xk[i])/pki
            else : tmax = min( (xmax[i]-xk[i])/pki,tmax )



    fresult = [None]
    xval = [None]
    def phi_func(x):
        fresult[:] = func(x)
        xval[0] = x
        return fresult[0]


    lsres = line_search_wolfe2(phi_func,
                        gradf,
                        xk,
                        pk,
                        gfk=gfk,
                        old_fval=fk,
                        old_old_fval=fk_old,
                        c1=c1,
                        c2=c2,
                        amax=tmax,
                        extra_condition=extra_condition,
                        maxiter=10)

    (tknew,fc,gc,fknew,_,_) = lsres
    funcCalls = fc
    gradCalls = gc

    if tknew is None :
        funcCalls += 1
        fknew,success,constrViol,fobj = func(xk)
        return 0.0,funcCalls,gradCalls,fknew,success,constrViol,fobj

    if (xval[0] == xk + pk*tknew).all() :
        fknew,success,constrViol,fobj  = fresult[:]
    else :
        fknew,success,constrViol,fobj = func(xk + pk*tknew)
        funcCalls += 1


    return tknew,funcCalls,gradCalls,fknew,success,constrViol,fobj




def exactLineSearch(func,gradf,xk,fk,gfk,pk,xmin,xmax,
                            fk_old = None,
                            c1=0.0001,
                            c2=0.4,
                            extra_condition=None):

    dim = len(pk)
    tmax = None
    for i,pki in enumerate(pk) :
        if pki < 0.0 :
            if tmax is None :
                tmax = (xmin[i]-xk[i])/pki
            else : tmax = min( (xmin[i]-xk[i])/pki,tmax )
        if pki > 0.0 :
            if tmax is None :
                tmax = (xmax[i]-xk[i])/pki
            else : tmax = min( (xmax[i]-xk[i])/pki,tmax )



    fresult = [None]
    xval = [None]
    def phi_func(t):
        fresult[:] = func(xk+t*pk)
        xval[0] = t
        return fresult[0]


    minres = minimize_scalar(phi_func,bounds=[0.0,tmax/2],method='bounded')
    tknew = minres.x
    funcCalls = minres.nfev
    gradCalls = 0

    if tknew is None :
        funcCalls += 1
        fknew,success,constrViol,fobj = func(xk)
        return 0.0,funcCalls,gradCalls,fknew,success,constrViol,fobj

    if (xval[0] == xk + pk*tknew).all() :
        fknew,success,constrViol,fobj  = fresult[:]
    else :
        fknew,success,constrViol,fobj = func(xk + pk*tknew)
        funcCalls += 1


    return tknew,funcCalls,gradCalls,fknew,success,constrViol,fobj