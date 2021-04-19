import numpy as np

__all__ = ["approxPenalGradient","approxDerivative","multipleFuncAndGradEvaluator"]

def approxDerivative(fpenal,dh=1e-12,method="finite-difference"):
    if method == 'complex' :
        df = lambda xk : np.imag(fpenal(xk+1j*dh)[0]/dh)
    if method == "finite-difference" :
        df = lambda xk : (fpenal(xk+dh)[0]-fpenal(xk-dh)[0])/(2*dh)
    return df


def approxPenalGradient(fpenal,dim,dh=1e-12,method="finite-difference"):
    if method == 'complex' :
        def gradf(xk):
            xi = xk + 1j*np.eye(dim)*dh
            gfk = np.imag([fpenal(xij)[0] for xij in xi])/dh
            return gfk.T

    if method == "finite-difference" :
        def gradf(xk):
            fk = fpenal(xk)[0]
            xi = xk + np.eye(dim)*dh
            gfk = np.array([ (fpenal(xij)[0]-fk)/dh for xij in xi ])
            return gfk.T
    return gradf





def multipleFuncAndGradEvaluator(funcList,
                            gradientList,
                            dim,
                            dh=1e-12,
                            precallfunc=None):

    nf = len(funcList)
    def multipleEval(x,geval=True):
        gfMat = np.zeros((dim,nf))
        fVec = np.zeros(nf)

        if precallfunc is not None :
            precallfunc(x)
        # evaluation des fonctions
        for k,fk in enumerate(funcList):
            fVec[k] = fk(x)
        if geval == False :
            return fVec
        # evaluation des gradients
        allGradient = True
        for k,gfk in enumerate(gradientList) :
            if gfk is not None :
                gfMat[:,k] = gfk(x)
            else :
                allGradient = False

        if allGradient :
            return fVec,gfMat

        #approximations
        xi = x + np.eye(dim)*dh
        for j,xij in enumerate(xi):
            if precallfunc is not None :
                precallfunc(xij)
            for k,(fk,gfk,fvalk) in enumerate(zip(funcList,gradientList,fVec)):
                if gfk is None :
                    gfMat[j,k] = (fk(xij)-fvalk)/dh
        return fVec,gfMat

    return multipleEval