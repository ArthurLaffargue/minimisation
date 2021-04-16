import numpy as np
from _numerical_deriv import approxPenalGradient,multipleFuncAndGradEvaluator

__all__ = ["penalizationFunction",
            "constraintOptimizationWrapper",
            "constraintsWrapper"]

def penalizationFunction(f,constraints=[],precallfunc=None,penalityFactor=1,absTol=0.0):

    def fpenal(x):
        if precallfunc :
            precallfunc(x)
        objective = f(x)
        feasibility = True
        penality = 0.0
        constrViolation = []
        for c in constraints :
            type = c["type"]
            g = c['fun']
            gi = g(x)

            if type == 'strictIneq'  :
                feasibility &= gi>0
                constrViol = np.minimum(gi,0.0)
                constrViolation.append(constrViol)
                penality += constrViol**2

            if type == 'ineq' :
                feasibility &= gi>=0
                constrViol = np.minimum(gi,0.0)
                constrViolation.append(constrViol)
                penality += constrViol**2

            if type == 'eq'  :
                constrViol = gi
                feasibility &= constrViol<=absTol
                constrViolation.append(constrViol)
                penality += constrViol**2
        penalObjective = objective + penalityFactor*penality
        return penalObjective,feasibility,constrViolation,objective

    return fpenal




def transformConstrFunc(fc,type):
    if type == 'ineq'  :

        phi_c = lambda x : np.minimum(fc(x),0.0)

    if type == 'eq'  :
        phi_c = fc

    return phi_c




class constraintOptimizationWrapper :

    def __init__(self,func,
                      dim,
                      gf=None,
                      constraints=[],
                      precallfunc=None,
                      penalityFactor=100.0,
                      gradientMethod="finite-difference",
                      dh=1e-12) :


        nc = len(constraints)
        self.nc = nc
        self.funcList = [func]
        self.gradientList = [gf]

        self.dim = dim
        self.dh = dh
        self.gradientMethod = gradientMethod
        self.precallfunc = precallfunc

        self.cvals = np.zeros(nc)
        self.fval = 0.0
        self.xfeval = None

        self.rho = penalityFactor
        self.rhoFactor = 2.0

        if gf is None :
            self.exactGradientCondition = False
        else :
            self.exactGradientCondition = True

        self.constrType = []
        for i,ci in enumerate(constraints) :
            fc = ci["fun"]
            type = ci["type"]
            self.constrType.append(type)

            if "jac" in ci :
                gc = ci["jac"]
            else :
                gc = None
                self.exactGradientCondition = False

            phi_c = transformConstrFunc(fc,type)
            self.funcList += [phi_c]
            self.gradientList += [gc]


    ## Penalization method

    def initPenalizationMethod(self):

        if self.exactGradientCondition :
            self.penalGradient = self.exactPenalGradient
        else :
            self.penalGradient = approxPenalGradient(self.penalFunction,
                                                self.dim,
                                                method=self.gradientMethod,
                                                dh=self.dh)


    def penalFunction(self,x):
        self.xfeval = x
        if self.precallfunc :
            self.precallfunc(x)
        self.fval = self.funcList[0](x)
        for k,fk in enumerate(self.funcList[1:]) :
            self.cvals[k] = fk(x)

        self.fpenal_val = self.fval + self.rho*np.sum(self.cvals**2)
        feasibility = (self.cvals == 0.0).all()

        return self.fpenal_val,feasibility,self.cvals,self.fval



    def exactPenalGradient(self,x):

        if (self.xfeval == x).all() :
            self.cvals = self.cvals
        else :
            if self.precallfunc :
                self.precallfunc(x)
            for k,fk in enumerate(self.funcList[1:]) :
                self.cvals[k] = fk(x)


        self.gfpenal_val = self.gradientList[0](x)
        for k in range(self.nc) :
            self.gfpenal_val += 2.0*self.rho*self.cvals[k]*self.gradientList[k+1](x)

        return self.gfpenal_val


    def increasePenalization(self):
        self.rho = self.rho*self.rhoFactor




def constraintsWrapper(func,
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
