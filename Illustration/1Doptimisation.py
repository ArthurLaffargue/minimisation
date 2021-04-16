import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search
plt.rc('font',family='Serif')

x = np.linspace(2.7,7.5,250)
f = lambda x : np.sin(x) + np.sin(10/3*x)
df = lambda x : np.imag(f(x+1j*1e-6)/1e-6)

y = f(x)
dy = df(x)

zeroGrad = np.abs(dy)<1e-1
optiGlob = (y == y.max())|(y ==y.min())
optiLoc = zeroGrad
optiLoc[optiGlob] = False

plt.figure(figsize=(9,3.5))
plt.plot(x,y,color='k',label='f(x)')
plt.plot(x[optiGlob],y[optiGlob],
        ls='',
        marker='D',
        markeredgecolor='k',
        markerfacecolor='r',
        label='Optimum globaux')
plt.plot(x[optiLoc],y[optiLoc],
        ls='',
        marker='o',
        markeredgecolor='k',
        markerfacecolor='c',
        label='Optimum locaux')

plt.title("$f(x) = sin(x) + sin(10/3x)$",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("f(x)",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("OptimumLocGlob.svg",dpi=150)





def goldenSection(f,a,b):
    alpha1 = (3-np.sqrt(5))/2
    alpha2 = 1 - alpha1
    a,b=min(a,b),max(a,b)

    X = [a,b]

    while 2*(b-a)/(a+b) >1e-6 :
        x1 = a + alpha1*(b-a)
        x2 = a + alpha2*(b-a)

        f1 = f(x1)
        f2 = f(x2)

        if f1<f2 :
            b = x2
            X.append(x2)
        else :
            a = x1
            X.append(x1)
    return (a+b)/2,np.array(X)

def armijoLineSearch(f,xk,fk,gk,pk,xmin,xmax):
    beta = 0.001
    alpha = 0.33*(xmax-xmin)/np.abs(pk)
    gamma = 0.5
    x = max(min(xmax,xk+alpha*pk),xmin)
    alpha = (x-xk)/pk
    maxIter = 100
    iter = 0
    while (f(xk+alpha*pk) > (fk+alpha*beta*gk*pk)) and iter<maxIter :
        alpha = gamma*alpha
        iter += 1
    return alpha

def wolfLineSearch(f,xk,fk,gk,pk,xmin,xmax):
    c1=0.0001
    c2=0.9
    alpha = 0
    beta = None
    tau = 0.5

    t = 0.33*(xmax-xmin)/np.abs(pk)
    x = max(min(xmax,xk+t*pk),xmin)
    t = (x-xk)/pk
    maxIter = 100

    for i in range(maxIter) :
        if f(xk + t*pk) > fk + c1*t*pk*gk :
            beta = t
            t = 0.5*(alpha+beta)
        elif pk*df(xk+pk*t) < c2*gk :
            alpha = t
            if beta is None :
                t = 2*alpha
            else :
                t = 0.5*(alpha+beta)
        else :
            return t

        x = max(min(xmax,xk+t*pk),xmin)
        t = (x-xk)/pk
    return alpha

def gradDesc(f,x0,a,b):
    gf = lambda x : df(x)

    a,b=min(a,b),max(a,b)
    step = (b-a)/200
    gf0 = gf(x0)
    f0 = f(x0)
    res = 1
    X = [x0]
    maxIter = 200
    iter = 0
    while res > 1e-6 and maxIter>iter:

        pk = -gf0

        alpha_max = step
        x1 = np.linspace(x0,max(min(x0+pk*step,b),a),150)
        index = np.argmin(f(x1))
        x1 = x1[index]

        res = np.abs(x1-x0)/x0
        x0 = x1.copy()
        gf0 = gf(x0)
        f0 = f(x0)
        X.append(x0)
        iter += 1
    return x0,np.array(X)

def gradDescArmijo(f,x0,a,b):
    gf = lambda x : df(x)

    a,b=min(a,b),max(a,b)
    step = (b-a)/200
    gf0 = gf(x0)
    f0 = f(x0)
    res = 1
    X = [x0]
    maxIter = 200
    iter = 0
    while res > 1e-6 and maxIter>iter:

        pk = -gf0

        alpha = armijoLineSearch(f,x0,f0,gf0,pk,a,b)
        # alpha = wolfLineSearch(f,x0,f0,gf0,pk,a,b)
        x1 = x0 + alpha*pk

        res = np.abs(x1-x0)/x0
        x0 = x1.copy()
        gf0 = gf(x0)
        f0 = f(x0)
        X.append(x0)
        iter += 1
    return x0,np.array(X)



xgold,Xgold = goldenSection(f,2.7,7.5)
xgrad,Xgrad = gradDesc(f,2.7,2.7,7.5)
xgradArmijo,XgradArmijo = gradDescArmijo(f,2.7,2.7,7.5)

plt.figure(figsize=(9,3.5))

plt.plot(x,y,color='grey')
plt.plot(Xgold,f(Xgold),ls='',marker='D',markeredgecolor='k',label='Golden-Search')
plt.plot(XgradArmijo,f(XgradArmijo),ls='',marker='s',markeredgecolor='k',label='Gradient-Armijo')
plt.plot(Xgrad,f(Xgrad),ls='',marker='o',markeredgecolor='k',label='Gradient')
plt.title("Minimisation de $f(x)=sin(x) + sin(10/3x)$",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("f(x)",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("Minimisation1D.svg",dpi=150)
plt.show()