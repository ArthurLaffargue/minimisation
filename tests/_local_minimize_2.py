import numpy as np
from scipy.optimize import minimize
import sys
import time
import matplotlib.pyplot as plt
plt.rc('font',family='Serif')

sys.path.append("..")
from _minimize_gradient import *
from _minimize_BFGS import *

x0 = np.array([-1,1])
xmin = np.array([-1.2,-1.2])
xmax = np.array([1.2,1.2])

A = np.array([[2,-5],[3,2]])
b = np.array([-2.5,1.5])
c = 0.0

def func(x):
    return x.dot(A.dot(x)) + b.dot(x) + c


#------- SCIPY -------#

xscipy = [x0]
res = minimize(func, x0,method="CG",callback=xscipy.append)
xscipy = np.array(xscipy)
nfev = res.nfev + 3*res.njev


print("\nSCIPY : ")
print(res)
print("nfev : ",nfev)

#------- GRADIENT CONJUGUE -------#
start = time.time()
dictgrad = conjugateGradient(func,x0,xmin,xmax,
                            returnDict=True,
                            storeIterValues=True)

dictbfgs = BFGS(func,x0,xmin,xmax,
                            returnDict=True,
                            storeIterValues=True)
end = time.time()

xgrad = dictgrad["xHistory"]
fgrad = dictgrad['fmin']
xbfgs = dictbfgs["xHistory"]

for s in dictgrad :
    if not( s.endswith("History") ):
        print(s," : ",dictgrad[s])

print("\n")
for s in dictbfgs :
    if not( s.endswith("History") ):
        print(s," : ",dictbfgs[s])


# Représentation fonction
n = 150
x = np.linspace(xmin[0],xmax[0],n)
y = np.linspace(xmin[1],xmax[1],n)
X,Y = np.meshgrid(x,y)
z = np.zeros((2,n**2))
z[0] = X.flatten()
z[1] = Y.flatten()

fz = np.array([func(zi) for zi in z.T])


figContour = plt.figure("Contour")
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=np.linspace(fz.min(),fz.max(),100),cmap="jet")

plt.plot(xgrad[:,0],xgrad[:,1],'r-o')
plt.plot(xbfgs[:,0],xbfgs[:,1],'b-o')
# plt.plot(xscipy[:,0],xscipy[:,1],'b-o')
plt.grid(True)
plt.title("func(x,y)")


plt.show()






