import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from _numerical_deriv import *


# f = lambda x : (x[0]-x[1])**2 + x[0]**3
# gf = lambda x : np.array([  2*(x[0]-x[1]) + 3*x[0]**2 ,
#                             -2*(x[0]-x[1])]).T

p = 10
f = lambda x : (x[0]-1)**2 + p*(x[0]**2-x[1])**2
gf = lambda x : np.array([2*(x[0]-1) + 2*p*(x[0]**2-x[1])*2*x[0],
                                -2*p*(x[0]**2-x[1])]).T



fpenal = lambda x : (f(x),None)
gf_approx = approxGradient(fpenal,2)

n = 150
x = np.linspace(-2,2,n)
y = np.linspace(-2,2,n)
X,Y = np.meshgrid(x,y)
z = np.zeros((2,n**2))
z[0] = X.flatten()
z[1] = Y.flatten()

fz = f(z)
gfz = gf(z)
gfz_approx = np.zeros_like(gfz)

for i,zi in enumerate(z.T) :
    gfz_approx[i] = gf_approx(zi)


print(np.isclose(gfz_approx,gfz).sum()/(2*n**2))



figContour = plt.figure("Contour")
contour = plt.contour(X,Y,fz.reshape((n,n)),
                    levels=np.linspace(fz.min(),fz.max(),100))
plt.grid(True)
plt.title("F(x,y)")


figGradx = plt.figure("Gradx")
plt.subplot(121)
contour = plt.contour(X,Y,gfz[:,0].reshape((n,n)),
                    levels=np.linspace(gfz[:,0].min(),gfz[:,0].max(),50))
plt.grid(True)
plt.title("GradX(x,y)")
plt.subplot(122)
contour = plt.contour(X,Y,gfz_approx[:,0].reshape((n,n)),
                    levels=np.linspace(gfz_approx[:,0].min(),gfz_approx[:,0].max(),50))
plt.grid(True)
plt.title("GradApproxX(x,y)")


figGradx = plt.figure("Grady")
plt.subplot(121)
contour = plt.contour(X,Y,gfz[:,1].reshape((n,n)),
                    levels=np.linspace(gfz[:,1].min(),gfz[:,1].max(),50))
plt.grid(True)
plt.title("GradY(x,y)")
plt.subplot(122)
contour = plt.contour(X,Y,gfz_approx[:,1].reshape((n,n)),
                    levels=np.linspace(gfz_approx[:,1].min(),gfz_approx[:,1].max(),50))
plt.grid(True)
plt.title("GradApproxY(x,y)")

plt.show()