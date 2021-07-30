import numpy as np
import matplotlib.pyplot as plt
import sys
import time
plt.rc('font',family='Serif')

sys.path.append("..")
from minimization import minimize_scalar

# FONCTION ET MINIMISATION
xmin,xmax = 2.7,7.5
xinit = (xmax+xmin)/2


f = lambda x : np.sin(x)*x + np.sin(10/3*x)
df = lambda x : np.cos(x)*x + np.sin(x) + 10/3*np.cos(10/3*x)
x = np.linspace(xmin,xmax,250)
y = f(x)

start = time.time()
dictgold = minimize_scalar(f,xmin,xmax,returnDict=True,method="goldenSearch")
inter = time.time()
dictgrad = minimize_scalar(f,xmin,xmax,xinit=xinit,returnDict=True,gf=df,method="scalarGradient")
end = time.time()

xgold = dictgold["x"]
xgrad = dictgrad["x"]

fgold = dictgold['fmin']
fgrad = dictgrad['fmin']



print(f"Golden-Search {inter-start} s")
for s in dictgold :
    print(s," : ",dictgold[s])
print(f"\nGradient {end-inter} s")
for s in dictgrad :
    print(s," : ",dictgrad[s])

plt.figure(figsize=(9,3.5))

plt.plot(x,y,color='k',label="$f(x)$")
plt.plot(xgold,f(xgold),ls='',marker='D',markeredgecolor='k',label='Golden-Search')
plt.plot(xgrad,f(xgrad),ls='',marker='o',markeredgecolor='k',label='Gradient')
plt.title("Minimisation scalaire",fontsize=14)
plt.xlabel("x",fontsize=12)
plt.ylabel("f(x)",fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("Minimisation1D.svg",dpi=150)
plt.show()