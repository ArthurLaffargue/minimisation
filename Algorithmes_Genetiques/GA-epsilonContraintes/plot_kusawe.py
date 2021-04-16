import numpy as np
import matplotlib.pyplot as plt
import os
plt.rc("font",family='Serif')
listfile = [f for f in os.listdir() if f.startswith("kursawe")]

fig1= plt.figure(1)

all_point = []
k = 1
for f in listfile :
    front = np.loadtxt(f)
    all_point += [xi for xi in front]
    plt.plot(front[:,0],front[:,1],'+',label='Optimisation '+str(k))
    k+=1

plt.grid(True)
plt.xlabel("f1")
plt.ylabel("f2")
plt.title('Optimisation bi-critères')
fig1.tight_layout()
plt.legend()


all_point = np.array(all_point)
pareto = []
for X in all_point :


    if not( ((X[0]>all_point[:,0])&(X[1]>all_point[:,1])).any()) :
        pareto.append(X)

pareto = np.array(pareto)

np.savetxt("kursaweGAEPSC.txt",pareto)

fig1= plt.figure(2)

plt.plot(pareto[:,0],pareto[:,1],ls='',marker='o',markerfacecolor='r',markeredgecolor='k')
plt.grid(True)
plt.xlabel("f1",fontsize=12)
plt.ylabel("f2",fontsize=12)
plt.title('Optimisation bi-critères : Epsilon contrainte',fontsize=14)
fig1.tight_layout()
plt.savefig("Front_pb4.svg",dpi=300)

plt.show()