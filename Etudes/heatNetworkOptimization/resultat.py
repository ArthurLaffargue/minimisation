import numpy as np
import matplotlib.pyplot as plt
import os
plt.rc('font',family='Serif')

listPareto = os.listdir("paretoFront/")

fig1 = plt.figure(1) #Pareto
fig2 = plt.figure(2) #Constraints violation
fig3 = plt.figure(3) #Iteration & funcCalls

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax31 = fig3.add_subplot(211)
ax32 = fig3.add_subplot(212)

for file in listPareto[::-1] :

    data = np.loadtxt("paretoFront/"+file)
    method = file.split(".")[0]
    energyFactor = data[:,0]
    economicFactor = data[:,1]
    constrViol = data[:,2]
    itercount = data[:,3]
    funcalls = data[:,4]

    ax1.plot(economicFactor*100,energyFactor*100,"o",label=method,markeredgecolor='k')
    ax2.plot(constrViol,"o",label=method)
    ax31.plot(itercount,"s",label=method)
    ax32.plot(funcalls,"D",label=method)

    print("\n")
    print("-"*20)
    print(method)
    print("funcalls : ",np.sum(funcalls))
    print("energyMinMax : ",energyFactor.min()*100,energyFactor.max()*100)
    print("economicMinMax : ",economicFactor.min()*100,economicFactor.max()*100)

ax1.legend()
ax2.legend()
ax31.legend()
ax32.legend()

ax1.grid()
ax2.grid()
ax31.grid()
ax32.grid()

ax1.set_xlabel("Investissement [min-max]%")
ax2.set_xlabel("Résolution n°")
ax31.set_xlabel("Résolution n°")
ax32.set_xlabel("Résolution n°")

ax1.set_ylabel("Performances [min-max]%")
ax2.set_ylabel("Violation des contrainte (norme)")
ax31.set_ylabel("Nombre d'itération")
ax32.set_ylabel("Nombre d'évaluation de la fonction")

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()


fig1.savefig("frontsPareto_network.svg",dpi=150)

plt.show()


ax1.set_xlim(20,40)
ax1.set_ylim(0,20)
fig1.tight_layout()
fig1.savefig("frontsPareto_zoom.svg",dpi=150)


ax1.set_xlim(0,100)
ax1.set_ylim(0,100)
fig1.tight_layout()