import numpy as np
import matplotlib.pyplot as plt
plt.rc("font",family='Serif')
## PARETO
def fast_non_dominated_sort(x1,x2,selection_size=25,maxRank=5):
    size_pop = len(x1)
    index_R =  np.array(range(size_pop),dtype=int)
    Fi = []
    list_Sr = []
    list_nr = []
    rank_R = np.ones_like(index_R,dtype=int)*-1

    for r,(x1r,x2r) in enumerate(zip(x1,x2)) :

        dominated_test = (x1r<x1)&(x2r<x2)
        dominance_test = (x1r>x1)&(x2r>x2)


        nr = np.count_nonzero(dominated_test) #number of dominated solutions
        Sr = [q for q in range(size_pop) if dominance_test[q]]

        list_Sr.append(Sr)
        list_nr.append(nr)

        if nr == 0 :
            rank_R[r] = 1
            Fi.append(r)


    i = 1
    size = len(Fi)
    while len(Fi) > 0 and size<selection_size and max(rank_R)<maxRank :
        print(len(Fi))
        Q = []
        for r in Fi :
            Sr = list_Sr[r]
            for q in Sr :
                nq = list_nr[q]
                nq = nq - 1
                list_nr[q] = nq
                if nq == 0 :
                    rank_R[q] = i+1
                    Q.append(q)
        i += 1
        Fi = Q
        size += len(Fi)

    filtre = rank_R>0
    rank_R = rank_R[filtre]
    sorted_index = np.argsort(rank_R)
    y1,y2 = x1[filtre],x2[filtre]


    return rank_R

## PLOTS
files = ["GA-epsilonContraintes/kursaweGAEPSC.txt",
         "SCIPY-epsilonContraintes/kursaweScipy.txt",
         "NSGAII/kursaweNSGAII.txt"]

legende = ["GA : epsilon-Contrainte",
        "scipy : epslion-Contrainte",
        "NSGAII"]
algo = ["GAEPSC","SCIPY","NSGAII"]

marker = ["o",'o','o']
fig1 = plt.figure(figsize=(9,6))
data = []
for k,file in enumerate(files) :
    pareto = np.loadtxt(file)

    plt.plot(pareto[:,0],pareto[:,1],
                ls='',
                marker=marker[k],
                markeredgecolor='k',
                alpha=0.85,
                label=legende[k])

    data += [(algo[k],p[0],p[1]) for p in pareto]

plt.grid(True)
plt.xlabel("f1",fontsize=12)
plt.ylabel("f2",fontsize=12)
plt.title('Comparaison problème de KURSAWE',fontsize=14)
fig1.tight_layout()
plt.legend()
plt.savefig("compKursawe.svg",dpi=300)
plt.xlim(-19.15,-17.85)
plt.ylim(-4,0.25)
plt.savefig("compKursawe_part1.svg",dpi=300)


plt.autoscale()
plt.show()


## EXCEL
import pandas as pd
paretoDataFrame = pd.DataFrame(data,columns=["algorithme","f1","f2"])
y1 = paretoDataFrame["f1"].to_numpy()
y2 = paretoDataFrame["f2"].to_numpy()

rank = fast_non_dominated_sort(-y1,-y2,selection_size=200000,maxRank=200000)
paretoDataFrame["rangPareto"] = rank

paretoDataFrame.to_excel("comparaisonKursawe.xlsx",index=False)
