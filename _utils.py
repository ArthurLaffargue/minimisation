import numpy as np
import numpy.random as rd


def uniform_init_population_lhs(shape):
    popsize = shape[0]
    ndof = shape[1]
    dsize  = 1.0/popsize

    sample_array = dsize*rd.sample(size=shape) + np.linspace(0.,1.,num=popsize,endpoint=False)[:,np.newaxis]
    init_pop = np.zeros(shape)
    for dim in range(ndof) : 
        rdm_order = rd.permutation(popsize)
        init_pop[:,dim] = sample_array[rdm_order,dim]

    return init_pop


def uniform_init_population_random(shape):
    return rd.sample(shape)

def fast_non_dominated_sort(x1,x2,selection_size=None):
    size_pop = len(x1)
    if selection_size is None :
        selection_size = size_pop
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
    while size > 0 and size<selection_size:
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

    non_ranked_sol = rank_R < 1
    rank_R[non_ranked_sol] = max(rank_R)+1

    return rank_R

def crowning_distance(x1,x2,rank_R):
    """
    Compute the crowning distance between two objective x1,x2
    """
    size = len(x1)
    max_rank = max(rank_R)
    crowning_dist = np.zeros_like(x1,dtype=float)
    x1min,x1max = x1.min(),x1.max()
    x2min,x2max = x2.min(),x2.max()
    index_array = np.array(list(range(size)))

    for rank_k in range(1,max_rank+1):
        index_sol = index_array[rank_R == rank_k]
        x1_k = x1[index_sol]
        x2_k = x2[index_sol]
        distance_k = np.zeros_like(x1_k,dtype=float)

        sortindex = np.argsort(x1_k)
        x1_k = x1_k[sortindex]
        x2_k = x2_k[sortindex]
        index_sol = index_sol[sortindex]
        distance_k = distance_k[sortindex]
        distance_k[1:-1:] = (x1_k[2:] - x1_k[:-2])/(x1max-x1min)
        distance_k[0] = 1.0
        distance_k[-1] = 1.0

        sortindex = np.argsort(x2_k)
        x2_k = x2_k[sortindex]
        index_sol = index_sol[sortindex]
        distance_k = distance_k[sortindex]
        distance_k[1:-1:] += (x2_k[2:] - x2_k[:-2])/(x2max-x2min)
        distance_k[0] += 1.0
        distance_k[-1] += 1.0

        crowning_dist[index_sol] = distance_k

    dist_min,dist_max = crowning_dist.min(),crowning_dist.max()
    crowning_fitness =  (crowning_dist-dist_min)/(dist_max-dist_min)
    return crowning_fitness

if __name__ == "__main__": 
    import matplotlib.pyplot as plt

    shape = (100,2)
    x = uniform_init_population_lhs(shape)
    y = uniform_init_population_random(shape)

    plt.plot(x[:,0],x[:,1],'o')
    plt.plot(y[:,0],y[:,1],'s')
    plt.grid(True)
    plt.axis('equal')
    plt.show()