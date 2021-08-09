import numpy as np
import numpy.random as rd


def uniform_init_population_lhs(shape):
    popsize = shape[0]
    ndof = shape[1]
    dsize  = 1.0/popsize

    sample_array = dsize*rd.sample(size=shape) + np.linspace(0,1,num=popsize,endpoint=False)[:,np.newaxis]
    init_pop = np.zeros(shape)
    for dim in range(ndof) : 
        rdm_order = rd.permutation(popsize)
        init_pop[:,dim] = sample_array[rdm_order,dim]

    return init_pop


def uniform_init_population_random(shape):
    return rd.sample(shape)




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