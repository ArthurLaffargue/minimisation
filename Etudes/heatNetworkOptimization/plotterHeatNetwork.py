import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
plt.rc('font',family='Serif')


def plotNetworkTop(branchTopology,
                   nodesTopology,
                   fig=None,
                   nodeLabel=True,
                   title="Réseau hydraulique",
                   dhx=0.0,dhy=0.0):


    if fig is None :
        fig = plt.figure("NetworkTop")

    networkCode = branchTopology["network"].to_numpy(dtype=int)
    branch = branchTopology[["node_i","node_j"]].to_numpy(dtype=int)
    nodePos = nodesTopology[["posX","posY"]].to_numpy(dtype=float)
    filtreNodePlant = nodesTopology["nodeType"].str.startswith("plant")
    nodeType = nodesTopology["nodeType"].to_numpy()
    nodePlant = nodesTopology[filtreNodePlant]["nodeNumber"].to_numpy(dtype=int)
    plantNumber = nodesTopology[filtreNodePlant]["nodeType"].to_numpy(dtype=str)
    plantConnexion = branchTopology["plantConnexion"].to_numpy(dtype=int)

    dhx = dhx*(nodePos[:,0].max()-nodePos[:,0].min())
    dhy = dhy*(nodePos[:,1].max()-nodePos[:,1].min())


    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.axis("off")

    for k,pos in enumerate(nodePos):
        if nodeLabel :
            ax.text(pos[0]+dhx,pos[1]+dhy,str(k))
        if nodeType[k].startswith("plant") :
            ax.plot(pos[0],pos[1],marker="D",
                                    markeredgecolor='k',
                                    markerfacecolor='k',
                                    ls = '')
        else :
            ax.plot(pos[0],pos[1],marker="o",
                                    markeredgecolor='k',
                                    markerfacecolor='none',
                                    ls = '')

    lines = []
    for connexion in np.unique(plantConnexion):
        filterConnexion = connexion==plantConnexion
        if connexion == 0 :
            label = 'reseau secondaire'
        else :
            label = "Réseau n°"+str(connexion)
        for k,bk in enumerate(branch[filterConnexion]):
            if k == 0 :
                l = plt.plot(nodePos[bk,0],nodePos[bk,1],label=label)
                color = l[0].get_color()
            else :
                plt.plot(nodePos[bk,0],nodePos[bk,1],color=color)


    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    return fig




def plotBranchField(branchTopology,nodesTopology,Ufield,
                            fig=None,
                            title='Ufield',
                            nodeLabel=False,
                            cbarLabel='values') :

    if fig is None :
        fig = plt.figure(title)

    branch = branchTopology[["node_i","node_j"]].to_numpy(dtype=int)
    pos = nodesTopology[["posX","posY"]].to_numpy(dtype=float)
    nodeType = nodesTopology["nodeType"].to_numpy()

    norm = plt.Normalize(Ufield.min(), Ufield.max())

    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    branchPos = 1/2*(pos[branch[:,0]] + pos[branch[:,1]])


    cm =  plt.cm.get_cmap('jet')
    axUfield = ax.scatter(branchPos[:,0],branchPos[:,1],c = Ufield,alpha = 0.75, s=0, cmap=cm)



    for k,posk in enumerate(pos):
        if nodeLabel :
            plt.text(posk[0],posk[1],str(k))
        if nodeType[k].startswith("plant") :
            plt.plot(posk[0],posk[1],marker="D",
                                    markeredgecolor='k',
                                    markerfacecolor='none',
                                    ls = '')
        else :
            plt.plot(posk[0],posk[1],marker="o",
                                    markeredgecolor='k',
                                    markerfacecolor='none',
                                    ls = '')




    for k,(i,j) in enumerate(branch) :
        (x1,x2),(y1,y2) = (pos[i,0],pos[j,0]),(pos[i,1],pos[j,1])
        Uij = Ufield[k]


        x = np.array([x1,x2])
        y = np.array([y1,y2])
        u = np.array([Uij,Uij])

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='jet', norm=norm,alpha = 0.85)
        lc.set_array(u)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

        ax.arrow(x1,y1,(x2-x1)/2,(y2-y1)/2,color = 'k',head_width=0.085,width = 0,alpha = 1)


    plt.axis("equal")
    cbar = fig.colorbar(axUfield,ax=ax,label = cbarLabel)