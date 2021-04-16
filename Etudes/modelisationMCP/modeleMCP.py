import numpy as np
from scipy.integrate import odeint


n = 2
u0 = 0.042 #m/s  inlet velocity
di = 25e-3 #m internal diameter
de = 0.127 #m external diameter
L = 2.9 #m Length
Tin = lambda t : (305)*np.ones_like(t) #K inlet temperature
T0 = [373.15,373.15] #K initial condition

k_pcm = 15 #W/(K.m) conductivity paraffin
rho_pcm = 1045 #kg/m3 Density paraffin
cp_p = 1.65e3 #J/(kg.K) Specific Heat

Lf =  (120+2*n)*1e3 #J/kg Latent Heat capacity
Ts = 351 #K solidus temperature
Tl = 358 #K liquidus temperature

rho_f = 991.2 #kg/m3
cp_f = 4186 #J/(K.kg)


"""
Compute other parameters
"""
e = (de-di)/2 #m thickness
ri,re = di/2,de/2
Slat = np.pi*di*L #m2 Lateral surface
Ssec = np.pi*di**2/4 #m2 Cross section
Spcm  = np.pi/4*(de**2-di**2)
Vf = Ssec*L #m3 Volume of water
Vpcm = L*Spcm #m3 Volume of pcm
G = Ssec*u0*rho_f #kg/s Mass flow rate
DTsl = (Tl-Ts)/2
Tm = (Tl+Ts)/2
cp_pcm = lambda T : cp_p + Lf/(DTsl*np.sqrt(np.pi))*np.exp(-(T-Tm)**2/DTsl**2)
Rcond = np.log(re/(ri+e/2))/(2*np.pi*L*k_pcm) #K/W  Thermal radial resistance


"""
Données issue de datapcm.txt
"""
DATA = np.loadtxt("simulationMCP.txt")
t = DATA[:,0]
Tinlet = DATA[:,1]
Toutlet = DATA[:,2]
phiWater = DATA[:,3]
nt = len(t)


def simulationMCP_0D(h):

    C0 = 1/(Rcond+1/(h*Slat))#W/K Thermal radial conductance
    A = lambda T : np.array([[  -u0/L-C0/(rho_f*cp_f*Vf)    ,  C0/(rho_f*cp_f*Vf) ],
                            [ C0/(Vpcm*rho_pcm*cp_pcm(T[1]) ) ,-C0/(Vpcm*rho_pcm*cp_pcm(T[1]) ) ]])
    B = lambda T,t :  np.array([u0/L*Tin(t),0])

    Eq = lambda T,t : A(T).dot(T)+B(T,t)

    Y = odeint(Eq,T0,t)

    Tout = Y[:,0]
    hFwater = G*cp_f*(Tout-Tin(t))
    res = np.sum((Tout-Toutlet)**2)/nt
    # res = np.sqrt(np.sum((phiWater-hFwater)**2))

    return res

class simulationMCP_1D :

    def __init__(self,size=25):
        self.__size = size
        dxi = L/size
        self.__dSi = Slat/size
        dVpcm = Vpcm/size
        dVf = Vf/size
        self.__Rcond = np.log(re/(ri+e/2))/(2*np.pi*dxi*k_pcm)
        Ki_pcm = Spcm*k_pcm/dxi

        self.__Kmat_pcm = Ki_pcm*( np.eye(size,k=1)+np.eye(size,k=-1)-2*np.eye(size) )
        self.__Kmat_pcm[0,0] = -Ki_pcm
        self.__Kmat_pcm[-1,-1] = -Ki_pcm

        self.__Kmat_f = G*cp_f*(np.eye(size,k=-1) - np.eye(size))


        self.__invCmat_pcm = lambda Tpcm : np.diag( 1/(dVpcm*rho_pcm*cp_pcm(Tpcm)) )
        self.__invCmat_f = np.eye(size)/(dVf*rho_f*cp_f)

        self.__dTdt = np.zeros(size*2)

        self.__Bvecf0 = np.zeros(size)
        self.__Bvecf0[0] = 1
        self.__Bvecf = lambda t : self.__Bvecf0*G*cp_f*Tin(t)

        self.__Tinit = np.ones(2*size)
        self.__Tinit[:size],self.__Tinit[size:] = T0



    def solveEquation(self,h) :
        C0 = 1/(self.__Rcond+1/(h*self.__dSi))
        self.__C0 = C0

        Y = odeint(self.equation,self.__Tinit,t)

        self.Twater = Y[:,self.__size:]
        self.Tpcm = Y[:,:self.__size]
        Tout = self.Twater[:,-1]


        hFwater = G*cp_f*(Tout-Tin(t))
        self.__res = np.sum((Tout-Toutlet)**2)/nt

        # res = np.sqrt(np.sum((phiWater-hFwater)**2))

    def sqrtError(self,h) :
        return self.__res





    def equation(self,T,t):
        Tmcp = T[:self.__size]
        Tf = T[self.__size:]

        dTmcpdt = self.__invCmat_pcm(Tmcp).dot( self.__Kmat_pcm.dot(Tmcp) +  self.__C0*(Tf-Tmcp)  )
        dTfdt = self.__invCmat_f.dot( self.__Kmat_f.dot(Tf) +  self.__C0*(Tmcp-Tf) + self.__Bvecf(t)  )

        self.__dTdt[:self.__size] = dTmcpdt
        self.__dTdt[self.__size:] = dTfdt

        return self.__dTdt


if __name__ == '__main__' :


    import matplotlib.pyplot as plt
    plt.rc('font',family='Serif')

    simPCM1D = simulationMCP_1D(15)
    simPCM1D.solveEquation(155.86)


    Tout = simPCM1D.Twater[:,-1]
    Tmcp = simPCM1D.Tpcm

    plt.figure(figsize=(7,4.5))

    plt.plot(t/3600,Toutlet,'b-',lw=2,label="2D CFD")
    plt.plot(t/3600,Tout,'r-',lw=2,label="1D Python")

    plt.grid(True)
    plt.xlabel('Temps (h)',fontsize=12)
    plt.ylabel('Température (K)',fontsize=12)
    plt.title("Comparaison simulation 1D et 2D",fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig("ModélisationMCP.svg",dpi=150)

    plt.show()






