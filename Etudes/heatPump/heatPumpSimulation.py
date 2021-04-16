import numpy as np


## Donnees thermo

mair = 5.0
cpair = 1004
Taci = 273.15
Taro = 273.15 + 35
Taei = 273.15 + 24
Uconv = 25
Tapp_min = 4.0

## Donnees economiques
cexchanger = 100
ccomp = 220
celec = 0.06
cres = 85
cmaintenance = 0.1
cinstallation = 0.2
rInterest = 0.1
nbrHeures = 4000
nbrAnnees = 10

deltaTc = lambda Tc,Taci,Taco : ((Tc-Taci)-(Tc-Taco))/np.log((Tc-Taci)/(Tc-Taco))
deltaTe = lambda Te,Taei,Taeo : ((Taei-Te)-(Taeo-Te))/np.log((Taei-Te)/(Taeo-Te))

In = np.zeros(nbrAnnees)
Mn = np.zeros(nbrAnnees)
Rn = np.zeros(nbrAnnees)


class heatPump :

    def __init__(self):
        self.__Tc = 0.0
        self.__Te = 0.0
        self.__cout = 0.0
        self.__Wcomp = -1.0


    def simulateHeatPump(self,x):

        Taeo,Te,Tc = x
        cop = Tc/(Tc-Te)*0.31
        Qeva = mair*cpair*(Taei-Taeo)
        Wcomp = Qeva/cop
        Qcond = Qeva + Wcomp
        Taco = Qcond/(mair*cpair) + Taci
        Qelec = mair*cpair*(Taro-Taco)

        if ((Tc-Taci)/(Tc-Taco)) > 0.001 :
            dTc = deltaTc(Tc,Taci,Taco)
        else :
            dTc = ((Taei-Te)+(Taeo-Te))/2

        if ((Taei-Te)/(Taeo-Te)) > 0.001 :
            dTe = deltaTe(Te,Taei,Taeo)
        else :
            dTe = ((Tc-Taco)+(Tc-Taci))/2


        Acond = Qcond/(Uconv*dTc)
        Aeva = Qeva/(Uconv*dTe)

        if Qelec > 0.0 :
            ACelec = celec*Qelec/1000*nbrHeures + celec*Wcomp/1000*nbrHeures
        else :
            ACelec = celec*Wcomp/1000*nbrHeures
        ICeva = cexchanger*Aeva
        ICcond = cexchanger*Acond
        ICcomp = ccomp*Wcomp/1000
        ICres = cres*Qelec/1000

        In[0] = (ICeva + ICcond + ICcomp + ICres)*(1+cinstallation)
        Mn[:]  = In[0] * cmaintenance
        Rn[:] = ACelec

        cout = 0
        for k in range(nbrAnnees):
            cout += (In[k]+Mn[k]+Rn[k])/(1+rInterest)**(k+1)

        self.__Tc = Tc
        self.__Te = Te

        self.__Tapp_eva = Taeo-Te
        self.__Tapp_eva2 = Taei - Te
        self.__Tapp_cond = Tc - Taci
        self.__Tapp_cond2 = Tc - Taco
        self.__Qelec = Qelec

        self.__cout = cout
        self.__Wcomp = Wcomp
        self.__cop = cop

        dictSim = {}
        dictSim["cop"] = cop
        dictSim['Qcond'] = Qcond
        dictSim['Qeva'] = Qeva
        dictSim['Qelec'] = Qelec

        self.__dictSim = dictSim

        return None


    def cost(self,x):
        return self.__cout

    def contrainte1(self,x):
        return self.__Tapp_eva - Tapp_min

    def contrainte2(self,x):
        return self.__Tapp_eva2 - Tapp_min

    def contrainte3(self,x):
        return self.__Tapp_cond - Tapp_min

    def contrainte4(self,x):
        return self.__Tapp_cond2 - Tapp_min

    def contrainte5(self,x):
        return self.__cop - 1.0

    def contrainte6(self,x):
        return self.__Qelec


    def printDictSim(self,x) :
        self.simulateHeatPump(x)

        print("\n")
        for s in self.__dictSim :
            print(s," : ",self.__dictSim[s])


