import numpy as np
import matplotlib.pyplot as plt
def Heaviside(x):
    if x<0:
        return 0
    else:
        return 1


delP = np.array([1.4,1.3,1.2,1.1,1.0,0.95,0.90,0.85,0.80,0.7,0.6,0.5,0.4,0.3,0.2])*10**3
Te = np.array([28.5,30,30,30.5,30.5,31,31,31,31,31,31.5,32,32,32,32])+273
Ts = np.array([36.6,38.0,38.4,39.8,40.2,40.8,41.2,41.8,42.6,44.0,46.6,48.2,51.4,56.6,64.6])+273
U = 200
I = 5
T = np.array([[70.5,91.4,79.3,77.4,75.6,76.4,68.9,66.3,69.7,65.1,44.7,28.4,28.1,77.9,22.7],[73.3,94.6,82.2,80.6,78.2,78.8,71.1,68.2,71.8,66.7,46.0,28.9,28.6,80.6,23.8],[75.6,97.2,84.8,82.8,80.4,81.3,73.0,69.9,73.1,68.1,47.0,29.5,29.0,82.5,24.7],[78.3,100,87.4,85.5,83,83.4,75.1,71.8,75.2,69.6,48.1,29.4,29.2,84.9,25.1],[80.5,102.3,89.6,87.4,85.2,85.4,77.1,73.7,76.2,70.7,48.8,29.8,29.3,29.3,86.7,25.7],[81.5,103.3,90.8,88.7,85.9,86.2,77.8,74.3,77.3,71.3,49.3,29.9,29.6,87.8,25.8],[82.7,104.5,92.0,89.8,87.0,87.3,78.6,74.8,77.6,71.8,49.7,29.9,29.5,88.7,25.8],[83.7,105.2,92.9,90.8,87.9,88.0,79.6,75.7,78.4,72.5,50.2,29.9,29.8,90,26.1],[84.9,106.8,94.6,92.4,89.6,89.6,81.1,77.1,79.7,73.6,50.9,30.1,29.8,91.8,26.2],[89.2,111.1,98.9,96.4,93.2,92.8,84.1,79.5,81.9,75.3,52.1,30.3,29.9,95.1,26.4],[93.9,116.4,104.1,101.3,97.9,97.3,88.1,83.1,85.3,78.1,53.9,30.8,30.2,100,27.2],[97.9,120.3,108.6,105.9,102.3,101.3,91.9,86.9,88.8,80.8,55.6,30.8,30.6,105.1,27.6],[103.3,126.6,115.1,112.2,108.3,107.4,97.9,91.9,93.5,84.8,58.2,30.9,30.3,112.1,27.5],[113.4,137.4,126.0,122.7,118.3,116.3,105.9,98.6,99.4,90.4,62.0,31.6,30.8,122.6,28.6],[126.9,151.8,140.9,137.0,131.6,128.9,117.8,109.2,108.9,97.3,66.4,31.4,30.9,135.9,28.7]])
x = np.array([0,0.115,0.240,0.515,0.79,1.065,1.340,1.615,1.740,1.865,1.990,2.490,2.920])
d = 0.03175
L = 3
Lch = 2
Lcal = 2.4
dd = 0.0413
def rho(T):
    t = T-273
    if t<25:
        return 1.205
    elif t<35:
        return 1.165
    elif t<45:
        return 1.128
    elif t<55:
        return 1.093
    elif t<65:
        return 1.060
    elif t<75:
        return 1.029
    elif t<85:
        return 1.000
    elif t<95:
        return 0.972
    elif t<110:
        return 0.946
    else:
        return 0.898
x1 = np.array([3-x[12-i] for i in range(0,13)])
lamb = 2.76*10**(-2)
lambc = 0.037
def mu(T):
    t = T-273
    if t<25:
        return 18.1*10**(-6)
    elif t<35:
        return 18.6*10**(-6)
    elif t<45:
        return 19.1*10**(-6)
    elif t<55:
        return 19.6*10**(-6)
    elif t<65:
        return 20.1*10**(-6)
    elif t<75:
        return 20.6*10**(-6)
    elif t<85:
        return 21.1*10**(-6)
    elif t<95:
        return 21.5*10**(-6)
    elif t<110:
        return 21.9*10**(-6)
    else:
        return 22.8*10**(-6)
        
Sc = Lch*(d+0.001625+0.04)*np.pi
G = np.zeros(15)
u = np.zeros(15)

for i in range(0,15):
    G[i] = 0.654*dd**2*np.pi/4*(2*rho(Te[i])*delP[i])**(0.5)
Ta = lambda Te, Ts, x: Te+ 0.5 * (np.sign(x-1) + 1)*(Ts-Te)*(x-1)/(L-1)
T1 = np.array([T[0][12-i] for i in range(0,13)])+273
T2 = np.array([T[1][12-i] for i in range(0,13)])+273
T3 = np.array([T[2][12-i] for i in range(0,13)])+273
T4 = np.array([T[3][12-i] for i in range(0,13)])+273
T5 = np.array([T[4][12-i] for i in range(0,13)])+273
T6 = np.array([T[5][12-i] for i in range(0,13)])+273
T7 = np.array([T[6][12-i] for i in range(0,13)])+273
T8 = np.array([T[7][12-i] for i in range(0,13)])+273
T9 = np.array([T[8][12-i] for i in range(0,13)])+273
T10 = np.array([T[9][12-i] for i in range(0,13)])+273
T11 = np.array([T[10][12-i] for i in range(0,13)])+273
T12 = np.array([T[11][12-i] for i in range(0,13)])+273
T13 = np.array([T[12][12-i] for i in range(0,13)])+273
T14 = np.array([T[13][12-i] for i in range(0,13)])+273
T15 = np.array([T[14][12-i] for i in range(0,13)])+273

plt.clf()
plt.plot(x1,Ta(Te[1],Ts[1],x1),'m-',label='fluide')
plt.plot(x1,T1,'c-o',label='paroi')
plt.legend(loc='upper left')
plt.ylabel('T (K)')
plt.xlabel('x (m)')
plt.grid(True)
plt.savefig('T.png')



def moy(N):
    s = 0
    for i in range(0,len(N)):
        s += N[i]
    return s/len(N)
'''
alpha1 = (U*I-Sc*lambc*moy(T1[3:]-T[0][14]))/(Lch*np.pi*d*moy(T1[3:]-Ta(Te[0],Ts[0],x1[3:])))
alpha2 = (U*I-Sc*lambc*moy(T2[3:]-T[1][14]))/(Lch*np.pi*d*moy(T2[3:]-Ta(Te[1],Ts[1],x1[3:])))
alpha3 = (U*I-Sc*lambc*moy(T3[3:]-T[2][14]))/(Lch*np.pi*d*moy(T3[3:]-Ta(Te[2],Ts[2],x1[3:])))
alpha4 = (U*I-Sc*lambc*moy(T4[3:]-T[3][14]))/(Lch*np.pi*d*moy(T4[3:]-Ta(Te[3],Ts[3],x1[3:])))
alpha5 = (U*I-Sc*lambc*moy(T5[3:]-T[4][14]))/(Lch*np.pi*d*moy(T5[3:]-Ta(Te[4],Ts[4],x1[3:])))
alpha6 = (U*I-Sc*lambc*moy(T6[3:]-T[5][14]))/(Lch*np.pi*d*moy(T6[3:]-Ta(Te[5],Ts[5],x1[3:])))
alpha7 = (U*I-Sc*lambc*moy(T7[3:]-T[6][14]))/(Lch*np.pi*d*moy(T7[3:]-Ta(Te[6],Ts[6],x1[3:])))
alpha8 = (U*I-Sc*lambc*moy(T8[3:]-T[7][14]))/(Lch*np.pi*d*moy(T8[3:]-Ta(Te[7],Ts[7],x1[3:])))
alpha9 = (U*I-Sc*lambc*moy(T9[3:]-T[8][14]))/(Lch*np.pi*d*moy(T9[3:]-Ta(Te[8],Ts[8],x1[3:])))
alpha10 = (U*I-Sc*lambc*moy(T10[3:]-T[9][14]))/(Lch*np.pi*d*moy(T10[3:]-Ta(Te[9],Ts[9],x1[3:])))
alpha11 = (U*I-Sc*lambc*moy(T11[3:]-T[10][14]))/(Lch*np.pi*d*moy(T11[3:]-Ta(Te[10],Ts[10],x1[3:])))
alpha12 = (U*I-Sc*lambc*moy(T12[3:]-T[11][14]))/(Lch*np.pi*d*moy(T12[3:]-Ta(Te[11],Ts[11],x1[3:])))
alpha13 = (U*I-Sc*lambc*moy(T13[3:]-T[12][14]))/(Lch*np.pi*d*moy(T13[3:]-Ta(Te[12],Ts[12],x1[3:])))
alpha14 = (U*I-Sc*lambc*moy(T14[3:]-T[13][14]))/(Lch*np.pi*d*moy(T14[3:]-Ta(Te[13],Ts[13],x1[3:])))
alpha15 = (U*I-Sc*lambc*moy(T15[3:]-T[14][14]))/(Lch*np.pi*d*moy(T15[3:]-Ta(Te[14],Ts[14],x1[3:]))) 
'''
alpha1 = (U*I-Sc*lambc*moy(T1[3:]-T[0][14]))/(Lch*np.pi*d*moy(T1-Ta(Te[0],Ts[0],x1)))
alpha2 = (U*I-Sc*lambc*moy(T2[3:]-T[1][14]))/(Lch*np.pi*d*moy(T2-Ta(Te[1],Ts[1],x1)))
alpha3 = (U*I-Sc*lambc*moy(T3[3:]-T[2][14]))/(Lch*np.pi*d*moy(T3-Ta(Te[2],Ts[2],x1)))
alpha4 = (U*I-Sc*lambc*moy(T4[3:]-T[3][14]))/(Lch*np.pi*d*moy(T4-Ta(Te[3],Ts[3],x1)))
alpha5 = (U*I-Sc*lambc*moy(T5[3:]-T[4][14]))/(Lch*np.pi*d*moy(T5-Ta(Te[4],Ts[4],x1)))
alpha6 = (U*I-Sc*lambc*moy(T6[3:]-T[5][14]))/(Lch*np.pi*d*moy(T6-Ta(Te[5],Ts[5],x1)))
alpha7 = (U*I-Sc*lambc*moy(T7[3:]-T[6][14]))/(Lch*np.pi*d*moy(T7-Ta(Te[6],Ts[6],x1)))
alpha8 = (U*I-Sc*lambc*moy(T8[3:]-T[7][14]))/(Lch*np.pi*d*moy(T8-Ta(Te[7],Ts[7],x1)))
alpha9 = (U*I-Sc*lambc*moy(T9[3:]-T[8][14]))/(Lch*np.pi*d*moy(T9-Ta(Te[8],Ts[8],x1)))
alpha10 = (U*I-Sc*lambc*moy(T10[3:]-T[9][14]))/(Lch*np.pi*d*moy(T10-Ta(Te[9],Ts[9],x1)))
alpha11 = (U*I-Sc*lambc*moy(T11[3:]-T[10][14]))/(Lch*np.pi*d*moy(T11-Ta(Te[10],Ts[10],x1)))
alpha12 = (U*I-Sc*lambc*moy(T12[3:]-T[11][14]))/(Lch*np.pi*d*moy(T12-Ta(Te[11],Ts[11],x1)))
alpha13 = (U*I-Sc*lambc*moy(T13[3:]-T[12][14]))/(Lch*np.pi*d*moy(T13-Ta(Te[12],Ts[12],x1)))
alpha14 = (U*I-Sc*lambc*moy(T14[3:]-T[13][14]))/(Lch*np.pi*d*moy(T14-Ta(Te[13],Ts[13],x1)))
alpha15 = (U*I-Sc*lambc*moy(T15[3:]-T[14][14]))/(Lch*np.pi*d*moy(T15-Ta(Te[14],Ts[14],x1))) 
alpha = np.array([alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15])

alpha1 = (U*I)/(Lch*np.pi*d*moy(T1-Ta(Te[0],Ts[0],x1)))
alpha2 = (U*I)/(Lch*np.pi*d*moy(T2-Ta(Te[1],Ts[1],x1)))
alpha3 = (U*I)/(Lch*np.pi*d*moy(T3-Ta(Te[2],Ts[2],x1)))
alpha4 = (U*I)/(Lch*np.pi*d*moy(T4-Ta(Te[3],Ts[3],x1)))
alpha5 = (U*I)/(Lch*np.pi*d*moy(T5-Ta(Te[4],Ts[4],x1)))
alpha6 = (U*I)/(Lch*np.pi*d*moy(T6-Ta(Te[5],Ts[5],x1)))
alpha7 = (U*I)/(Lch*np.pi*d*moy(T7-Ta(Te[6],Ts[6],x1)))
alpha8 = (U*I)/(Lch*np.pi*d*moy(T8-Ta(Te[7],Ts[7],x1)))
alpha9 = (U*I)/(Lch*np.pi*d*moy(T9-Ta(Te[8],Ts[8],x1)))
alpha10 = (U*I)/(Lch*np.pi*d*moy(T10-Ta(Te[9],Ts[9],x1)))
alpha11 = (U*I)/(Lch*np.pi*d*moy(T11-Ta(Te[10],Ts[10],x1)))
alpha12 = (U*I)/(Lch*np.pi*d*moy(T12-Ta(Te[11],Ts[11],x1)))
alpha13 = (U*I)/(Lch*np.pi*d*moy(T13-Ta(Te[12],Ts[12],x1)))
alpha14 = (U*I)/(Lch*np.pi*d*moy(T14-Ta(Te[13],Ts[13],x1)))
alpha15 = (U*I)/(Lch*np.pi*d*moy(T15-Ta(Te[14],Ts[14],x1))) 
alphasp = np.array([alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,alpha7,alpha8,alpha9,alpha10,alpha11,alpha12,alpha13,alpha14,alpha15])

Nu = alpha*d/lamb
Nusp = alphasp*d/lamb
def Pr(T):
    t = T-273
    if t<25:
        return 0.703
    elif t<35:
        return 0.701
    elif t<45:
        return 0.699
    elif t<55:
        return 0.698
    elif t<65:
        return 0.696
    elif t<75:
        return 0.694
    elif t<85:
        return 0.692
    elif t<95:
        return 0.690
    elif t<110:
        return 0.688
    else:
        return 0.686


Tmoy = np.array([moy(T1[3:]),moy(T2[3:]),moy(T3[3:]),moy(T4[3:]),moy(T5[3:]),moy(T6[3:]),moy(T7[3:]),moy(T8[3:]),moy(T9[3:]),moy(T10[3:]),moy(T11[3:]),moy(T12[3:]),moy(T13[3:]),moy(T14[3:]),moy(T15[3:])])
Tamoy = np.array([moy(Ta(Te[0],Ts[0],x1[3:])),moy(Ta(Te[1],Ts[1],x1[3:])),moy(Ta(Te[2],Ts[2],x1[3:])),moy(Ta(Te[3],Ts[3],x1[3:])),moy(Ta(Te[4],Ts[4],x1[3:])),moy(Ta(Te[5],Ts[5],x1[3:])),moy(Ta(Te[6],Ts[6],x1[3:])),moy(Ta(Te[7],Ts[7],x1[3:])),moy(Ta(Te[8],Ts[8],x1[3:])),moy(Ta(Te[9],Ts[9],x1[3:])),moy(Ta(Te[10],Ts[10],x1[3:])),moy(Ta(Te[11],Ts[11],x1[3:])),moy(Ta(Te[12],Ts[12],x1[3:])),moy(Ta(Te[13],Ts[13],x1[3:])),moy(Ta(Te[14],Ts[14],x1[3:]))])
Nut = np.zeros(15)
Re = np.zeros(15)

for i in range(0,15):
    u[i] = G[i]*4/(np.pi*d**2*rho(Tamoy[i]))
for i in range(0,15):
    Re[i] = u[i]*d*rho(Tamoy[i])/mu(Tamoy[i])
for i in range(0,15):
    Nut[i] = 0.021*Re[i]**(0.8)*Pr(Tamoy[i])**(0.4)*(Pr(Tamoy[i])/Pr(Tmoy[i]))**(0.25)
'''
for i in range(0,15):
    Nut[i] = 0.023*Re[i]**(0.8)*Pr(Tamoy[i])**(0.4)
'''

plt.clf()
plt.plot(Re,Nu,'bo',label='Experience')
plt.plot(Re,Nut,'g-',label='Loi theorique')
plt.legend(loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Re')
plt.ylabel('Nu')
plt.grid(True)
plt.savefig('NuM.png')
plt.axis([30000,110000,80,200])
plt.savefig('NuMp.png')

for i in range(0,15):
    Nut[i] = 0.023*Re[i]**(0.8)*Pr(Tamoy[i])**(0.4)
plt.clf()
plt.plot(Re,Nu,'b-o',label='Experience')
plt.plot(Re,Nut,'g-',label='Loi theorique')

plt.legend(loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Re')
plt.ylabel('Nu')
plt.grid(True)
plt.savefig('NuC.png')
plt.axis([30000,110000,80,200])
plt.savefig('NuCp.png')