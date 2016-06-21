# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 06:12:40 2016

@author: maksim
"""
import scipy.optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

don = pd.read_csv('d.csv')

Qmh = don.values[0,1]
lambe = don.values[2,1]
lambh = don.values[3,1]
Cpe = don.values[6,1]

alpha0 = float(don.values[11,26])
k = float(don.values[12,26])
a = float(don.values[13,26])
crit = float(don.values[14,26])

V = np.array([float(don.values[i,10]) for i in range(3,9)])
t = np.array([float(don.values[i,11]) for i in range(3,9)])
Tee = np.array([float(don.values[i,13]) for i in range(3,9)])
Tse = np.array([float(don.values[i,14]) for i in range(3,9)])
Teh = np.array([float(don.values[i,15]) for i in range(3,9)])
Tsh = np.array([float(don.values[i,16]) for i in range(3,9)])
Tsh[0]=51
Cph = don.values[7,1]

Qme = V/t
Wh = Cph*Qmh
We = Cpe*Qme
dT1 = Tsh-Tee
dT2 = Teh-Tse
dTl = (dT1-dT2)/np.log(dT1/dT2)
kF1 = np.log(dT2/dT1)*(1/Wh-1/We)**(-1)
kF=We*(Tse-Tee)/dTl
kFt = alpha0/(1+k/(Qme**a))
plt.plot(Qme[0:],kF1[0:],'m-o',label = 'experience')
'''plt.plot(Qme[0:],kF[0:],'b-o',label = 'experience')'''
plt.plot(Qme[0:],kFt[0:],'c-^', label = 'theorie')
plt.grid(True)
plt.legend(loc = 'lower right')
plt.xlabel('Qm (kg/s)')
plt.ylabel('AlphaG (W/C)')
plt.savefig('AlphaG1.png')
plt.clf()

'''plt.plot(Qme[1:],kF1[1:],'b-o',label = 'traitement 1')
plt.plot(Qme[1:],kF2[1:],'g-*', label = 'traitement 2')
plt.plot(Qme[1:],kFt[1:],'c-^', label = 'theorie')
plt.grid(True)
plt.legend(loc = 'lower right')
plt.xlabel('Qm (kg/s)')
plt.ylabel('AlphaG (W/C)')
plt.savefig('AlphaG2.png')
plt.clf()'''

Wmin = np.zeros(6)
for i in range(0,6):
    Wmin[i] = min(Wh,We[i])
Eps1 = (Teh-Tsh)*Wh/((Teh-Tee)*Wmin)
NUT = kF/Wmin
Wmax = np.zeros(6)
for i in range(0,6):
    Wmax[i] = max(Wh,We[i])
R = Wmin/Wmax
Eps = lambda NUT,R : (1-np.exp(-NUT*(1-R)))/(1-R*np.exp(-NUT*(1-R)))
NUT[2], NUT[5], NUT[3], NUT[1], NUT[4] = NUT[1], NUT[4], NUT[5], NUT[2], NUT[3]
R[2], R[5], R[3], R[1], R[4] = R[1], R[4], R[5], R[2], R[3]
Eps1[2], Eps1[5], Eps1[3], Eps1[1], Eps1[4] = Eps1[1], Eps1[4], Eps1[5], Eps1[2], Eps1[3]
plt.plot(NUT,Eps(NUT,R),'r-^')
plt.grid(True)
plt.xlabel('NUT')
plt.ylabel('Epsilon')
plt.savefig('Epsilon.png')
plt.clf()
'''plt.plot(NUT,Eps1,'r-^')'''

plt.plot(Qme[0:],kF1[0:],'m-o',label = 'experience')
plt.plot(Qme[0:],kF[0:],'b-o',label = 'experience')
plt.plot(Qme[0:],kFt[0:],'c-^', label = 'theorie')
'''plt.clf()
plt.plot(Qme[0:],dTl*kF1[0:],'m-o',label = 'experience')
plt.plot(Qme[0:],dTl*kF[0:],'b-o',label = 'experience')
plt.plot(Qme[0:],dTl*kFt[0:],'c-^', label = 'theorie')'''