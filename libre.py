# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:42:02 2015

@author: maksim
"""

'''
P = Pconv + Pray + Pcond
Pray = eps * sigma * (Tb^4 * Sb - Tp^4 * Sp)
Pcond = -k * grad(T) ~ k (Tb-Tair)*Sfil/Lfil 
P = UI
Nu = Pconv * Db/(kair*(Tb-Tp))
Ra = g*(Tb-Tp)*Db^3/(nu*alf*T)
alf = k/(rho*Cp)
P = rho*R*T 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy


U = 7.1
I = 0.71
P = U*I
Cp = 1006
kair = 0.025
Db = 0.00635
Lb = 0.1605
Dp = 0.46
Lp = 0.76
Lf = 0.30
Df = 0.002

Tb = np.array([138.58,138.58,138.58,137.8,136.76,136.75,135.46,132.86,128.96,125.32,120.12,115.44,110.24,108.16,89.96])+273
Tp = np.array([21.06,21.06,21.06,21.06,21.06,21.06,21.06,21.06,21.06,21.06,21.06,21.06,21.06,21.06,21.06])+273
Pres = np.array([14,35,60,82,130,210,440,880,1700,3300,6900,13000,24000,50000,100000])

Sb = np.pi*Db*Lb+2*np.pi*Db**2/4
Sp = np.pi*Dp*Lp+2*np.pi*Dp**2/4
Sf = np.pi*Df*Lf
sf = np.pi*Df**2/4

sigma = 5.67*10**(-8)

Pray = sigma*(Tb**4*Sb-Tp**4*Sb)
Pcond = 2*401*(Tb-Tp)*sf/Lf
Pconv = P - Pray - Pcond

Nu = Pconv*Db/(kair*Sb*(Tb-Tp))

mu = 17.2*10**(-6)
rho = Pres/(287*Tp)
nu = mu/rho
alf = kair/(rho*Cp)

Ra = 9.8*(Tb-Tp)*Db**3/(nu*alf*(Tp+Tb)/2)

plt.plot(Ra,Nu,'bo',label = 'experience')
plt.plot(Ra,0.59*Ra**(1/4),'b-',label = 'theorie')
plt.legend(loc='lower right')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Ra')
plt.ylabel('Nu')
plt.grid(True)

plt.savefig('NuRa.png')
'''
plt.clf()
plt.plot(Pres,Pconv)
#plt.yscale('log')
plt.xscale('log')
plt.xlabel('P (Pa)')
plt.ylabel('Phi (W)')
plt.grid(True)
plt.savefig('FluxPres.png')
'''