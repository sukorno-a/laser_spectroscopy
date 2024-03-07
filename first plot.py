# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:55:01 2024

@author: David
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pylab as pl

def gauss_function(x, a, x0, sigma,c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c

file = "P200P03.csv"
results = pd.read_csv(file, sep = ',', header=None)

x = results[0]
C1 = results[1]
C2 = results[2]


fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x,C2,color="Orange",label="Channel 1")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()


first_peak = results.loc[(results[0] > -0.018) & (results[0] < -0.014)]

popt1, pcov1 = sp.optimize.curve_fit(gauss_function, first_peak[0], first_peak[1], p0 = [-1, -0.0155, 0.01,-0.05],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=10000)
x_new=np.linspace(-0.018,-0.014,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt1[0],popt1[1],popt1[2],popt1[3]))
ax.plot(first_peak[0],first_peak[1],color="Orange",label="Channel 1")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

second_peak = results.loc[(results[0] > -0.012) & (results[0] < -0.007)]

popt2, pcov2 = sp.optimize.curve_fit(gauss_function, second_peak[0], second_peak[1], p0 = [-1, -0.01, 0.01,0])
x_new=np.linspace(-0.012,-0.007,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt2[0],popt2[1],popt2[2],popt2[3]))
ax.plot(second_peak[0],second_peak[1],color="Orange",label="Channel 1")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

third_peak = results.loc[(results[0] > 0.0025) & (results[0] < 0.008)]

popt3, pcov3 = sp.optimize.curve_fit(gauss_function, third_peak[0], third_peak[1], p0 = [-1, 0.0015, 0.01, 0])
x_new=np.linspace(0.0025,0.008,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt3[0],popt3[1],popt3[2],popt3[3]))
ax.plot(third_peak[0],third_peak[1],color="Orange",label="Channel 1")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

fourth_peak = results.loc[(results[0] > 0.015) & (results[0] < 0.0195)]

popt4, pcov4 = sp.optimize.curve_fit(gauss_function, fourth_peak[0], fourth_peak[1], p0 = [-1, 0.0175, 0.001, 0])
x_new=np.linspace(0.015,0.0195,100)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_new,gauss_function(x_new,popt4[0],popt4[1],popt4[2],popt4[3]))
ax.plot(fourth_peak[0],fourth_peak[1],color="Orange",label="Channel 1")
ax.axis(True)
sns.set(style="ticks")
ax.grid(True, which='both')
ax.margins(x=0)
plt.xlabel("Something (s)",size=20)
plt.ylabel("Something (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()