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
from matplotlib import rcParams
rcParams["font.family"] = 'Times New Roman'

def gauss_function(x, a, x0, sigma,c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c

def lorentzian(x,A,G,x0):
    return (A/np.pi) * ((G/2)/((x-x0)**2+(G/2)**2))

def six_lorentzian(x,A1,G1,x1,A2,G2,x2,A3,G3,x3,A4,G4,x4,A5,G5,x5,A6,G6,x6,c):
    return lorentzian(x,A1,G1,x1) + lorentzian(x,A2,G2,x2) + lorentzian(x,A3,G3,x3) + lorentzian(x,A4,G4,x4) + lorentzian(x,A5,G5,x5) + lorentzian(x,A6,G6,x6) + c

# # params = np.array([1, 0.5, 0])
# # x = np.linspace(-1,1,1000)
# # y = lorentzian(x, *params)
# # plt.plot(x,y)

# file = "P200P10.csv"
# results = pd.read_csv(file, sep = ',', header=None)

# results = results.loc[(results[0] > 0.068) & (results[0] < 0.11)]
# x = results[0]
# C1 = results[1] #Without block
# C2 = results[2] #With block
# difference = results[3] #without - with

# # fig, ax = plt.subplots(figsize=(10,6))
# # # ax.plot(x,C1,color="Orange",label="Channel 1")
# # # ax.plot(x,C2,color="Blue",label="Channel 2")
# # ax.plot(x,difference,color="Orange",label="Difference")
# # ax.axis(True)
# # sns.set(style="ticks")
# # ax.grid(True, which='both')
# # ax.margins(x=0)
# # plt.xlabel("Something (s)",size=20)
# # plt.ylabel("Something (V)",size=20)
# # plt.title("Initial Plot", size=24)
# # plt.legend(fancybox=True, shadow=True, prop={'size': 18})
# # plt.xticks(size=18,color='#4f4e4e')
# # plt.yticks(size=18,color='#4f4e4e')
# # sns.set(style='whitegrid')
# # plt.show()

# ##FIRST PEAK CORRESPONDS TO 87Rb F=2
# first_peak = results.loc[(results[0] > 0.075) & (results[0] < 0.079)]

# popt1, pcov1 = sp.optimize.curve_fit(gauss_function, first_peak[0], first_peak[2], p0 = [-1, 0.07775, 0.01,-0.5],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=10000)
# x_new=np.linspace(0.075,0.0785,100)

# params1,ub1,lb1 = np.array([0.04,0.0005,0.07568]),np.array([np.inf,0.002,0.0778]),np.array([0,0,0.0755])
# params2,ub2,lb2 = np.array([0.2,0.0005,0.0761]),np.array([np.inf,0.001,0.0762]),np.array([0,0,0.076])
# params3,ub3,lb3 = np.array([0.23,0.0005,0.0765]),np.array([np.inf,0.001,0.0766]),np.array([0,0,0.0764])
# params4,ub4,lb4 = np.array([0.8,0.0005,0.07675]),np.array([np.inf,0.001,0.0768]),np.array([0,0,0.0767])
# params5,ub5,lb5 = np.array([1.25,0.0005,0.07715]),np.array([np.inf,0.001,0.0773]),np.array([0,0,0.077])
# params6,ub6,lb6 = np.array([0.35,0.0004,0.077857]),np.array([np.inf,0.001,0.779]),np.array([0,0,0.0778])

# poptfit1, pcovfit1 = sp.optimize.curve_fit(six_lorentzian,first_peak[0],first_peak[3],p0=[*params1,*params2,*params3,*params4,*params5,*params6,0],bounds=((*lb1,*lb2,*lb3,*lb4,*lb5,*lb6,-np.inf), (*ub1,*ub2,*ub3,*ub4,*ub5,*ub6,np.inf)),maxfev=100000)
# fit1 = six_lorentzian(x_new, *poptfit1)


# fig, ax = plt.subplots(figsize=(10,6))
# # ax.plot(x_new,gauss_function(x_new,popt1[0],popt1[1],popt1[2],popt1[3]))
# #ax.plot(first_peak[0],first_peak[1],color="Orange",label="Channel 1")
# #ax.plot(first_peak[0],first_peak[2],color="Blue",label="Channel 2")
# ax.plot(x_new,fit1,color="Orange",label="Fitted")
# # ax.plot(first_peak[0],first_peak[2],color="Blue",label="Doppler Shifted")
# ax.plot(first_peak[0],first_peak[1],color="Blue",label="Dobler Free")
# ax.plot(first_peak[0],first_peak[3],color="Blue",label="Difference")
# rcParams["font.family"] = 'Times New Roman'
# ax.axis(True)
# sns.set(style="ticks")
# ax.grid(True, which='both')
# ax.margins(x=0)
# plt.xlabel("Something (s)",size=20)
# plt.ylabel("Something (V)",size=20)
# plt.title("Initial Plot", size=24)
# plt.legend(fancybox=True, shadow=True, prop={'size': 18})
# plt.xticks(size=18,color='#4f4e4e')
# plt.yticks(size=18,color='#4f4e4e')
# sns.set(style='whitegrid')
# plt.show()


# ##FIRST PEAK CORRESPONDS TO 85Rb F=3
# second_peak = results.loc[(results[0] > 0.0815) & (results[0] < 0.0835)]

# popt2, pcov2 = sp.optimize.curve_fit(gauss_function, second_peak[0], second_peak[1], p0 = [-1, -0.01, 0.01,0],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=100000)
# x_new=np.linspace(0.0815,0.0835,100)

# fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(x_new,gauss_function(x_new,popt2[0],popt2[1],popt2[2],popt2[3]))
# # ax.plot(second_peak[0],second_peak[1],color="Orange",label="Channel 1")
# # ax.plot(second_peak[0],second_peak[2],color="Blue",label="Channel 2")
# ax.plot(second_peak[0],second_peak[3],color="Blue",label="Difference")
# ax.axis(True)
# sns.set(style="ticks")
# ax.grid(True, which='both')
# ax.margins(x=0)
# plt.xlabel("Something (s)",size=20)
# plt.ylabel("Something (V)",size=20)
# plt.title("Initial Plot", size=24)
# plt.legend(fancybox=True, shadow=True, prop={'size': 18})
# plt.xticks(size=18,color='#4f4e4e')
# plt.yticks(size=18,color='#4f4e4e')
# sns.set(style='whitegrid')
# plt.show()


# ##FIRST PEAK CORRESPONDS TO 85Rb F=2
# third_peak = results.loc[(results[0] > 0.0945) & (results[0] < 0.0955)]

# popt3, pcov3 = sp.optimize.curve_fit(gauss_function, third_peak[0], third_peak[1], p0=[-1, 0.0015, 0.01, 0],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=10000)
# x_new=np.linspace(0.0945,0.0955,100)

# params1,ub1,lb1 = np.array([0.04,0.0005,0.07568]),np.array([np.inf,0.002,0.0778]),np.array([0,0,0.0755])
# params2,ub2,lb2 = np.array([0.2,0.0005,0.0761]),np.array([np.inf,0.001,0.0762]),np.array([0,0,0.076])
# params3,ub3,lb3 = np.array([0.23,0.0005,0.0765]),np.array([np.inf,0.001,0.0766]),np.array([0,0,0.0764])
# params4,ub4,lb4 = np.array([0.8,0.0005,0.07675]),np.array([np.inf,0.001,0.0768]),np.array([0,0,0.0767])
# params5,ub5,lb5 = np.array([1.25,0.0005,0.07715]),np.array([np.inf,0.001,0.0773]),np.array([0,0,0.077])
# params6,ub6,lb6 = np.array([0.35,0.0004,0.077857]),np.array([np.inf,0.001,0.779]),np.array([0,0,0.0778])

# poptfit3, pcovfit3 = sp.optimize.curve_fit(six_lorentzian,third_peak[0],third_peak[3],p0=[*params1,*params2,*params3,*params4,*params5,*params6,0],bounds=((*lb1,*lb2,*lb3,*lb4,*lb5,*lb6,-np.inf), (*ub1,*ub2,*ub3,*ub4,*ub5,*ub6,np.inf)),maxfev=100000)
# fit3 = six_lorentzian(x_new, *poptfit3)

# fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(x_new,gauss_function(x_new,popt3[0],popt3[1],popt3[2],popt3[3]))
# # ax.plot(third_peak[0],third_peak[1],color="Orange",label="Channel 1")
# # ax.plot(third_peak[0],third_peak[2],color="Blue",label="Channel 2")
# ax.plot(third_peak[0],third_peak[3],color="Blue",label="Difference")
# ax.axis(True)
# sns.set(style="ticks")
# ax.grid(True, which='both')
# ax.margins(x=0)
# plt.xlabel("Something (s)",size=20)
# plt.ylabel("Something (V)",size=20)
# plt.title("Initial Plot", size=24)
# plt.legend(fancybox=True, shadow=True, prop={'size': 18})
# plt.xticks(size=18,color='#4f4e4e')
# plt.yticks(size=18,color='#4f4e4e')
# sns.set(style='whitegrid')
# plt.show()


# ##FIRST PEAK CORRESPONDS TO 87Rb F=1
# fourth_peak = results.loc[(results[0] > 0.104) & (results[0] < 0.106)]

# popt4, pcov4 = sp.optimize.curve_fit(gauss_function, fourth_peak[0], fourth_peak[1], p0 = [-1, 0.0175, 0.001, 0],bounds=((-np.inf,-np.inf,-np.inf,-np.inf), (0,np.inf,np.inf,np.inf)),maxfev=10000)
# x_new=np.linspace(0.104,0.106,100)


# fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(x_new,gauss_function(x_new,popt4[0],popt4[1],popt4[2],popt4[3]))
# # ax.plot(fourth_peak[0],fourth_peak[1],color="Orange",label="Channel 1")
# # ax.plot(fourth_peak[0],fourth_peak[2],color="Blue",label="Channel 2")
# ax.plot(fourth_peak[0],fourth_peak[3],color="Blue",label="Difference")
# ax.axis(True)
# sns.set(style="ticks")
# ax.grid(True, which='both')
# ax.margins(x=0)
# plt.xlabel("Something (s)",size=20)
# plt.ylabel("Something (V)",size=20)
# plt.title("Initial Plot", size=24)
# plt.legend(fancybox=True, shadow=True, prop={'size': 18})
# plt.xticks(size=18,color='#4f4e4e')
# plt.yticks(size=18,color='#4f4e4e')
# sns.set(style='whitegrid')
# plt.show()

#----------------------------------------------------------------------------

x = results[0]
fabry_perot = results[4]

y_centres = np.array([])
x_centres = np.array([])

count=0
for i, val in enumerate(fabry_perot):
    if val >= 0.05 and i >= count:
        for j,j_val in enumerate(fabry_perot[i:]):
            if j_val <= 0.05:
                # this came to me in a dream :)
                x_centre, y_centre = max(zip(x[i:i+j], np.convolve(fabry_perot,np.ones(10)/10,mode="same")[i:i+j]), key=lambda l: l[1])

                y_centres = np.append(y_centres,y_centre)
                x_centres = np.append(x_centres,x_centre)
                count = i + j
                break 
    else:
        continue

fabry_diffs = np.ediff1d(x_centres)
x_diffs = np.arange(0,len(fabry_diffs))
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(x_diffs,fabry_diffs,color="red",label="Difference in Fabry-Perot Peaks")
plt.show()

max_cal = np.max(fabry_diffs)
min_cal = np.min(fabry_diffs)

w_spacing = (3e8) / 0.8
# w_spacing=378e6

max_scale_len = ((np.max(x)-np.min(x)) * w_spacing)/min_cal
max_scale = np.linspace(0,max_scale_len,len(fabry_perot)) / 1e9

min_scale_len = ((np.max(x)-np.min(x)) * w_spacing)/max_cal
min_scale = np.linspace(0,min_scale_len,len(fabry_perot)) / 1e9


fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(x,C1,color="Orange",label="Channel 1")
# ax.plot(x,C2,color="Blue",label="Channel 2")
ax.plot(x,difference,color="red",label="Difference")
ax.plot(x,fabry_perot,color="black",label="Fabry-Perot")
# ax.scatter(x_centres,y_centres,color="green",label="Centres")
# ax.axis(True)
# ax.grid(True, which='both')
plt.xlabel("Time (s)",size=20)
plt.ylabel("Voltage (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(x,C1,color="Orange",label="Channel 1")
# ax.plot(x,C2,color="Blue",label="Channel 2")
ax.plot(max_scale,difference,color="red",label="Difference")
ax.plot(max_scale,fabry_perot,color="black",label="Fabry-Perot")
ax.axis(True)
ax.grid(True, which='both')
plt.xlabel("Freq. (GHz)",size=20)
plt.ylabel("Voltage (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(x,C1,color="Orange",label="Channel 1")
# ax.plot(x,C2,color="Blue",label="Channel 2")
ax.plot(min_scale,difference,color="red",label="Difference")
ax.plot(min_scale,fabry_perot,color="black",label="Fabry-Perot")
ax.axis(True)
ax.grid(True, which='both')
plt.xlabel("Freq. (GHz)",size=20)
plt.ylabel("Voltage (V)",size=20)
plt.title("Initial Plot", size=24)
plt.legend(fancybox=True, shadow=True, prop={'size': 18})
plt.xticks(size=18,color='#4f4e4e')
plt.yticks(size=18,color='#4f4e4e')
sns.set(style='whitegrid')
plt.show()
