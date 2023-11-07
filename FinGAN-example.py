
# -*- coding: utf-8 -*-
"""
Created on Tue 7th Nov 2023

@author: vuletic@maths.ox.ac.uk
"""

import FinGAN
import pandas as pd
import matplotlib.pyplot as plt


h = 1
l = 10
pred = 1


###location of the dataset and etfs-stocks list
dataloc = "/data/"
etflistloc = "/stocks-etfs-list.csv"

###number of epochs of training
n_epochs = 100

#number of available GPUs
ngpu = 1

###Location for saiving results
loc = "/Fin-GAN/"

#Models, plots, results folders in the location aboce
modelsloc = loc+"TrainedModels/"
plotsloc = loc+"Plots/"
resultsloc = loc+"Results/"



tanh_coeff = 100

z_dim = 8
hid_d = 8
hid_g = 8


checkpoint_epoch = 20
batch_size = 100
diter = 1

n_epochs = 100
ngrad = 25
vl_later = False
datastart = {'lrd':[],'lrg':[],'epochs':[],'SR_val':[]}

tr = 0.8
vl = 0.1

plot = False
z_dim = 8
hid_d_s = [8]
hid_g_s = [8]
#optional to explore different learning rates
lrg_s = [0.0001]
lrd_s = [0.0001]
vl_later = True

nres = len(lrg_s)
resultsname = "results.csv"
plt.rcParams['figure.figsize'] = [15.75, 9.385]
##tickers to be used
tickers = ['AMZN','HD']
datastart = {'lrd':[],'lrg':[],'epochs':[],'SR_val':[]}
#results_df has performance on both validation and test set
results_df = pd.DataFrame(data=datastart)
corrs = [False] * len(tickers)
for j in range(len(hid_d_s)):
    for i in range(nres):
        lrg = lrg_s[i]
        lrd = lrd_s[i]
        #  For LSTM_combos set hid_d and hid_g to 0 and 1
        # hid_d = 0
        # hid_g = 1
        for tickern in range(len(tickers)):
            ticker = tickers[tickern]
            print("******************")
            print(ticker)
            print("******************")
            df_temp, corrs[tickern] = FinGAN.FinGAN_combos(ticker,loc,modelsloc,plotsloc,dataloc, etflistloc,  vl_later, lrg, lrd, n_epochs, ngrad, h, l, pred, ngpu, tanh_coeff, tr, vl, z_dim, hid_d, hid_g, checkpoint_epoch, batch_size = 100, diter = 1, plot = plot)
            # df_temp, corrs[tickern] = FinGAN.LSTM_combos(ticker,loc,modelsloc,plotsloc,dataloc, etflistloc,  vl_later, lrg, lrd, n_epochs, ngrad, h, l, pred, ngpu, tanh_coeff, tr, vl, z_dim, hid_d, hid_g, checkpoint_epoch, batch_size = 100, diter = 1, plot = plot)
            results_df = pd.concat([results_df,df_temp], ignore_index=True)
            results_df.to_csv(resultsloc+resultsname)
            print("*************")
print("DONE")


