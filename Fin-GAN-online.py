#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 2023

@author: vuletic@maths.ox.ac.uk


"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy.random as rnd

def ETF_find(etflistloc, stock):
    """
    reading a file containing information on stock memberships
    input: stock ticker
    output: corresponding ETF ticker
    """
    data = pd.read_csv(etflistloc)
    out = np.array(data['ticker_y'][data['ticker_x']==stock])[0]
    return out
    

def excessreturns_closeonly(dataloc, stock, etf, plotcheck = False):
    """
    function to get a time series of DAILY CLOSING
    etf-excess log returns for a given stock
    all prices are adjusted for stock events
    input: location of datasets, stock ticker, etf ticker
    output: time series of etf excess log returns
    optional: plot sanity check
    """
    s_df = pd.read_csv(dataloc+stock+".csv")
    e_df = pd.read_csv(dataloc+etf+".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    d1 = pd.to_datetime("2022-01-01")
    smp = (dates_dt < d1)
    s_df = s_df[smp]
    e_df = e_df[smp]
    s_log = np.log(s_df['AdjClose'])
    e_log = np.log(e_df['AdjClose'])
    dates_dt = dates_dt[smp]
    s_ret = np.diff(s_log)
    e_ret = np.diff(e_log)
    excessret = s_ret - e_ret
    
    if plotcheck:
        plt.figure(stock+" price")
        plt.title(stock+" price")
        plt.plot(dates_dt,s_df['AdjClose'])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure("Returns "+stock)
        plt.title("Returns "+stock)
        plt.plot(dates_dt[1:],s_ret, alpha = 0.7, label = 'stock')
        plt.plot(dates_dt[1:],e_ret, alpha = 0.7, label = 'etf')
        plt.plot(dates_dt[1:],excessret, alpha = 0.7, label = 'excess return')
        plt.xlabel("date")
        plt.legend()
        plt.show()
    return excessret, dates_dt[1:]

def excessreturns(dataloc, stock, etf, plotcheck = False):
    """
    function to get a time series of alternating close and open
    etf-excess log returns for a given stock
    all prices are adjusted for stock events
    input: location of datasets, stock ticker, etf ticker
    output: time series of etf excess log returns
    optional: plot sanity check
    """
    s_df = pd.read_csv(dataloc+stock+".csv")
    e_df = pd.read_csv(dataloc+etf+".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    d1 = pd.to_datetime("2022-01-01")
    smp = (dates_dt < d1)
    s_df = s_df[smp]
    dates_dt = pd.to_datetime(s_df['date'])
    e_df = e_df[smp]
    s_logclose = np.log(s_df['AdjClose'])
    e_logclose = np.log(e_df['AdjClose'])
    s_logopen = np.log(s_df['AdjOpen'])
    e_logopen = np.log(e_df['AdjOpen'])
    s_log = np.zeros(2*len(s_logclose))
    e_log = np.zeros(2*len(s_logclose))
    for i in range(len(s_logclose)):
        s_log[2 * i] = s_logopen[i]
        s_log[2 * i + 1] = s_logclose[i]
        e_log[2 * i] = e_logopen[i]
        e_log[2 * i + 1] = e_logclose[i]
    s_ret = np.diff(s_log)
    e_ret = np.diff(e_log)
    s_ret[s_ret > 0.15] = 0.15
    s_ret[s_ret < -0.15] = -0.15
    e_ret[e_ret > 0.15] = 0.15
    e_ret[e_ret < -0.15] = -0.15
    excessret = s_ret - e_ret
    dates_dt = pd.to_datetime(s_df['date'])
    if plotcheck:
        plt.figure(stock+" price")
        plt.title(stock+" price")
        plt.plot(dates_dt,s_df['AdjClose'])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure("Returns "+stock)
        plt.title("Returns "+stock)
        plt.plot(range(len(s_ret)),s_ret, alpha = 0.7, label = 'stock')
        plt.plot(range(len(e_ret)),e_ret, alpha = 0.7, label = 'etf')
        plt.plot(range(len(e_ret)),excessret, alpha = 0.7, label = 'excess return')
        plt.legend()
        plt.show()
    return excessret, dates_dt

def rawreturns(dataloc, stock, plotcheck = False):
    """
    function to get a time series of raw log returns for a given stock/etf
    all prices are adjusted for stock events
    input: location of datasets, stock ticker, etf ticker
    output: time series of etf excess log returns
    optional: plot sanity check
    """
    s_df = pd.read_csv(dataloc+stock+".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    d1 = pd.to_datetime("2022-01-01")
    smp = (dates_dt < d1)
    s_df = s_df[smp]
    dates_dt = pd.to_datetime(s_df['date'])
    s_logclose = np.log(s_df['AdjClose'])
    s_logopen = np.log(s_df['AdjOpen'])
    s_log = np.zeros(2*len(s_logclose))
    for i in range(len(s_logclose)):
        s_log[2 * i] = s_logopen[i]
        s_log[2 * i + 1] = s_logclose[i]
    s_ret = np.diff(s_log)
    s_ret[s_ret > 0.15] = 0.15
    s_ret[s_ret < -0.15] = -0.15
    dates_dt = pd.to_datetime(s_df['date'])
    if plotcheck:
        plt.figure(stock+" price")
        plt.title(stock+" price")
        plt.plot(dates_dt,s_df['AdjClose'])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure("Returns "+stock)
        plt.title("Returns "+stock)
        plt.plot(range(len(s_ret)),s_ret)
        plt.legend()
        plt.show()
    return s_ret, dates_dt

def split_train_val_test(stock, dataloc, etflistloc, tr = 0.8, vl = 0.1, h = 1, l = 10, pred = 1, plotcheck=False):
    """
    prepare etf excess log returns for a given stock
    split into train, val, test
    h: sliding window
    l: condition window (number of previous values)
    pred: prediction window
    """
    etf = ETF_find(etflistloc, stock)
    excess_returns, dates_dt = excessreturns(dataloc, stock, etf, plotcheck)
    N = len(excess_returns)
    N_tr = int(tr*N)
    N_vl = int(vl*N)
    N_tst = N - N_tr - N_vl
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr+N_vl]
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr+N_vl]
    test_sr = excess_returns[N_tr+N_vl:]
    n = int((N_tr-l-pred)/h)+1
    train_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i,:] = train_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_vl-l-pred)/h)+1
    val_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        val_data[i,:] = val_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_tst-l-pred)/h)+1
    test_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i,:] = test_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    if plotcheck:
        plt.figure("Excess returns")
        plt.plot(dates_dt,excess_returns)
        plt.title(stock+ " excess returns")
        plt.axvline(x = dates_dt[N_tr],color = "red")
        plt.axvline(x = dates_dt[N_tr+N_vl],color = "red")
        plt.show()
    return train_data,val_data,test_data, dates_dt

def split_train_testraw(stock, dataloc, tr = 0.8, vl = 0.1, h = 1, l = 10, pred = 1, plotcheck=False):
    """
    prepare raw log returns for a given stock
    split into train, test
    h: sliding window
    l: condition window (number of previous values)
    pred: prediction window
    """
    excess_returns, dates_dt = rawreturns(dataloc, stock, plotcheck)
    N = len(excess_returns)
    N_tr = int(tr*N) + int(vl*N)
    N_tst = N - N_tr
    train_sr = excess_returns[0:N_tr]
    test_sr = excess_returns[N_tr:]
    n = int((N_tr-l-pred)/h)+1
    train_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i,:] = train_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_tst-l-pred)/h)+1
    test_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i,:] = test_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h

    return train_data,test_data


def split_train_val_testraw(stock, dataloc, tr = 0.8, vl = 0.1, h = 1, l = 10, pred = 1, plotcheck=False):
    """
    prepare raw log returns for a given stock
    split into train, val, test
    h: sliding window
    l: condition window (number of previous values)
    pred: prediction window
    """
    excess_returns, dates_dt = rawreturns(dataloc, stock, plotcheck)
    N = len(excess_returns)
    N_tr = int(tr*N)
    N_vl = int(vl*N)
    N_tst = N - N_tr - N_vl
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr+N_vl]
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr+N_vl]
    test_sr = excess_returns[N_tr+N_vl:]
    n = int((N_tr-l-pred)/h)+1
    train_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i,:] = train_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_vl-l-pred)/h)+1
    val_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        val_data[i,:] = val_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    n = int((N_tst-l-pred)/h)+1
    test_data = np.zeros(shape=(n,l+pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i,:] = test_sr[l_tot:l_tot+l+pred]
        l_tot = l_tot + h
    if plotcheck:
        plt.figure("returns")
        plt.plot(dates_dt,excess_returns)
        plt.title(stock+ " =returns")
        plt.axvline(x = dates_dt[N_tr],color = "red")
        plt.axvline(x = dates_dt[N_tr+N_vl],color = "red")
        plt.show()
    return train_data,val_data,test_data, dates_dt

#LSTM ForGAN generator
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        noise_dim: the dimension of the noise, a scalar
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, noise_dim,cond_dim, hidden_dim,output_dim,mean,std):
        super(Generator, self).__init__()
        self.input_dim = noise_dim+cond_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        #predicting a single value, so the output dimension is 1
        self.mean = mean
        self.std = std
        #Add the modules
   
        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
        # nn.init.xavier_normal_(self.lstm.weight)
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.linear1 = nn.Linear(in_features=self.hidden_dim+self.noise_dim, out_features=self.hidden_dim+self.noise_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(in_features=self.hidden_dim+self.noise_dim, out_features=output_dim)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.ReLU()
       

    def forward(self, noise,condition,h_0,c_0):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''
        #x = combine_vectors(noise.to(torch.float),condition.to(torch.float),2)
        condition = (condition-self.mean)/self.std
        out, (h_n, c_n) = self.lstm(condition, (h_0, c_0))
        out = combine_vectors(noise.to(torch.float),h_n.to(torch.float),dim=-1)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = out*self.std+self.mean
        return out

class LSTM(nn.Module):
    '''
    Values:
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, noise_dim,cond_dim, hidden_dim,output_dim,mean,std):
        super(LSTM, self).__init__()
        self.input_dim = noise_dim+cond_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.noise_dim = noise_dim
        #predicting a single value, so the output dimension is 1
        self.mean = mean
        self.std = std
        #Add the modules
   
        self.lstm = nn.LSTM(input_size=cond_dim, hidden_size=self.output_dim, num_layers=1, dropout=0)
        # nn.init.xavier_normal_(self.lstm.weight)
        # nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        # nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.activation = nn.ReLU()
       

    def forward(self, condition,h_0,c_0):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''
        #x = combine_vectors(noise.to(torch.float),condition.to(torch.float),2)
        condition = (condition-self.mean)/self.std
        out, (h_n, c_n) = self.lstm(condition, (h_0, c_0))
        out = out*self.std+self.mean
        return out



#discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
      in_dim: the input dimension (noise dim + conditin dim + forecast dim for the condition for this dataset), a scalar
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, in_dim, hidden_dim,mean,std):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.mean = mean
        self.std = std
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=1)
        nn.init.xavier_normal_(self.linear.weight)
        self.sigmoid = nn.Sigmoid()
       


    def forward(self, in_chan,h_0,c_0):
        '''
        in_chan: concatenated condition with real or fake
        h_0 and c_0: for the LSTM
        '''
        x = in_chan
        x = (x-self.mean)/self.std
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))      
        out = self.linear(h_n)
        out = self.sigmoid(out)
        return out
    
def combine_vectors(x, y,dim=-1):
    '''
    Function for combining two tensors
    '''
    combined = torch.cat([x,y],dim=dim)
    combined = combined.to(torch.float)
    return combined

def getPnL(predicted,real,nsamp):
    """
    PnL per trade given nsamp samples, predicted forecast, real data realisations
    in bpts
    """
    sgn_fake = torch.sign(predicted)
    PnL = torch.sum(sgn_fake*real)
    PnL = 10000*PnL/nsamp
    return PnL

def getSR(predicted,real):
    """
    Sharpe Ratio given forecasts predicted of real (not annualised)
    """
    sgn_fake = torch.sign(predicted)
    SR = torch.mean(sgn_fake * real) / torch.std(sgn_fake * real)
    return SR

def Evaluation2(ticker,freq,gen,test_data, val_data, h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name, plot = False):
    """
    Evaluation of a GAN model on a single stock
    """
    df_temp = False
    dt = {'lrd':lrd,'lrg':lrg,'type': losstype,'epochs':n_epochs, 'ticker':ticker,  'hid_g':hid_g, 'hid_d':hid_d}
    #print("Validation set best PnL (in bp): ",PnL_best)
    #print("Checkpoint epoch: ",checkpoint_last_epoch+1)
    ntest = test_data.shape[0]
    gen.eval()
    with torch.no_grad():
        condition1 = test_data[:,0:l]
        condition1 = condition1.unsqueeze(0)
        condition1 = condition1.to(device)
        condition1 = condition1.to(torch.float)
        ntest = test_data.shape[0]
        h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
        fake1 = gen(fake_noise,condition1,h0,c0)
        fake1 = fake1.unsqueeze(0).unsqueeze(2)
        generated1 = torch.empty([1,1,1,ntest,1000])
        generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()
        #generated1 = fake1.detach()
        for i in range(999):        
            fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
            fake1 = gen(fake_noise,condition1,h0,c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            #print(fake.shape)
            generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
            #generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
#             print(generated1.shape)
            del fake1
            del fake_noise
        #rmse = torch.sqrt(torch.mean((fake-real)**2))
        #mae = torch.mean(torch.abs(fake-real))
    #print("RMSE: ", rmse)
    #print("MAE: ",mae)
    b1 = generated1.squeeze()
    mn1 = torch.mean(b1,dim=1)
    real1 = test_data[:,-1]
    rl1 = real1.squeeze()
    rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
    mae1 = torch.mean(torch.abs(mn1-rl1))
    #print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE'] = rmse1.item()
    dt['MAE'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)
    PnL1 = getPnL(ft1,rl1,ntest)
    #print("PnL in bp", PnL)

    #look at the Sharpe Ratio
    n_b1 = b1.shape[1]
    PnL_ws1 = torch.empty(ntest)
    for i1 in range(ntest):
        fk1 = b1[i1,:]
        pu1 = (fk1>=0).sum()
        pu1 = pu1/n_b1
        pd1 = 1-pu1
        PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
        PnL_ws1[i1] = PnL_temp1.item()
    PnL_ws1 = np.array(PnL_ws1)
    PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
    PnL_even = np.zeros(int(0.5 * len(PnL_ws1)))
    PnL_odd = np.zeros(int(0.5 * len(PnL_ws1)))
    for i1 in range(len(PnL_wd1)):
        PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
        PnL_even[i1] = PnL_ws1[2 * i1]
        PnL_odd[i1] = PnL_ws1[2 * i1 + 1]
    PnL_test = PnL_wd1
    PnL_w_m1 = np.mean(PnL_wd1)
    PnL_w_std1 = np.std(PnL_wd1)
    SR1 = PnL_w_m1/PnL_w_std1
    #print("Sharpe Ratio: ",SR)
    dt['SR_w scaled'] = SR1*np.sqrt(252)
    dt['PnL_w'] = PnL_w_m1
    
    if (ntest % 2) == 0:
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    else:
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    print("Annualised (test) SR_w: ",SR1*np.sqrt(252))
    
    distcheck = np.array(b1[1,:].cpu())
    means = np.array(mn1.detach())
    reals = np.array(rl1.detach())
    dt['Corr'] = np.corrcoef([means,reals])[0,1]
    dt['Pos mn'] = np.sum(means >0)/ len(means)
    dt['Neg mn'] = np.sum(means <0)/ len(means)
    print('Correlation ',np.corrcoef([means,reals])[0,1] )
    
    dt['narrow dist'] = (np.std(distcheck)<0.0002)

    means_gen = means
    reals_test = reals
    distcheck_test = distcheck
    rl_test = reals[1]

    mn = torch.mean(b1,dim=1)
    mn = np.array(mn.cpu())
    dt['narrow means dist'] = (np.std(mn)<0.0002)

    ntest = val_data.shape[0]
    gen.eval()
    with torch.no_grad():
        condition1 = val_data[:,0:l]
        condition1 = condition1.unsqueeze(0)
        condition1 = condition1.to(device)
        condition1 = condition1.to(torch.float)
        ntest = val_data.shape[0]
        h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
        fake1 = gen(fake_noise,condition1,h0,c0)
        fake1 = fake1.unsqueeze(0).unsqueeze(2)
        generated1 = torch.empty([1,1,1,ntest,1000])
        generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()
        #generated1 = fake1.detach()
        for i in range(999):        
            fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
            fake1 = gen(fake_noise,condition1,h0,c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            #print(fake.shape)
            generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
            #generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
#             print(generated1.shape)
            del fake1
            del fake_noise
        #rmse = torch.sqrt(torch.mean((fake-real)**2))
        #mae = torch.mean(torch.abs(fake-real))
    #print("RMSE: ", rmse)
    #print("MAE: ",mae)
    b1 = generated1.squeeze()
    mn1 = torch.mean(b1,dim=1)
    real1 = val_data[:,-1]
    rl1 = real1.squeeze()
    rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
    mae1 = torch.mean(torch.abs(mn1-rl1))
    #print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE val'] = rmse1.item()
    dt['MAE val'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)
    #print("PnL in bp", PnL)

    #look at the Sharpe Ratio
    n_b1 = b1.shape[1]
    PnL_ws1 = torch.empty(ntest)
    for i1 in range(ntest):
        fk1 = b1[i1,:]
        pu1 = (fk1>=0).sum()
        pu1 = pu1/n_b1
        pd1 = 1-pu1
        PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
        PnL_ws1[i1] = PnL_temp1.item()
    PnL_ws1 = np.array(PnL_ws1)
    PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
    for i1 in range(len(PnL_wd1)):
        PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
    PnL_w_m1 = np.mean(PnL_wd1)
    PnL_w_std1 = np.std(PnL_wd1)
    SR1 = PnL_w_m1/PnL_w_std1
    #print("Sharpe Ratio: ",SR)
    dt['PnL_w val'] = PnL_w_m1
    dt['SR_w scaled val'] = SR1*np.sqrt(252)
    
    print("Annualised (val) SR_w : ",SR1*np.sqrt(252))
    
    means = np.array(mn1.detach())
    reals = np.array(rl1.detach())
    dt['Corr val'] = np.corrcoef([means,reals])[0,1]
    dt['Pos mn val'] = np.sum(means >0)/ len(means)
    dt['Neg mn val'] = np.sum(means <0)/ len(means)
    
    df_temp = pd.DataFrame(data=dt,index=[0])
    
    return df_temp, PnL_test, PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test

def Evaluation3(tickers,freq,gen,test, val, h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name, plot = False):
    """
    Evaluation of a GAN model in the universality setting (multiple tickers)
    """
    df_temp = False
    dt = {'lrd':[],'lrg':[],'type': [],'epochs':[], 'ticker':[],  'hid_g':[], 'hid_d':[]}
    results_df = pd.DataFrame(data = dt)
    PnLs_test = np.zeros((len(tickers), int(0.5 * test[0].shape[0])))
    PnLs_val = np.zeros((len(tickers), int(0.5 * val[0].shape[0])))
    means_test = np.zeros((len(tickers), test[0].shape[0]))
    means_val = np.zeros((len(tickers), val[0].shape[0]))
    # print(means_test.shape)
    #print("Validation set best PnL (in bp): ",PnL_best)
    #print("Checkpoint epoch: ",checkpoint_last_epoch+1)
    for ii in tqdm(range(len(tickers))):
        val_data = val[ii]
        test_data = test[ii]
        ticker = tickers[ii]
        dt = {'lrd':lrd,'lrg':lrg,'type': losstype,'epochs':n_epochs, 'ticker':ticker,  'hid_g':hid_g, 'hid_d':hid_d}
        ntest = test_data.shape[0]
        gen.eval()
        with torch.no_grad():
            condition1 = test_data[:,0:l]
            condition1 = condition1.unsqueeze(0)
            condition1 = condition1.to(device)
            condition1 = condition1.to(torch.float)
            ntest = test_data.shape[0]
            h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
            c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
            fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
            fake1 = gen(fake_noise,condition1,h0,c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            generated1 = torch.empty([1,1,1,ntest,1000])
            generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()
            #generated1 = fake1.detach()
            for i in range(999):        
                fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
                fake1 = gen(fake_noise,condition1,h0,c0)
                fake1 = fake1.unsqueeze(0).unsqueeze(2)
                #print(fake.shape)
                generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
                #generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
    #             print(generated1.shape)
                del fake1
                del fake_noise
            #rmse = torch.sqrt(torch.mean((fake-real)**2))
            #mae = torch.mean(torch.abs(fake-real))
        #print("RMSE: ", rmse)
        #print("MAE: ",mae)
        b1 = generated1.squeeze()
        mn1 = torch.mean(b1,dim=1)
        # print(mn1.shape)
        means_test[ii, :] = np.array(mn1.detach())
        real1 = test_data[:,-1]
        rl1 = real1.squeeze()
        rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
        mae1 = torch.mean(torch.abs(mn1-rl1))
        #print("RMSE: ",rmse,"MAE: ",mae)
        dt['RMSE'] = rmse1.item()
        dt['MAE'] = mae1.item()
        ft1 = mn1.clone().detach().to(device)        #print("PnL in bp", PnL)
    
        #look at the Sharpe Ratio
        n_b1 = b1.shape[1]
        PnL_ws1 = torch.empty(ntest)
        for i1 in range(ntest):
            fk1 = b1[i1,:]
            pu1 = (fk1>=0).sum()
            pu1 = pu1/n_b1
            pd1 = 1-pu1
            PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
            PnL_ws1[i1] = PnL_temp1.item()
        PnL_ws1 = np.array(PnL_ws1)
        PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
        for i1 in range(len(PnL_wd1)):
            PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
        PnLs_test[ii, :] = PnL_wd1
        PnL_w_m1 = np.mean(PnL_wd1)
        PnL_w_std1 = np.std(PnL_wd1)
        SR1 = PnL_w_m1/PnL_w_std1
        #print("Sharpe Ratio: ",SR)
        dt['PnL_w'] = PnL_w_m1
        dt['SR_w scaled'] = SR1 * np.sqrt(252)    
        # print("Annualised (test) SR_w: ",SR1*np.sqrt(252 * freq))
        # print("Annualised (test) SR_m: ", np.sqrt(252 * freq) * getSR(ft1,rl1).item())
        dist_loc = plotsloc+"distcheck-"+f_name+".png"
        
        distcheck = np.array(b1[1,:].cpu())
        means = np.array(mn1.detach())
        reals = np.array(rl1.detach())
        dt['Corr'] = np.corrcoef([means,reals])[0,1]
        dt['Pos mn'] = np.sum(means >0)/ len(means)
        dt['Neg mn'] = np.sum(means <0)/ len(means)
        # print('Correlation ',np.corrcoef([means,reals])[0,1] )
        
        dt['narrow dist'] = (np.std(distcheck)<0.0002)
    
        means_loc = plotsloc+"recovered-means-"+f_name+".png"
    
    
        mn = torch.mean(b1,dim=1)
        mn = np.array(mn.cpu())
        dt['narrow means dist'] = (np.std(mn)<0.0002)
    
    
        ntest = val_data.shape[0]
        gen.eval()
        with torch.no_grad():
            condition1 = val_data[:,0:l]
            condition1 = condition1.unsqueeze(0)
            condition1 = condition1.to(device)
            condition1 = condition1.to(torch.float)
            ntest = val_data.shape[0]
            h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
            c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
            fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
            fake1 = gen(fake_noise,condition1,h0,c0)
            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            generated1 = torch.empty([1,1,1,ntest,1000])
            generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()
            #generated1 = fake1.detach()
            for i in range(999):        
                fake_noise = torch.randn(1,ntest, z_dim, device=device,dtype=torch.float)
                fake1 = gen(fake_noise,condition1,h0,c0)
                fake1 = fake1.unsqueeze(0).unsqueeze(2)
                #print(fake.shape)
                generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
                #generated1 = combine_vectors(generated1, fake1.detach(), dim=-1)
    #             print(generated1.shape)
                del fake1
                del fake_noise
            #rmse = torch.sqrt(torch.mean((fake-real)**2))
            #mae = torch.mean(torch.abs(fake-real))
        #print("RMSE: ", rmse)
        #print("MAE: ",mae)
        b1 = generated1.squeeze()
        mn1 = torch.mean(b1,dim=1)
        means_val[ii, :] = np.array(mn1.detach())

        real1 = val_data[:,-1]
        rl1 = real1.squeeze()
        rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
        mae1 = torch.mean(torch.abs(mn1-rl1))
        #print("RMSE: ",rmse,"MAE: ",mae)
        dt['RMSE val'] = rmse1.item()
        dt['MAE val'] = mae1.item()
        ft1 = mn1.clone().detach().to(device)
        #print("PnL in bp", PnL)
    
        #look at the Sharpe Ratio
        n_b1 = b1.shape[1]
        PnL_ws1 = torch.empty(ntest)
        for i1 in range(ntest):
            fk1 = b1[i1,:]
            pu1 = (fk1>=0).sum()
            pu1 = pu1/n_b1
            pd1 = 1-pu1
            PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
            PnL_ws1[i1] = PnL_temp1.item()
        PnL_ws1 = np.array(PnL_ws1)
    
        PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
        for i1 in range(len(PnL_wd1)):
            PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
        PnLs_val[ii, :] = PnL_wd1
        PnL_w_m1 = np.mean(PnL_wd1)
        PnL_w_std1 = np.std(PnL_wd1)
        SR1 = PnL_w_m1/PnL_w_std1
        #print("Sharpe Ratio: ",SR)
        dt['PnL_w val'] = PnL_w_m1
        dt['SR_w scaled val'] = SR1*np.sqrt(freq)
        
        
        # print("Annualised (val) SR_w : ",SR1*np.sqrt(252 * freq))
        # print("Annualised (val) SR_m : ", np.sqrt(252 * freq) * getSR(ft1,rl1).item())
        
        means = np.array(mn1.detach())
        reals = np.array(rl1.detach())
        dt['Corr val'] = np.corrcoef([means,reals])[0,1]
        dt['Pos mn val'] = np.sum(means >0)/ len(means)
        dt['Neg mn val'] = np.sum(means <0)/ len(means)
        df_temp = pd.DataFrame(data=dt,index=[0])
        results_df = pd.concat([results_df,df_temp], ignore_index=True)
    PnL_test = np.sum(PnLs_test,axis=0)
    PnL_val = np.sum(PnLs_val,axis=0)
            
    return results_df, PnL_test, PnL_val, means_test, means_val

def GradientCheck(ticker, gen, disc, gen_opt, disc_opt, criterion, n_epochs, train_data,batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Gradient norm check
    """
    ntrain = train_data.shape[0]
    nbatches = ntrain//batch_size+1
    BCE_norm = torch.empty(nbatches*n_epochs, device = device)
    PnL_norm = torch.empty(nbatches*n_epochs, device = device)
    MSE_norm = torch.empty(nbatches*n_epochs, device = device)
    SR_norm = torch.empty(nbatches*n_epochs, device = device)
    STD_norm = torch.empty(nbatches*n_epochs, device = device)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]

    #currstep = 0
    #train the discriminator more

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)

            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)
            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            # Update generator
            # Zero out the generator gradients
            

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)
            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            MSE = (torch.norm(ft-rl)**2) / curr_batch_size
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))
            STD = torch.std(PnL_s)
            gen_opt.zero_grad() 
            SR.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            SR_norm[epoch*nbatches+i] = total_norm
            
            gen_opt.zero_grad() 
            PnL.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            PnL_norm[epoch*nbatches+i] = total_norm
            
            gen_opt.zero_grad() 
            MSE.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            MSE_norm[epoch*nbatches+i] = total_norm
            
            gen_opt.zero_grad() 
            STD.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            STD_norm[epoch*nbatches+i] = total_norm
            
            gen_opt.zero_grad()   
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            BCE_norm[epoch*nbatches+i] = total_norm
            gen_opt.step()
            

    alpha = torch.mean(BCE_norm / PnL_norm)
    beta =  torch.mean(BCE_norm / MSE_norm)
    gamma =  torch.mean(BCE_norm / SR_norm)
    delta = torch.mean(BCE_norm / STD_norm)
    print("Completed. ")
    print(r"$\alpha$:", alpha)
    print(r"$\beta$:", beta)
    print(r"$\gamma$:", gamma)
    print(r"$\delta$:", delta)
    
    if plot:
        plt.figure(ticker + " BCE norm")
        plt.title(ticker + " BCE norm")
        plt.plot(range(len(BCE_norm)),BCE_norm)
        plt.xlabel("iteration")
        plt.ylabel(r"$L^2$ norm")
        plt.show()
    
        plt.figure(ticker + " PnL norm")
        plt.title(ticker +" PnL norm")
        plt.plot(range(len(BCE_norm)),PnL_norm)
        plt.xlabel("iteration")
        plt.ylabel(r"$L^2$ norm")
        plt.show()
        
        plt.figure(ticker + " MSE norm")
        plt.title(ticker + " MSE norm")
        plt.plot(range(len(BCE_norm)), MSE_norm)
        plt.xlabel("iteration")
        plt.ylabel(r"$L^2$ norm")
        plt.show()
        
        plt.figure(ticker + " SR norm")
        plt.title("SR norm")
        plt.plot(range(len(BCE_norm)),SR_norm)
        plt.xlabel("iteration")
        plt.ylabel(r"$L^2$ norm")
        plt.show()
        
        plt.figure(ticker + " STD norm")
        plt.title(ticker + " STD norm")
        plt.plot(range(len(BCE_norm)),STD_norm)
        plt.ylabel(r"$L^2$ norm")
        plt.xlabel("iteration")
        plt.show()
        
        # plt.figure(ticker + " Norms")
        # plt.title(ticker + " gradient norms")
        # plt.plot(range(len(BCE_norm)),BCE_norm, label = "BCE")
        # plt.plot(range(len(BCE_norm)),PnL_norm, label = "PnL")
        # plt.plot(range(len(BCE_norm)),SR_norm, label = "SR")
        # plt.plot(range(len(BCE_norm)),STD_norm, label = "STD")
        # plt.ylabel(r"$L^2$ norm")
        # plt.xlabel("iteration")
        # plt.legend(loc = 'best')
        # plt.show()
    
    
    return gen, disc, gen_opt, disc_opt, alpha, beta, gamma, delta


def TrainLoopForGAN(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for the BCE GAN (ForGAN)
    """
    ntrain = train_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed, checkpoint epoch: ", checkpoint_last_epoch)
    # print("PnL val (best):", PnL_best)
    print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainPnLnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainPnLMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot=False):
    """
    Training loop: PnL and MSE loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more
    SR_best = 0
    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL + beta * SqLoss
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss


    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed ")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt

def TrainLoopMainPnLMSESRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL, MSE, SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL + beta * SqLoss - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt
def TrainLoopMainPnLMSESTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL, MSE, STD loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL + beta * SqLoss + delta * torch.std(PnL_s)
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt
def TrainLoopMainPnLSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    checkpoint_last_epoch = 0
    SR_best = 0
    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    
    return gen, disc, gen_opt, disc_opt
def TrainLoopMainMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: MSE loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))  + beta * SqLoss
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt
def TrainLoopMainSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt
def TrainLoopMainSRMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: SR, MSE loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) + beta * SqLoss - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss


    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, disc, gen_opt, disc_opt
def TrainLoopMainPnLSTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop: PnL, STD loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    discloss = [False] * (nbatches*n_epochs)
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    dscpred_real = [False] * (nbatches*n_epochs)
    dscpred_fake = [False] * (nbatches*n_epochs)
    PnL_best = 0
    checkpoint_last_epoch = 0
    SR_best = 0
    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            for j in range(diter):
                disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size
                noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)

            # Get outputs from the generator
                fake = gen(noise,condition,h_0g,c_0g)
                # fake = fake.unsqueeze(0)
                fake_and_condition = combine_vectors(condition,fake,dim=-1)
                fake_and_condition.to(torch.float)
                real_and_condition = combine_vectors(condition,real,dim=-1)
                
                disc_fake_pred = disc(fake_and_condition.detach(),h_0d,c_0d)
                disc_real_pred = disc(real_and_condition,h_0d,c_0d)

            #Updating the discriminator

                disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                #disc_loss.backward(retain_graph=True)
                disc_loss.backward()
                disc_opt.step()

            dscr = disc_real_pred[0][0][0].detach().item()
            dscfk = disc_fake_pred[0][0][0].detach().item()
            dscpred_real[epoch*nbatches+i] = dscr
            dscpred_fake[epoch*nbatches+i] = dscfk

            #fksmpl.append(fake.detach())
            #rlsmpl.append(real.detach())


            # Get the predictions from the discriminator



            dloss = disc_loss.detach().item()
            discloss[epoch*nbatches+i] = dloss



            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()

            noise = torch.randn(1,curr_batch_size, z_dim, device=device,dtype=torch.float)


            fake = gen(noise,condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            fake_and_condition = combine_vectors(condition,fake,dim=-1)

            disc_fake_pred = disc(fake_and_condition,h_0d,c_0d)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)

            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            STD = torch.std(PnL_s)

            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) - alpha * PnL + delta * STD
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
        plt.figure("Disc pred PnL STD")
        plt.plot(range(len(dscpred_fake)), dscpred_fake, alpha = 0.5, label = 'generated')
        plt.plot(range(len(dscpred_fake)), dscpred_real, alpha = 0.5, label = 'real')
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Gen loss PnL STD")
        plt.title("Gen loss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()
        
        plt.figure("Disc loss PnL STD")
        plt.title("Disc loss")
        plt.plot(range(len(discloss)),discloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    
    return gen, disc, gen_opt, disc_opt

def FinGAN_combos(ticker,loc,modelsloc,plotsloc,dataloc, etflistloc,  vl_later = True, lrg = 0.0001, lrd = 0.0001, n_epochs = 500, ngrad = 100, h = 1, l = 10, pred = 1, ngpu = 1, tanh_coeff = 100, tr = 0.8, vl = 0.1, z_dim = 32, hid_d = 64, hid_g = 8, checkpoint_epoch = 20, batch_size = 100, diter = 1, plot = False, freq = 2):
    """
    FinGAN: looking at all combinations, performance on both validation and test set for all
    """
    #initialise the networks first:
    datastart = {'lrd':[],'lrg':[],'epochs':[],'SR_val':[]}
    results_df = pd.DataFrame(data=datastart)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    if ticker[0] == 'X':  
        train_data,val_data,test_data, dates_dt = split_train_val_testraw(ticker, dataloc, tr, vl, h, l, pred, plotcheck = False)
    else:
        train_data,val_data,test_data, dates_dt = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plotcheck = False)
    data_tt = torch.from_numpy(train_data)
    train_data = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(test_data)
    test_data = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(val_data)
    validation_data = data_tt.to(torch.float).to(device)
    ntest = test_data.shape[0]
    condition_size = l
    target_size = pred
    ref_mean = torch.mean(train_data[0:batch_size,:])
    ref_std = torch.std(train_data[0:batch_size,:])
    discriminator_indim = condition_size+target_size

    gen = Generator(noise_dim=z_dim,cond_dim=condition_size, hidden_dim=hid_g,output_dim=pred,mean =ref_mean,std=ref_std)
    gen.to(device)

    disc = Discriminator(in_dim=discriminator_indim, hidden_dim=hid_d,mean=ref_mean,std=ref_std)
    disc.to(device)

    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=lrd)

    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    gen, disc, gen_opt, disc_opt, alpha, beta, gamma, delta = GradientCheck(ticker, gen, disc, gen_opt, disc_opt, criterion, ngrad, train_data,batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)

    f_name = modelsloc + ticker + "-Fin-GAN-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg-"
    f_name1 = ticker + "-Fin-GAN-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    PnL_test = [False] * 10
    print("PnL")
    losstype = "PnL"
    genPnL, discPnL, gen_optPnL, disc_optPnL = TrainLoopMainPnLnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnL.state_dict()}, f_name + "PnL_generator_checkpoint.pth")
    df_temp, PnL_test[0], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnL,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    
    pd.DataFrame(PnL_test[0]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")
    plt.figure("Cummulative PnL "+ticker)
    plt.title("Cummulative PnL "+ticker)
    plt.grid(b = True)
    plt.xlabel("date")
    plt.xticks(rotation=45)
    plt.ylabel("bpts")
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[0]), label = "PnL")
    plt.legend(loc='best')
    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.title("Intraday cummulative PnL "+ticker)
        plt.grid(b=True)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        
        plt.ylabel("bpts")
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = "PnL")
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.title("Overnight cummulative PnL "+ticker)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.grid(b=True)
        
        plt.ylabel("bpts")
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = "PnL")
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.title("Overnight cummulative PnL "+ticker)
        plt.grid(b=True)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.ylabel("bpts")
        
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = "PnL")
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.title("Intraday cummulative PnL "+ticker)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.grid(b=True)
        plt.ylabel("bpts")
        
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = "PnL")
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.title("Simulated distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True, stacked=True, label = "PnL")
    plt.xlabel("excess return")
    plt.ylabel("density")
    plt.grid(b=True)
    plt.legend(loc='best')
    plt.axvline(rl_test, color='k', linestyle='dashed', linewidth = 2)
    
    plt.figure("Means "+ticker)
    plt.title("Simulated means "+ticker)
    plt.hist(reals_test, alpha = 0.6, bins = 100,density = True, stacked=True, label = "True")
    plt.hist(means_gen,alpha = 0.5, bins=100, density = True, stacked=True,label = "PnL")
    plt.xlabel("excess return")
    plt.ylabel("density")
    plt.legend(loc='best')
    plt.grid(b=True)
    
    
    print("PnL MSE")
    losstype = "PnL MSE"
    genPnLMSE, discPnLMSE, gen_optPnLMSE, disc_optPnLMSE = TrainLoopMainPnLMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSE.state_dict()}, f_name + "PnLMSE_generator_checkpoint.pth")
    df_temp,  PnL_test[1], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    
    pd.DataFrame(PnL_test[1]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[1]), label = losstype)
    plt.legend(loc='best')
    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')

    
    print("PnL MSE STD")
    losstype = "PnL MSE STD"
    genPnLMSESTD, discPnLMSESTD, gen_optPnLMSESTD, disc_optPnLMSESTD = TrainLoopMainPnLMSESTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLMSESTD.state_dict()}, f_name + "PnLMSESTD_generator_checkpoint.pth")
    df_temp,  PnL_test[2], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLMSESTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE STD", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[2]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[2]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')
    
    print("PnL MSE SR")
    losstype = "PnL MSE SR"
    genPnLMSESR, discPnLMSESR, gen_optPnLMSESR, disc_optPnLMSESR = TrainLoopMainPnLMSESRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLMSESR.state_dict()}, f_name + "PnLMSESR_generator_checkpoint.pth")
    df_temp,  PnL_test[3], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLMSESR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[3]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[3]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')
    
    print("PnL SR")
    losstype = "PnL SR"
    genPnLSR, discPnLSR, gen_optPnLSR, disc_optPnLSR = TrainLoopMainPnLSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "PnLSR_generator_checkpoint.pth")
    df_temp,  PnL_test[4], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[4]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[4]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50, density = True,stacked=True,label = losstype)
    plt.legend(loc='best')
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100, density = True,stacked=True,label = losstype)
    plt.legend(loc='best')
    
    print("PnL STD")
    losstype = "PnL STD"
    genPnLSTD, discPnLSTD, gen_optPnLSTD, disc_optPnLSTD = TrainLoopMainPnLSTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "PnLSTD_generator_checkpoint.pth")
    df_temp,  PnL_test[5], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genPnLSTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL STD", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[5]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[5]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')
    
    print("SR")
    losstype = "SR"
    genSR, discSR, gen_optSR, disc_optSR = TrainLoopMainSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "SR_generator_checkpoint.pth")
    df_temp,  PnL_test[6], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[6]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[6]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')
    
    print("SR MSE")
    losstype = "SR MSE"
    genSRMSE, discSRMSE, gen_optSRMSE, disc_optSRMSE = TrainLoopMainSRMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device,  plot)
    torch.save({'g_state_dict': genSRMSE.state_dict()}, f_name + "SRMSE_generator_checkpoint.pth")
    df_temp,  PnL_test[7], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genSRMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[7]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[7]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50, density = True,stacked=True,label = losstype)
    plt.legend(loc='best')
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')
    
    print("MSE")
    losstype = "MSE"
    genMSE, discMSE, gen_optMSE, disc_optMSE = TrainLoopMainMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genMSE.state_dict()}, f_name + "MSE_generator_checkpoint.pth")
    df_temp,  PnL_test[8], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[8]), label = losstype)
    plt.legend(loc='best')
    pd.DataFrame(PnL_test[8]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.legend(loc='best')
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50,density = True, stacked=True,label = losstype)
    plt.legend(loc='best')
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True,stacked=True, label = losstype)
    plt.legend(loc='best')

    print("ForGAN")
    losstype = "ForGAN"
    genfg, discfg, gen_optfg, disc_optfg = TrainLoopForGAN(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genfg.state_dict()}, f_name + "ForGAN_generator_checkpoint.pth")
    df_temp,  PnL_test[9], PnL_even, PnL_odd, means_gen, reals_test, distcheck_test, rl_test = Evaluation2(ticker,freq,genfg,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "BCE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[9]), label = losstype)
    plt.legend(loc='best')
    plt.savefig(plotsloc+ticker+"-FinGAN-CummPnL.png")
    plt.show()
    pd.DataFrame(PnL_test[9]).to_csv(loc+"PnLs/"+ticker+"-FinGAN-"+losstype+".csv")

    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.savefig(plotsloc+ticker+"-FinGAN-intradaycummPnL.png")
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.savefig(plotsloc+ticker+"-FinGAN-overnightcummPnL.png")
        plt.legend(loc='best')
        plt.show()
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.savefig(plotsloc+ticker+"-FinGAN-overnightcummPnL.png")
        plt.legend(loc='best')
        plt.show()
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even),label = losstype)
        plt.savefig(plotsloc+ticker+"-FinGAN-intradaycummPnL.png")
        plt.legend(loc='best')
        plt.show()
    
    plt.figure("Sample distribution "+ticker)
    plt.hist(distcheck_test,alpha = 0.5, bins=50, density = True,stacked=True, label = losstype)
    plt.savefig(plotsloc+ticker+"-FinGAN-sample-dist.png")
    plt.legend(loc='best')
    plt.show()
    
    plt.figure("Means "+ticker)
    plt.hist(means_gen,alpha = 0.5, bins=100,density = True, stacked=True,label = losstype)
    plt.savefig(plotsloc+ticker+"-FinGAN-means.png")
    plt.legend(loc='best')
    plt.show()

    corr_m = np.corrcoef(PnL_test)
    
    # can return tge best (validation) generator here too

    return results_df,corr_m


def GradientCheckLSTM(ticker, gen, gen_opt, n_epochs, train_data,batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Gradient check for LSTM-Fin
    """
    ntrain = train_data.shape[0]
    nbatches = ntrain//batch_size+1
    PnL_norm = torch.empty(nbatches*n_epochs, device = device)
    MSE_norm = torch.empty(nbatches*n_epochs, device = device)
    SR_norm = torch.empty(nbatches*n_epochs, device = device)
    STD_norm = torch.empty(nbatches*n_epochs, device = device)

    fake_and_condition = False
    real_and_condition = False

    disc_fake_pred = False
    disc_real_pred = False
    totlen = train_data.shape[0]

    #currstep = 0
    #train the discriminator more

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)

           
            fake = gen(condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            MSE = (torch.norm(ft-rl)**2) / curr_batch_size
            SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))
            STD = torch.std(PnL_s)
            gen_opt.zero_grad() 
            SR.backward(retain_graph=True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #list of gradient norms
            SR_norm[epoch*nbatches+i] = total_norm
            
            gen_opt.zero_grad() 
            PnL.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            PnL_norm[epoch*nbatches+i] = total_norm
            
            gen_opt.zero_grad() 
            STD.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            STD_norm[epoch*nbatches+i] = total_norm
            
            gen_opt.zero_grad() 
            MSE.backward(retain_graph = True)
            total_norm = 0
            for p in gen.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            MSE_norm[epoch*nbatches+i] = total_norm
            
            gen_opt.step()
            

    alpha = torch.mean(MSE_norm / PnL_norm)
    beta =  0
    gamma =  torch.mean(MSE_norm / SR_norm)
    delta = torch.mean(MSE_norm / STD_norm)
    print("Completed. ")
    print(r"$\alpha$:", alpha)
    print(r"$\beta$:", beta)
    print(r"$\gamma$:", gamma)
    print(r"$\delta$:", delta)
    
    if plot:

    
        plt.figure(ticker + " PnL norm")
        plt.title("PnL norm")
        plt.plot(range(len(MSE_norm)),PnL_norm)
        plt.show()
        
        plt.figure(ticker + " MSE norm")
        plt.title("MSE norm")
        plt.plot(range(len(MSE_norm)), MSE_norm)
        plt.show()
        
        plt.figure(ticker + " SR norm")
        plt.title("SR norm")
        plt.plot(range(len(MSE_norm)),SR_norm)
        plt.show()
        
        plt.figure(ticker + " std norm")
        plt.title("std norm")
        plt.plot(range(len(MSE_norm)),STD_norm)
        plt.show()
    
    return gen, gen_opt, alpha, beta, gamma, delta



def Evaluation2LSTM(ticker,freq,gen,test_data, val_data, h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name, plot = False):
    """
    LSTM(-FIn) evaluation on  a single stock
    """
    df_temp = False
    dt = {'lrd':lrd,'lrg':lrg,'type': losstype,'epochs':n_epochs, 'ticker':ticker}
    #print("Validation set best PnL (in bp): ",PnL_best)
    #print("Checkpoint epoch: ",checkpoint_last_epoch+1)
    ntest = test_data.shape[0]
    gen.eval()
    with torch.no_grad():
        condition1 = test_data[:,0:l]
        condition1 = condition1.unsqueeze(0)
        condition1 = condition1.to(device)
        condition1 = condition1.to(torch.float)
        ntest = test_data.shape[0]
        h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        fake1 = gen(condition1,h0,c0)
        #rmse = torch.sqrt(torch.mean((fake-real)**2))
        #mae = torch.mean(torch.abs(fake-real))
    #print("RMSE: ", rmse)
    #print("MAE: ",mae)
    b1 = fake1[0,:,0]
    mn1 = b1
    real1 = test_data[:,-1]
    rl1 = real1.squeeze()

    rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
    mae1 = torch.mean(torch.abs(mn1-rl1))
    #print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE'] = rmse1.item()
    dt['MAE'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)
    PnL1 = getPnL(ft1,rl1,ntest)
    #print("PnL in bp", PnL)
    PnLs = 10000 * np.sign(np.array(ft1.detach())) * np.array(rl1.detach())
    PnLd = np.zeros(int(0.5*len(PnLs)))
    PnL_even = np.zeros(int(0.5*len(PnLs)))
    PnL_odd = np.zeros(int(0.5*len(PnLs)))
    for i1 in range(len(PnLd)):
        PnLd[i1] = PnLs[2*i1] + PnLs[2*i1+1]
        PnL_even[i1] = PnLs[2*i1]
        PnL_odd[i1] = PnLs[2 * i1 + 1]
    PnL1 = np.mean(PnLd)
    #print("PnL in bp", PnL)
    dt['PnL_m test'] = np.mean(PnLd)
    PnL_test = PnLd

    dt['SR_m scaled test'] = np.sqrt(252) * np.mean(PnLd) / np.std(PnLd)


    print("Annualised (test) SR_m: ", np.sqrt(252) * np.mean(PnLd) / np.std(PnLd))
        
    if (ntest % 2) == 0:
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    else:
        dt['Open-to-Close SR_w'] = np.sqrt(252) * np.mean(PnL_even) / np.std(PnL_even)
        dt['Close-to-Open SR_w'] = np.sqrt(252) * np.mean(PnL_odd) / np.std(PnL_odd)
    means = np.array(mn1.detach())
    reals = np.array(rl1.detach())
    dt['Corr'] = np.corrcoef([means,reals])[0,1]
    print('Correlation ', np.corrcoef([means,reals])[0,1])
    dt['Pos mn'] = np.sum(means >0)/ len(means)
    dt['Neg mn'] = np.sum(means <0)/ len(means)
    ntest = val_data.shape[0]
    gen.eval()
    with torch.no_grad():
        condition1 = val_data[:,0:l]
        condition1 = condition1.unsqueeze(0)
        condition1 = condition1.to(device)
        condition1 = condition1.to(torch.float)
        ntest = val_data.shape[0]
        h0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        c0 = torch.zeros((1,ntest,hid_g),device=device,dtype=torch.float)
        fake1 = gen(condition1,h0,c0)
        
        #rmse = torch.sqrt(torch.mean((fake-real)**2))
        #mae = torch.mean(torch.abs(fake-real))
    #print("RMSE: ", rmse)
    #print("MAE: ",mae)
    b1 = fake1[0,:,0]
    mn1 = b1
    real1 = val_data[:,-1]
    rl1 = real1.squeeze()
    rmse1 = torch.sqrt(torch.mean((mn1-rl1)**2))
    mae1 = torch.mean(torch.abs(mn1-rl1))
    #print("RMSE: ",rmse,"MAE: ",mae)
    dt['RMSE val'] = rmse1.item()
    dt['MAE val'] = mae1.item()
    ft1 = mn1.clone().detach().to(device)
    PnLs = 10000 * np.sign(np.array(ft1.detach())) * np.array(rl1.detach())
    PnLd = np.zeros(int(0.5*len(PnLs)))
    for i1 in range(len(PnLd)):
        PnLd[i1] = PnLs[2*i1] + PnLs[2*i1+1]
    PnL1 = np.mean(PnLd)
    #print("PnL in bp", PnL)
    dt['PnL_m val'] = PnL1

    dt['SR_m scaled val'] = np.sqrt(252) * np.mean(PnLd) / np.std(PnLd)

    
    
    print("Annualised (val) SR_m : ", np.sqrt(252 * freq) * getSR(ft1,rl1).item())
    means = np.array(mn1.detach())
    reals = np.array(rl1.detach())
    dt['Corr val'] = np.corrcoef([means,reals])[0,1]
    dt['Pos mn val'] = np.sum(means >0)/ len(means)
    dt['Neg mn val'] = np.sum(means <0)/ len(means)
    df_temp = pd.DataFrame(data=dt,index=[0])
    return df_temp, PnL_test, PnL_even, PnL_odd



def TrainLoopnLSTMPnL(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the PnL loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            
            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            gen_loss = SqLoss - alpha * PnL
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:

        
        plt.figure("LSTM loss PnL")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTMPnLSTD(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the PnL, STD loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            
            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            STD = torch.std(PnL_s)
            gen_loss = SqLoss - alpha * PnL + delta * STD
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:

        
        plt.figure("LSTM loss PnL STD")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTMPnLSR(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the PnL,SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            
            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            PnL = torch.mean(PnL_s)
            SR = torch.mean(PnL_s) / torch.std(PnL_s)
            gen_loss = SqLoss - alpha * PnL - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:
    
        plt.figure("LSTM loss PnL SR")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTMSR(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the SR loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            
            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            SR = torch.mean(PnL_s) / torch.std(PnL_s)
            gen_loss = SqLoss - gamma * SR
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:

        
        plt.figure("LSTM loss SR")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTMSTD(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM-Fin with the STD loss
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            
            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            
            sign_approx = torch.tanh(tanh_coeff * ft)
            PnL_s  = sign_approx * rl
            STD = torch.std(PnL_s)
            gen_loss = SqLoss + delta * STD
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:

        
        plt.figure("LSTM loss STD")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def TrainLoopnLSTM(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lr_d = 0.0001, lr_g = 0.0001, h = 1, l = 10, pred = 1, diter =1, tanh_coeff = 100, device = 'cpu', plot = False):
    """
    Training loop for LSTM
    """
    ntrain = train_data.shape[0]
    nval = validation_data.shape[0]
    nbatches = ntrain//batch_size+1
    genloss = [False] * (nbatches*n_epochs)

    fake_and_condition = False
    real_and_condition = False

    totlen = train_data.shape[0]


    #currstep = 0

    #train the discriminator more

    PnL_best = 0
    SR_best = 0
    checkpoint_last_epoch = 0

    gen.train()

    for epoch in tqdm(range(n_epochs)):
        perm = torch.randperm(ntrain)
        train_data = train_data[perm,:]
        #shuffle the dataset for the optimisation to work
        for i in range(nbatches):
            curr_batch_size = batch_size
            if i==(nbatches-1):
                curr_batch_size = totlen-i*batch_size
            h_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            c_0d = torch.zeros((1,curr_batch_size,hid_d),device=device,dtype= torch.float)
            h_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)
            c_0g = torch.zeros((1,curr_batch_size,hid_g),device=device,dtype= torch.float)

            condition = train_data[(i*batch_size):(i*batch_size+curr_batch_size),0:l]
            condition = condition.unsqueeze(0)
            real = train_data[(i*batch_size):(i*batch_size+curr_batch_size),l:(l+pred)]
            real = real.unsqueeze(0)


            ### Update discriminator ###
            # Zero out the discriminator gradients
            
            # Update generator
            # Zero out the generator gradients
            gen_opt.zero_grad()


            fake = gen(condition,h_0g,c_0g)
            
            #fake1 = fake1.unsqueeze(0).unsqueeze(2)
            
            ft = fake.squeeze(0).squeeze(1)
            rl = real.squeeze(0).squeeze(1)
            SqLoss = (torch.norm(ft-rl)**2) / curr_batch_size
            
            gen_loss = SqLoss
            gen_loss.backward()
            gen_opt.step()
            gloss = gen_loss.detach().item()
            genloss[epoch*nbatches+i] = gloss

    if plot:

        
        plt.figure("LSTM loss")
        plt.title("LSTMloss")
        plt.plot(range(len(genloss)),genloss)
        plt.show()

    # SR_best = SR_best * np.sqrt(252)
    print("Training completed")
    # print("PnL val (best):", PnL_best)
    return gen, gen_opt

def LSTM_combos(ticker,loc,modelsloc,plotsloc,dataloc, etflistloc,  vl_later = True, lrg = 0.0001, lrd = 0.0001, n_epochs = 500, ngrad = 100, h = 1, l = 10, pred = 1, ngpu = 1, tanh_coeff = 100, tr = 0.8, vl = 0.1, z_dim = 32, hid_d = 64, hid_g = 8, checkpoint_epoch = 20, batch_size = 100, diter = 1, plot = False, freq = 2):
    """
    Training and evaluation on (test and val) of LSTM and LSTM-Fin
    """
    #initialise the networks first:
    datastart = {'lrd':[],'lrg':[],'epochs':[],'SR_val':[]}
    results_df = pd.DataFrame(data=datastart)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if ticker[0] == 'X':  
        train_data,val_data,test_data, dates_dt = split_train_val_testraw(ticker, dataloc, tr, vl, h, l, pred, plotcheck = False)
    else:
        train_data,val_data,test_data, dates_dt = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plotcheck = False)
    data_tt = torch.from_numpy(train_data)
    train_data = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(test_data)
    test_data = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(val_data)
    validation_data = data_tt.to(torch.float).to(device)
    
    condition_size = l
    target_size = pred
    ref_mean = torch.mean(train_data[0:batch_size,:])
    ref_std = torch.std(train_data[0:batch_size,:])

    gen = LSTM(noise_dim = 0,cond_dim=condition_size, hidden_dim=hid_g,output_dim=pred,mean =ref_mean,std=ref_std)
    gen.to(device)
    criterion = False

    PnL_test = [False] * 6
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lrg)

    gen, gen_opt,  alpha, beta, gamma, delta = GradientCheckLSTM(ticker, gen, gen_opt, ngrad, train_data,batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)

    f_name = modelsloc + ticker + "-LSTM-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    f_name1 = ticker + "-LSTM-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    
    print("PnL")
    losstype = "PnL"
    genPnL,  gen_optPnL = TrainLoopnLSTMPnL(gen,  gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnL.state_dict()}, f_name + "PnL_lstm_checkpoint.pth")
    df_temp, PnL_test[0], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genPnL,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    
    ntest = test_data.shape[0]
    pd.DataFrame(PnL_test[0]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
    plt.figure("Cummulative PnL "+ticker)
    plt.title("Cummulative PnL "+ticker)
    plt.grid(b = True)
    plt.xlabel("date")
    plt.xticks(rotation=45)
    plt.ylabel("bpts")
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[0]), label = "PnL")
    plt.legend(loc='best')
    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.title("Intraday cummulative PnL "+ticker)
        plt.grid(b=True)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        
        plt.ylabel("bpts")
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = "PnL")
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.title("Overnight cummulative PnL "+ticker)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.grid(b=True)
        
        plt.ylabel("bpts")
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = "PnL")
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.title("Overnight cummulative PnL "+ticker)
        plt.grid(b=True)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.ylabel("bpts")
        
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = "PnL")
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.title("Intraday cummulative PnL "+ticker)
        plt.xlabel("date")
        plt.xticks(rotation=45)
        plt.grid(b=True)
        plt.ylabel("bpts")
        
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = "PnL")
        plt.legend(loc='best')
    
    print("PnL STD")
    losstype = "PnL STD"
    genPnLMSESTD, gen_optPnLMSESTD = TrainLoopnLSTMPnLSTD(gen,  gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSESTD.state_dict()}, f_name + "PnLMSESTD_lstm_checkpoint.pth")
    df_temp, PnL_test[1], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genPnLMSESTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL STD LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[1]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
    
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[1]), label = losstype)
    plt.legend(loc='best')
    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    
    print("PnL SR")
    losstype = "PnL SR"
    genPnLMSESR, gen_optPnLMSESR = TrainLoopnLSTMPnLSR(gen, gen_opt,  criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSESR.state_dict()}, f_name + "PnLMSESR_lstm_checkpoint.pth")
    df_temp, PnL_test[2], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genPnLMSESR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL SR LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[2]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
    
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[1]), label = losstype)
    plt.legend(loc='best')
    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    
    
    print("STD")
    losstype = "STD"
    genPnLSR, gen_optPnLMSESR = TrainLoopnLSTMSTD(gen, gen_opt,  criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "STD_lstm_checkpoint.pth")
    df_temp, PnL_test[3], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genPnLSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "STD LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[3]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
    
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[1]), label = losstype)
    plt.legend(loc='best')
    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    
    
    print("SR")
    losstype = "SR"
    genSR, gen_optSR = TrainLoopnLSTMSR(gen,  gen_opt , criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "SR_lstm_checkpoint.pth")
    df_temp, PnL_test[4], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR LSTM", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[4]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[1]), label = losstype)
    plt.legend(loc='best')
    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
    
    
    print("MSE")
    losstype = "MSE"
    genMSE, gen_optMSE = TrainLoopnLSTM(gen, gen_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genMSE.state_dict()}, f_name + "MSE_lstm_checkpoint.pth")
    df_temp, PnL_test[5], PnL_even, PnL_odd = Evaluation2LSTM(ticker,freq,genMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    pd.DataFrame(PnL_test[5]).to_csv(loc+"PnLs/"+ticker+"-LSTM-"+losstype+".csv")
    plt.figure("Cummulative PnL "+ticker)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_test[1]), label = losstype)
    plt.legend(loc='best')
    plt.savefig(plotsloc+"LSTM-"+ticker+"-cummulativePnL.png")
    
    if (test_data.shape[0] % 2 == 0):
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        plt.savefig(plotsloc+"LSTM-"+ticker+"-intradayPnL.png")
        
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
        plt.savefig(plotsloc+"LSTM-"+ticker+"-overnight.png")
    else:
        plt.figure("Overnight cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_odd), label = losstype)
        plt.legend(loc='best')
        plt.savefig(plotsloc+"LSTM-"+ticker+"-overnight.png")
        
        plt.figure("Intraday cummulative PnL "+ticker)
        plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnL_even), label = losstype)
        plt.legend(loc='best')
        plt.savefig(plotsloc+"LSTM-"+ticker+"-intradayPnL.png")
    
    corrm = np.corrcoef(PnL_test)

    return results_df, corrm

def FinGAN_universal(tickers1, other,loc,modelsloc,plotsloc,dataloc, etflistloc,  vl_later = True, lrg = 0.0001, lrd = 0.0001, n_epochs = 500, ngrad = 100, h = 1, l = 10, pred = 1, ngpu = 1, tanh_coeff = 100, tr = 0.8, vl = 0.1, z_dim = 32, hid_d = 64, hid_g = 8, checkpoint_epoch = 20, batch_size = 100, diter = 1, plot = False, freq = 2):
    """
    FinGAN loss combos in the universal setting
    """
    #initialise the networks first:
    datastart = {'lrd':[],'lrg':[],'epochs':[],'SR_val':[]}
    results_df = pd.DataFrame(data=datastart)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    ticker = tickers1[0]
    train_data,val_data,test_data, dates_dt = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plot)
    ntr = train_data.shape[0]
    nvl = val_data.shape[0]
    ntest = test_data.shape[0]
    n_tickers1 = len(tickers1)
    n_tickers = len(tickers1) + len(other)
    train_data = np.zeros((ntr * n_tickers1, l + pred))
    validation_data = [False] * n_tickers
    test_data = [False] * n_tickers
    for i in range(n_tickers1):
        ticker = tickers1[i]
        if ticker[0] == "X":
            train,val,test, _ = split_train_val_testraw(ticker, dataloc,  tr, vl, h, l, pred, plot)
        else:
            train,val,test, _ = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plot)
        data_tt = torch.from_numpy(test)
        test_data[i] = data_tt.to(torch.float).to(device)
        train_data[i*ntr:(i+1)*ntr] = train
        data_tt = torch.from_numpy(val)
        validation_data[i] = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(train_data)
    train_data = data_tt.to(torch.float).to(device)
    for i in range(len(other)):
        ticker = tickers1[i]
        _,val,test, _ = split_train_val_test(ticker, dataloc, etflistloc,  tr, vl, h, l, pred, plot)
        data_tt = torch.from_numpy(test)
        test_data[i + n_tickers1] = data_tt.to(torch.float).to(device)
        data_tt = torch.from_numpy(val)
        validation_data[i + n_tickers1] = data_tt.to(torch.float).to(device)
    
    tickers = np.concatenate((tickers1,other))
    condition_size = l
    target_size = pred
    ref_mean = torch.mean(train_data[0:batch_size,:])
    ref_std = torch.std(train_data[0:batch_size,:])
    discriminator_indim = condition_size+target_size

    gen = Generator(noise_dim=z_dim,cond_dim=condition_size, hidden_dim=hid_g,output_dim=pred,mean =ref_mean,std=ref_std)
    gen.to(device)

    disc = Discriminator(in_dim=discriminator_indim, hidden_dim=hid_d,mean=ref_mean,std=ref_std)
    disc.to(device)

    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=lrd)

    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    gen, disc, gen_opt, disc_opt, alpha, beta, gamma, delta = GradientCheck(ticker, gen, disc, gen_opt, disc_opt, criterion, ngrad, train_data,batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)

    f_name = modelsloc +  "vuniversal-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    f_name1 = ticker + "-universal-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    
    PnLs_test = [False] * 10
    PnLs_val = [False] * 10
    means_test = [False] * 10
    means_val = [False] * 10
    print("PnL")
    losstype = "PnL"
    genPnL, discPnL, gen_optPnL, disc_optPnL = TrainLoopMainPnLnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnL.state_dict()}, f_name + "PnL_generator_checkpoint.pth")
    df_temp, PnLs_test[0], PnLs_val[0], means_test[0], means_val[0] = Evaluation3(tickers,freq,genPnL,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.title("Portfolio cummulative PnL " )
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[0]),label=losstype)
    plt.grid(b=True)
    plt.ylabel("bpts")
    plt.legend(loc='best')

    
    print("PnL MSE")
    genPnLMSE, discPnLMSE, gen_optPnLMSE, disc_optPnLMSE = TrainLoopMainPnLMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSE.state_dict()}, f_name + "PnLMSE_generator_checkpoint.pth")
    df_temp, PnLs_test[1], PnLs_val[1], means_test[1], means_val[1] = Evaluation3(tickers,freq,genPnLMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL MSE"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[1]),label=losstype)
    plt.legend(loc='best')
    
    print("PnL MSE STD")
    genPnLMSESTD, discPnLMSESTD, gen_optPnLMSESTD, disc_optPnLMSESTD = TrainLoopMainPnLMSESTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSESTD.state_dict()}, f_name + "PnLMSESTD_generator_checkpoint.pth")
    df_temp, PnLs_test[2], PnLs_val[2], means_test[2], means_val[2]= Evaluation3(tickers,freq,genPnLMSESTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE STD", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL MSE STD"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[2]),label=losstype)
    plt.legend(loc='best')
    
    print("PnL MSE SR")
    genPnLMSESR, discPnLMSESR, gen_optPnLMSESR, disc_optPnLMSESR = TrainLoopMainPnLMSESRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLMSESR.state_dict()}, f_name + "PnLMSESR_generator_checkpoint.pth")
    df_temp, PnLs_test[3], PnLs_val[3], means_test[3], means_val[3] = Evaluation3(tickers,freq,genPnLMSESR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL MSE SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL MSE SR"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[3]),label=losstype)
    plt.legend(loc='best')
    
    print("PnL SR")
    genPnLSR, discPnLSR, gen_optPnLSR, disc_optPnLSR = TrainLoopMainPnLSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "PnLSR_generator_checkpoint.pth")
    df_temp, PnLs_test[4], PnLs_val[4], means_test[4], means_val[4] = Evaluation3(tickers,freq,genPnLSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL SR"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[4]),label=losstype)
    plt.legend(loc='best')
    
    print("PnL STD")
    genPnLSTD, discPnLSTD, gen_optPnLSTD, disc_optPnLSTD = TrainLoopMainPnLSTDnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "PnLSTD_generator_checkpoint.pth")
    df_temp, PnLs_test[5], PnLs_val[5], means_test[5], means_val[5] = Evaluation3(tickers,freq,genPnLSTD,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "PnL STD", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "PnL STD"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[5]),label=losstype)
    plt.legend(loc='best')
    
    print("SR")
    genSR, discSR, gen_optSR, disc_optSR = TrainLoopMainSRnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genPnLSR.state_dict()}, f_name + "SR_generator_checkpoint.pth")
    df_temp, PnLs_test[6], PnLs_val[6], means_test[6], means_val[6] = Evaluation3(tickers,freq,genSR,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "SR"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[6]),label=losstype)
    plt.legend(loc='best')
    
    print("SR MSE")
    genSRMSE, discSRMSE, gen_optSRMSE, disc_optSRMSE = TrainLoopMainSRMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genSRMSE.state_dict()}, f_name + "SRMSE_generator_checkpoint.pth")
    df_temp, PnLs_test[7], PnLs_val[7], means_test[7], means_val[7] = Evaluation3(tickers,freq,genSRMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "SR MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "SR MSE"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[7]),label=losstype)
    plt.legend(loc='best')
    
    print("MSE")
    genMSE, discMSE, gen_optMSE, disc_optMSE = TrainLoopMainMSEnv(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genMSE.state_dict()}, f_name + "MSE_generator_checkpoint.pth")
    df_temp, PnLs_test[8], PnLs_val[8], means_test[8], means_val[8] = Evaluation3(tickers,freq,genMSE,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "MSE", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "MSE"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[8]),label=losstype)
    plt.legend(loc='best')

    print("ForGAN")
    genFG, discFG, gen_optFG, disc_optFG = TrainLoopForGAN(gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data[0], batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, pred, diter, tanh_coeff, device, plot)
    torch.save({'g_state_dict': genFG.state_dict()}, f_name + "ForGAN_generator_checkpoint.pth")
    df_temp, PnLs_test[9], PnLs_val[9], means_test[9], means_val[9] = Evaluation3(tickers,freq,genFG,test_data,validation_data,h,l,pred,hid_d,hid_g, z_dim, lrg, lrd, n_epochs, "ForGAN", 0, device, plotsloc, f_name1)
    results_df = pd.concat([results_df,df_temp], ignore_index=True)
    losstype = "BCE"
    plt.figure(" portfolio cumPnL- "+ f_name)
    plt.plot(dates_dt[-int(ntest/2):], np.cumsum(PnLs_test[9]),label=losstype)
    plt.legend(loc='best')
    plt.savefig(plotsloc+"UniversalPnLCumm.png")
    
    return results_df, PnLs_test, PnLs_val, means_test, means_val