# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:36:24 2023

@author: vulet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import FinGAN

data_ETFs = pd.read_csv("ETFs-data.csv")
data_ETFs['date_dt'] = pd.to_datetime(data_ETFs['date'])
data_ETFs['AdjClose'] = data_ETFs['PRC'] / data_ETFs['CFACPR']
data_ETFs['AdjOpen'] = data_ETFs['OPENPRC'] / data_ETFs['CFACPR']
ETF_list = data_ETFs['TICKER'].unique()[2:]

ETF_dfs = [None] * len(ETF_list)

for i in tqdm(range(len(ETF_list))):
    ETF = ETF_list[i]
    ETF_dfs[i] = data_ETFs[data_ETFs.TICKER == ETF].copy().reset_index()
    ETF_dfs[i].to_csv("C:\\Users\\vulet\\Documents\\FinGAN\\data\\"+ETF+".csv")
    plt.figure(ETF + "price")
    plt.plot(ETF_dfs[i]['date_dt'],ETF_dfs[i]['AdjOpen'],label='open')
    plt.plot(ETF_dfs[i]['date_dt'],ETF_dfs[i]['AdjClose'],label='close')
    plt.legend(loc='best')
    plt.xlabel("Date")
    plt.ylabel("Price in USD")
    plt.title(ETF + " adjusted price")
    plt.show()


data_stocks = pd.read_csv("Stocks-data.csv")
data_stocks['date_dt'] = pd.to_datetime(data_stocks['date'])
data_stocks['AdjClose'] = data_stocks['PRC'] / data_stocks['CFACPR']
data_stocks['AdjOpen'] = data_stocks['OPENPRC'] / data_stocks['CFACPR']
Stock_list = data_stocks['TICKER'].unique()

Stock_dfs = [None] * len(Stock_list)
for i in tqdm(range(len(Stock_list))):
    Ticker = Stock_list[i]
    Stock_dfs[i] = data_stocks[data_stocks.TICKER == Ticker].copy().reset_index()
    Stock_dfs[i].to_csv("C:\\Users\\vulet\\Documents\\FinGAN\\data\\"+Ticker+".csv")

stock_df = Stock_dfs[0]
stock_df['log'] = np.log(stock_df['AdjClose'])
stock_df['ret'] = stock_df['log'].diff()
plt.figure("ret")
plt.plot(stock_df['date_dt'],stock_df['ret'])
plt.xlabel("Date")
plt.ylabel("log return")
plt.title("log return")
plt.show()

dates_dt = stock_df['date_dt'].copy()[1:]
dataloc = "C:\\Users\\vulet\\Documents\\FinGAN\\data\\"
stock = 'AMZN'
etf = 'XLY'
e_ret, sret, etfret = FinGAN.exces_returns(dataloc, stock, etf, plotcheck = False)
plt.figure()
plt.plot(dates_dt,sret, alpha = 0.6, label = 'stock')
plt.plot(dates_dt,etfret, alpha = 0.6, label = 'etf')
plt.plot(dates_dt,e_ret, alpha = 0.6, label = 'excess return')
plt.legend()
plt.show()

etflistloc = "C:\\Users\\vulet\\Documents\\FinGAN\\stocks-etfs-list.csv"
ticker = "PEP"
etf = FinGAN.ETF_find(etflistloc,ticker)
e_ret, sret, etfret = FinGAN.excess_returns_plotcheck(dataloc, ticker, etf)
train_data,val_data,test_data, dates_train, dates_val, dates_test = FinGAN.split_train_val_test(ticker, dataloc, etflistloc, tr = 0.8, vl = 0.1, h = 1, l = 10, pred = 1, plotcheck=False)
