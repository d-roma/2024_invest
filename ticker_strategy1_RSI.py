import yfinance as yf
import mplfinance as mpf
import datetime
from scipy.signal import argrelextrema

import matplotlib
matplotlib.use('QtAgg')

import pandas as pd
from matplotlib import pyplot as plt

import numpy as np

def compute_rsi(data, window=14):
    # Calculate the difference in price
    delta = data['Close'].diff()

    # Calculate gains and losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate the average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

if __name__ == '__main__':
    tick = yf.Ticker('SPY', )
    #period = '5y'
    period = 'max'

    ### Strategy 1 -- RSI
    # TODO Use weekly signals, apply on daily data

    print("###### Strategy 1")
    df = tick.history(period=period, interval="1wk",
                      start=None, end=None, )

    df.loc[:,'RSI'] = compute_rsi(df, window=14)
    df.loc[:,'MA50'] = df['Close'].rolling(window=int(50/5), min_periods=1).mean()
    df.loc[:,'MA100'] = df['Close'].rolling(window=int(100/5), min_periods=1).mean()
    df.loc[:,'MA200'] = df['Close'].rolling(window=int(200/5), min_periods=1).mean()

    df = df.iloc[30:]

    df.loc[:,'above_ma100'] = np.where((df['Close'] > df['MA100']), 1, 0)
    df.loc[:,'above_ma50'] = np.where((df['Close'] > df['MA50']), 1, 0)

    df["buy_signal"] = 0
    df["sell_signal"] = 0
    df["status"] = 0

    status = 0 # 0 out, 1 inside market
    p_rsi_below_40 = False
    p_row = None
    p_max = 0
    drawdown = 0
    for index, row in df.iterrows():
        data = row.Close
        # If we are out of the market
        if status == 0:
            # If above MA100 and RSI goes above 40
            if row['RSI'] > 60 and (data > row['MA50']):
                status = 1
                p_rsi_below_40 = False
                df.loc[index, "buy_signal"] = 1
                p_max = 0
            # If we are out of the market, and RSI below 40, set RSI low-threshold flag
            if row['RSI'] < 40:
                p_rsi_below_40 = True
        # If we are in the market
        else:
            if data > p_max:
                p_max = data
            c_drawdown = (data - p_max)/p_max
            if c_drawdown < drawdown:
                drawdown = c_drawdown
            if (row.High < row['MA200']):
                status = 0
                df.loc[index, "sell_signal"] = 1
        df.loc[index, "status"] = status


    fig, axs = plt.subplots(2, 1, layout='constrained')
    axs[0].plot(df.index, df.Close, label='Close')
    axs[0].plot(df.index, df.MA100, label="MA100")
    axs[0].plot(df.index, df.MA50, label="MA50")
    index_buy = df.index[np.where(df["buy_signal"] > 0)]
    axs[0].scatter(index_buy, df.Close[index_buy], marker="*", color="darkgreen", s=100)
    index_sell = df.index[np.where(df["sell_signal"] > 0)]
    axs[0].scatter(index_sell, df.Close[index_sell], marker="*", color="darkred", s=100)
    axs[0].legend(loc=0)
    axs[0].grid(True)
    axs[1].plot(df.index, df.RSI, )
    axs[1].grid(True)
    plt.show()

    # Compute gain, guarantying order
    i = 0
    increments = []
    for b_index in index_buy:
        if index_sell.size < i+1:
            break
        s_index = index_sell[i]
        b_val = df.loc[b_index, 'Close']
        s_val = df.loc[s_index, 'Close']
        increment = (s_val - b_val) / b_val
        increments.append((increment, b_index, s_index))
        i += 1

    print(increments)
    st1_inc = np.array([x[0] for x in increments])
    st1_cummulative_increments = np.cumprod(1 + st1_inc)

    time_span = df.index[-1] - df.index[0]
    print("Benefit:\t\t", "%.2f" % (st1_cummulative_increments[-1]*100), " %")
    print("Annualized benefit:\t", "%.2f" % ((st1_cummulative_increments[-1]*100)/(time_span.days/365)), " %")
    print("Max. drawdown:\t", "%.2f" % (100*drawdown), " %")

