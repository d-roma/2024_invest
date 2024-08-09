import yfinance as yf
import mplfinance as mpf
import datetime
from scipy.signal import argrelextrema

import matplotlib
matplotlib.use('QtAgg')

import pandas as pd
from matplotlib import pyplot as plt

import numpy as np

# Define the Parabolic SAR calculation function -- ChatGPT revision pending
def compute_parabolic_sar(data, af=0.02, af_max=0.2):
    """
    Compute the Parabolic SAR for a given DataFrame with high, low, and close prices.

    Parameters:
    - data: DataFrame with columns 'High', 'Low', and 'Close'.
    - af: Acceleration Factor (default is 0.02).
    - af_max: Maximum Acceleration Factor (default is 0.2).

    Returns:
    - DataFrame with an additional 'SAR' column representing the Parabolic SAR.
    """
    high = data['High']
    low = data['Low']
    close = data['Close']

    sar = pd.Series(index=close.index)
    trend = 1  # Start with an uptrend (+1) or downtrend (-1)
    af = af  # Initial Acceleration Factor
    ep = high[0]  # Starting extreme point (high for uptrend)
    sar[0] = low[0]  # Initial SAR value

    for i in range(1, len(close)):
        if trend == 1:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], low[i - 1], low[i])  # Ensure SAR is below/at previous two lows
            if high[i] > ep:
                ep = high[i]
                af = min(af + 0.02, af_max)  # Increase af, but not above af_max
            if close[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = 0.02  # Reset af to initial value
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = max(sar[i], high[i - 1], high[i])  # Ensure SAR is above/at previous two highs
            if low[i] < ep:
                ep = low[i]
                af = min(af + 0.02, af_max)  # Increase af, but not above af_max
            if close[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = 0.02  # Reset af to initial value

    data['SAR'] = sar


if __name__ == '__main__':
    tick = yf.Ticker('SPY', )
    #period = '5y'
    period = 'max'

    ### Benchmark

    df_original = tick.history(period=period, interval="1d",
                      start=None, end=None, )

    print("###### Strategy 3 - SAR")
    df = tick.history(period=period, interval="1d",
                      start=None, end=None, )

    compute_parabolic_sar(df, af=0.02, af_max=0.2)

    df.loc[:, 'SAR_buy_signal'] = np.where(df['SAR'] < df['Close'], 1, 0)

    df.loc[:, 'MA50'] = df['Close'].rolling(window=int(50), min_periods=1).mean()
    df.loc[:, 'MA100'] = df['Close'].rolling(window=int(100), min_periods=1).mean()
    df.loc[:, 'MA200'] = df['Close'].rolling(window=int(200), min_periods=1).mean()

    df = df.iloc[30:]

    df["buy_signal"] = 0
    df["sell_signal"] = 0
    df["status"] = 0

    status = 0  # 0 out, 1 in
    p_max = 0
    drawdown = 0
    for index, row in df.iterrows():
        data = row.Close
        # If we are out of the market
        if status == 0:
            # Wait for SAR
            if row['SAR_buy_signal'] > 0:
                status = 1
                df.loc[index, "buy_signal"] = 1
                p_max = 0
        # We are in the market
        else:
            if data > p_max:
                p_max = data
            c_drawdown = (data - p_max) / p_max
            if c_drawdown < drawdown:
                drawdown = c_drawdown
            if row['SAR_buy_signal'] < 1:
                status = 0
                df.loc[index, "sell_signal"] = 1
        df.loc[index, "status"] = status

    plt.figure()
    plt.plot(df.index, df.Close, label='Close')
    plt.plot(df.index, df.MA200, label="MA200")
    plt.plot(df.index, df.SAR, label="sar", marker='o', linestyle='None', markersize=1)
    index_buy = df.index[np.where(df["buy_signal"] > 0)]
    plt.scatter(index_buy, df.Close[index_buy], marker="*", color="darkgreen", s=100)
    index_sell = df.index[np.where(df["sell_signal"] > 0)]
    plt.scatter(index_sell, df.Close[index_sell], marker="*", color="darkred", s=100)
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()

    # Compute gain, guarantying order
    i = 0
    increments = []
    for b_index in index_buy:
        if index_sell.size < i + 1:
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
    print("Benefit:\t\t", "%.2f" % (st1_cummulative_increments[-1] * 100), " %")
    print("Annualized benefit:\t", "%.2f" % ((st1_cummulative_increments[-1] * 100) / (time_span.days / 365)), " %")
    print("Max. drawdown:\t", "%.2f" % (100 * drawdown), " %")