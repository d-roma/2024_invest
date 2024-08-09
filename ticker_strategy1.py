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
    return data

if __name__ == '__main__':
    tick = yf.Ticker('VOO', )
    #period = '5y'
    period = 'max'

    ### Benchmark

    df_original = tick.history(period=period, interval="1d",
                      start=None, end=None, )

    ### Buy-and-hold results

    print("###### Buy & hold results")
    df = df_original.copy()

    bh_result = 100 * (df.iloc[-1].Close - df.iloc[0].Open) / df.iloc[0].Open

    # Calculate daily returns
    df['Monthly Return'] = df['Close'].pct_change()
    # Calculate cumulative returns
    df['Cumulative Return'] = (1 + df['Monthly Return']).cumprod()
    # Calculate the cumulative maximum return
    df['Cumulative Max'] = df['Cumulative Return'].cummax()
    # Calculate the drawdown
    df['Drawdown'] = (df['Cumulative Return'] - df['Cumulative Max']) / df['Cumulative Max']

    # Calculate the maximum drawdown
    bh_max_drawdown = 100 * df['Drawdown'].min()

    print("Benefit:\t\t", "%.2f" % bh_result, " %")
    print("Max. drawdown:\t", "%.2f" % bh_max_drawdown, " %")

    ### Ideal enter-quit results

    print("###### Ideal enter & quit")
    df = df_original.copy()

    n = 4 * 30
    df['buy'] = df.iloc[argrelextrema(df.Low.values, np.less_equal,
                                      order=n)[0]]['Low']
    df['sell'] = df.iloc[argrelextrema(df.High.values, np.greater_equal,
                                       order=n)[0]]['High']

    df.loc[:,'RSI'] = compute_rsi(df, window=14)
    df.loc[:,'MA50'] = df['Close'].rolling(window=int(50), min_periods=1).mean()
    df.loc[:,'MA100'] = df['Close'].rolling(window=int(100), min_periods=1).mean()
    df.loc[:,'MA200'] = df['Close'].rolling(window=int(200), min_periods=1).mean()

    #df_plt = df[["Open", "High", "Low", "Close", "Volume", "buy", "sell"]]  # Select required columns
    df_plt = df
    df_plt.index.name = "Date"  # Set the index name
    df_plt = df_plt.reset_index()  # Reset the index
    df_plt["Date"] = pd.to_datetime(df_plt["Date"])  # Convert the Date column to datetime format
    df_plt_candle = df_plt.set_index("Date")
    apds = []
    apd = mpf.make_addplot(df_plt_candle['buy'], type='scatter', markersize=50, color='darkgreen',
                           marker="*")
    apds.append(apd)
    apd = mpf.make_addplot(df_plt_candle['MA50'], type='line', color='lightgreen')
    apds.append(apd)
    apd = mpf.make_addplot(df_plt_candle['MA100'], type='line', color='lightyellow')
    apds.append(apd)
    apd = mpf.make_addplot(df_plt_candle['MA200'], type='line', color='lightcoral')
    apds.append(apd)
    apd = mpf.make_addplot(df_plt_candle['sell'], type='scatter', markersize=50, color='darkred',
                           marker="*")
    apds.append(apd)
    mpf.plot(df, type='candle', addplot=apds, volume=True)
    mpf.show()

    buy_series = df['buy'].dropna()
    sell_series = df['sell'].dropna()

    # Ensure first buy goes prior first sell
    b_index = buy_series.index[0]
    for s_index, s_item in sell_series.items():
        if s_index < b_index:
            sell_series = sell_series[1, :]
        else:
            break

    pd_buy = pd.DataFrame(buy_series.rename("value"), )
    pd_buy["type"] =  1
    pd_sell = pd.DataFrame(sell_series.rename("value"), )
    pd_sell["type"] =  0
    orders = pd.concat((pd_buy, pd_sell)).sort_index()
    # Compute gain, guarantying order
    increments = []
    p_buy_val = None
    p_sell_val = None
    p_type = None
    for b_index, b_item in orders.iterrows():
        if p_type == b_item.type:
            continue
        if b_item.type == 1:
            p_buy_val = b_item.value
            p_buy_idx = b_index
        if b_item.type == 0:
            increment = (b_item.value - p_buy_val) / p_buy_val
            increments.append((increment, p_buy_idx, b_index))
        p_type = b_item.type

    print(increments)
    increment_values = np.array([x[0] for x in increments])
    cummulative_increments = np.cumprod(1 + increment_values)
    time_span = df.index[-1] - df.index[0]

    print("Benefit:\t\t", "%.2f" % (cummulative_increments[-1]*100), " %")
    print("Annualized benefit:\t", "%.2f" % ((cummulative_increments[-1]*100)/(time_span.days/365)), " %")

    print("Max. drawdown:\t 0 by definition")

    ### Strategy 1

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


    ### Entra quant retroces FIBO 50%

    ### Entrar quant es desinverteixen totes les mitges (MA50 > MA200), sortir quan es perd la mitja de 200 del tot (maxim per sota MA200).