import numpy as np

import matplotlib
matplotlib.use('QtAgg')
from matplotlib import pyplot as plt

import yfinance as yf
import pandas as pd
### Current version requires to comment line 86 of file pandas_ta\utils\data\yahoofinance.py
### as follows
"""
if Imports["yfinance"] and ticker is not None:
    import yfinance as yfra
    # yfra.pdr_override()
"""
### https://twopirllc.github.io/pandas-ta/
import pandas_ta as ta


if __name__ == '__main__':
    #period = '5y'
    period = 'max'

    print("###### Strategy 3 - SAR & MA")

    df = pd.DataFrame()
    df = df.ta.ticker("SPY", period=period, interval="1d",
                      start=None, end=None, )

    df.ta.sma(length=50, append=True)
    df.ta.sma(length=100, append=True)
    df.ta.sma(length=200, append=True)

    # df.ta.psar(af0=0.001, af=0.005, max_af=0.005, append=True) 1523%, -19%
    df.ta.psar(af0=0.001, af=0.005, max_af=0.005, append=True)

    PSARl_name = [i for i in df.columns if i.startswith('PSARl_')][0]
    PSARs_name = [i for i in df.columns if i.startswith('PSARs_')][0]

    df.rename(columns={PSARl_name: 'PSARl', PSARs_name: 'PSARs' }, inplace=True)

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
            if not np.isnan(row["PSARl"]) and (row.Low > row['SMA_200']):
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
            if not np.isnan(row["PSARs"]) or (row.High < row['SMA_200']):
                status = 0
                df.loc[index, "sell_signal"] = 1
        df.loc[index, "status"] = status

    plt.figure()
    plt.plot(df.index, df.Close, label='Close')
    plt.plot(df.index, df.SMA_200, label="SMA_200")
    plt.plot(df.index, df["PSARl"], label="SAR_Buy", marker='o', linestyle='None', markersize=1)
    plt.plot(df.index, df["PSARs"], label="SAR_Sell", marker='o', linestyle='None', markersize=1)
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