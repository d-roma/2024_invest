import matplotlib
import mplfinance as mpf
import yfinance as yf
from scipy.signal import argrelextrema

matplotlib.use('QtAgg')

import pandas as pd

import numpy as np

if __name__ == '__main__':
    tick = yf.Ticker('SPY', )
    # period = '5y'
    period = 'max'

    ### Benchmark

    df_original = tick.history(period=period, interval="1d",
                               start=None, end=None, )

    ### Ideal enter-quit results

    print("###### Ideal enter & quit")
    df = df_original.copy()

    n = 4 * 30
    df['buy'] = df.iloc[argrelextrema(df.Low.values, np.less_equal,
                                      order=n)[0]]['Low']
    df['sell'] = df.iloc[argrelextrema(df.High.values, np.greater_equal,
                                       order=n)[0]]['High']

    df.loc[:, 'MA50'] = df['Close'].rolling(window=int(50), min_periods=1).mean()
    df.loc[:, 'MA100'] = df['Close'].rolling(window=int(100), min_periods=1).mean()
    df.loc[:, 'MA200'] = df['Close'].rolling(window=int(200), min_periods=1).mean()

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
    pd_buy["type"] = 1
    pd_sell = pd.DataFrame(sell_series.rename("value"), )
    pd_sell["type"] = 0
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

    print("Benefit:\t\t", "%.2f" % (cummulative_increments[-1] * 100), " %")
    print("Annualized benefit:\t", "%.2f" % ((cummulative_increments[-1] * 100) / (time_span.days / 365)), " %")

    print("Max. drawdown:\t 0 by definition")

    # df_plt = df[["Open", "High", "Low", "Close", "Volume", "buy", "sell"]]  # Select required columns
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
