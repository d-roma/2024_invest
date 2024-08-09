import matplotlib
import yfinance as yf

matplotlib.use('QtAgg')

from matplotlib import pyplot as plt

import numpy as np

if __name__ == '__main__':
    tick = yf.Ticker('SPY', )
    # period = '5y'
    period = 'max'

    ### Strategy 2

    print("###### Strategy 2 - MA")
    df = tick.history(period=period, interval="1d",
                      start=None, end=None, )

    df.loc[:, 'MA50'] = df['Close'].rolling(window=int(50), min_periods=1).mean()
    df.loc[:, 'MA100'] = df['Close'].rolling(window=int(100), min_periods=1).mean()
    df.loc[:, 'MA200'] = df['Close'].rolling(window=int(200), min_periods=1).mean()

    df = df.iloc[30:]

    df["buy_signal"] = 0
    df["sell_signal"] = 0
    df["status"] = 0

    status = 0  # 0 out, 1 MA inversion, 2 MA desinversio -- entrar
    p_max = 0
    drawdown = 0
    for index, row in df.iterrows():
        data = row.Close
        # If we are out of the market
        if status == 0:
            # Wait for mean inversion
            if row['MA50'] < row['MA200']:
                status = 1
        # Wait for mean desinversio
        elif status == 1:
            if row['MA200'] < row['MA50']:
                status = 2
                df.loc[index, "buy_signal"] = 1
                p_max = 0
        # We are in the market
        else:
            if data > p_max:
                p_max = data
            c_drawdown = (data - p_max) / p_max
            if c_drawdown < drawdown:
                drawdown = c_drawdown
            if (row.High < row['MA200']):
                status = 0
                df.loc[index, "sell_signal"] = 1
        df.loc[index, "status"] = status

    plt.figure()
    plt.plot(df.index, df.Close, label='Close')
    plt.plot(df.index, df.MA200, label="MA200")
    plt.plot(df.index, df.MA50, label="MA50")
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

    ### Entra quant retroces FIBO 50%

    ### Entrar quant es desinverteixen totes les mitges (MA50 > MA200), sortir quan es perd la mitja de 200 del tot (maxim per sota MA200).
