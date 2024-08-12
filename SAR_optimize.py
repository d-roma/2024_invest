import matplotlib

matplotlib.use('QtAgg')
from matplotlib import pyplot as plt

import numpy as np

import pandas as pd
import pandas_ta as ta
### Current version requires to comment line 86 of file pandas_ta\utils\data\yahoofinance.py
### as follows
"""
if Imports["yfinance"] and ticker is not None:
    import yfinance as yfra
    # yfra.pdr_override()
"""
### https://twopirllc.github.io/pandas-ta/


def SAR_benchmark(df, af, max_af, af0=0.001):
    df.ta.psar(af0=af0, af=af, max_af=max_af, append=True)

    PSARl_name = [i for i in df.columns if i.startswith('PSARl_')][0]
    PSARs_name = [i for i in df.columns if i.startswith('PSARs_')][0]

    df.rename(columns={PSARl_name: 'PSARl', PSARs_name: 'PSARs'}, inplace=True)

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
            if not np.isnan(row["PSARl"]):
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
            # if not np.isnan(row["PSARs"]) or (row.High < row['SMA_200']):
            if not np.isnan(row["PSARs"]):
                status = 0
                df.loc[index, "sell_signal"] = 1
        df.loc[index, "status"] = status

    # Add a "fake" sell signal at the end to consider gain up to current date
    df.loc[index, "sell_signal"] = 1

    index_buy = df.index[np.where(df["buy_signal"] > 0)]
    index_sell = df.index[np.where(df["sell_signal"] > 0)]

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

    st1_inc = np.array([x[0] for x in increments])
    st1_cummulative_increments = np.cumprod(1 + st1_inc)

    time_span = df.index[-1] - df.index[0]
    benefit = st1_cummulative_increments[-1] * 100
    return benefit, drawdown


if __name__ == '__main__':
    # period = '5y'
    period = 'max'

    print("###### Strategy 3 - SAR Optimization")

    df = pd.DataFrame()
    df = df.ta.ticker("SPY", period=period, interval="1d",
                      start=None, end=None, )

    df.ta.sma(length=50, append=True)
    df.ta.sma(length=100, append=True)
    df.ta.sma(length=200, append=True)

    af_array = np.linspace(0.0001, 0.005, 10)
    max_af_array = np.linspace(0.005, 0.1, 10)
    af0_array = np.linspace(0.0001, 0.001, 10)
    results_benefit = np.empty((af_array.size * max_af_array.size))
    results_drawdown = np.empty((af_array.size * max_af_array.size))
    results_values = np.empty((af_array.size * max_af_array.size, 2))
    max_af = 0.01
    for i, af in enumerate(af_array):
        #for j, max_af in enumerate(max_af_array):
        for j, af0 in enumerate(af0_array):
            benefit, drawdown = SAR_benchmark(df.copy(), af, max_af, af0=af0)
            results_benefit[i * af_array.size + j] = benefit
            results_drawdown[i * af_array.size + j] = drawdown
            #results_values[i * af_array.size + j, :] = (af, max_af)
            results_values[i * af_array.size + j, :] = (af, af0)

    print(results_benefit.max())
    print(results_drawdown.max())
    print(results_values[results_benefit.argmax()])

    af_array = ["%.4f" % i for i in af_array]
    #max_af_array = ["%.3f" % i for i in max_af_array]
    af0_array = ["%.4f" % i for i in af0]

    fig, ax = plt.subplots()
    im = plt.imshow(results_benefit.reshape(len(af_array), len(max_af_array)))
    ax.set_xticks(np.arange(len(af_array)), labels=af_array)
    #ax.set_yticks(np.arange(len(max_af_array)), labels=max_af_array)
    ax.set_yticks(np.arange(len(af0_array)), labels=af0_array)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             rotation_mode="anchor")
    plt.colorbar()
    plt.xlabel("AF")
    #plt.ylabel("MAX_AF")
    plt.ylabel("AF0")
    plt.savefig("SAR_optimization_benefit.png", bbox_inches='tight')

    fig, ax = plt.subplots()
    im = plt.imshow(results_drawdown.reshape(len(af_array), len(max_af_array)))
    ax.set_xticks(np.arange(len(af_array)), labels=af_array)
    #ax.set_yticks(np.arange(len(max_af_array)), labels=max_af_array)
    ax.set_yticks(np.arange(len(af0_array)), labels=af0_array)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left",
             rotation_mode="anchor")
    plt.colorbar()
    plt.savefig("SAR_optimization_drawdown.png", bbox_inches='tight')

    print("Finished")
