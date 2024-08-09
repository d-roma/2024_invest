import matplotlib
import yfinance as yf

matplotlib.use('QtAgg')

from matplotlib import pyplot as plt

if __name__ == '__main__':
    tick = yf.Ticker('SPY', )
    # period = '5y'
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

    plt.figure()
    plt.plot(df.index, 100 * (df.Close - df.iloc[0].Close) / df.iloc[0].Close)
    plt.ylabel("Relative change [%]")
    plt.grid(True)
    plt.show()
