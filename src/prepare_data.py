from utils.data import create_dataset_from_dir, read_stock_history
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def test_plot():
    data = pd.read_csv("c:/data/equity/price/FXI.csv")
    data.plot(title="FXI", fig=plt.gcf(), rot=30)
    plt.show()

if __name__ == '__main__':

    # test_plot()
    # exit(0)
    # end = datetime.strptime('2019-11-08', '%Y-%m-%d')
    # start = datetime.strptime('2004-10-08', '%Y-%m-%d')
    # print((end - start).days)
    # exit(0)
    # row = [1, 2, 3, 4, 5, 6]
    # print(list(map(float, row[1:5] +row[5:5])))
    # exit(0)
    results = create_dataset_from_dir('c:/data/equity/price', 'c:/data/equity/price/target_prices.h5',
                                      tickers=['FXI', 'SPY', 'EWZ', 'XLK', 'XLF'],
                                      start='2004-10-08', end='2019-12-30')

    data, tickers = read_stock_history('c:/data/equity/price/target_prices.h5')
    print(data.shape)
    # print(data[0])
    # print(data[1])
    # print(data[2])
    print(tickers)