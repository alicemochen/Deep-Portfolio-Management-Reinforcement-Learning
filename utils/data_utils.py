import numpy as np
import yfinance as yf


def get_ochl_tickers(tickers, start, end):
    df = yf.download(tickers, start, end, group_by="tickers")
    data = {}
    for ticker in tickers:
        ticker_df = df[ticker][['Open', 'Close', 'High', 'Low', 'Volume']]
        ticker_df[['Open', 'Close', 'High', 'Low']].fillna(method='ffill', inplace=True)
        ticker_df['Volume'].fillna(0, inplace=True)
        data[ticker] = ticker_df
    return data


# Last feature has to be daily rtn
def get_stock_features_default(tickers, start, end):
    tickers_data = get_ochl_tickers(tickers, start, end)
    feature_list = []
    for ticker in tickers:
        ticker_df = tickers_data[ticker]
        # today open to prev day open, yday close to open, yday high to open, yday low to open
        prev_open = np.array(ticker_df.Open[:-1])
        prev_close = ticker_df.Close[:-1]
        prev_high = ticker_df.High[:-1]
        prev_low = ticker_df.Low[:-1]
        open = np.array(ticker_df.Open[1:])
        ticker_features = np.array([prev_close/prev_open,
                                    prev_high / prev_open,
                                    prev_low/ prev_open,
                                    open/prev_open])
        feature_list.append(ticker_features)
    return np.stack(feature_list, axis=1)


if __name__=="__main__":
    X = get_ochl_tickers(["XLE", "XLP", "XLU", "XLV", "XLB", "XLRE", "XLI", "XLF"], "2022-01-01", "2022-07-01")
    print(X)


