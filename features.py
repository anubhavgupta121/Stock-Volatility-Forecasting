import numpy as np
import pandas as pd

import yfinance as yf


vix_raw = yf.download("^VIX", start="1990-01-01", end="2026-10-01")
vix_close = vix_raw["Close"]["^VIX"]            


def creater_features(data : pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(data["Close"]["^GSPC"]).diff().iloc[1:]
    sq_returns = np.square(log_returns)
    target = np.sqrt(sq_returns.rolling(5).mean()).shift(-5).iloc[:log_returns.shape[0] - 5]
    hlrange = (np.log(data["High"]["^GSPC"]/data["Low"]["^GSPC"])**2)/(4*np.log(2))
    df = target.rename("target").to_frame()
    df["past_5_days"] = np.sqrt(sq_returns.rolling(5).mean())
    df["past_22_days"] = np.sqrt(sq_returns.rolling(22).mean())
    df["today_vol"] = np.abs(log_returns)   
    df["high_low_1"] = np.sqrt(hlrange)
    df["high_low_5"] = np.sqrt(hlrange.rolling(5).mean())
    df["high_low_22"] = np.sqrt(hlrange.rolling(22).mean())


    gap = np.log(data["Open"]["^GSPC"] / data["Close"]["^GSPC"].shift(1))
    df["overnight_gap"] = np.abs(gap)

    df["vix"] = vix_close / 100 / np.sqrt(252)

    df["neg_return"] = np.abs(np.minimum(log_returns, 0))  
    df["vol_ratio"] = df["past_5_days"] / df["past_22_days"] 



    return df.dropna()