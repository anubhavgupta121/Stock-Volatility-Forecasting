import numpy as np
import pandas as pd


def creater_features(data : pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(data["Close"]["^GSPC"]).diff().iloc[1:]
    sq_returns = np.square(log_returns)
    target = np.sqrt(sq_returns.rolling(5).mean()).shift(-5).iloc[:log_returns.shape[0] - 5]

    df = target.rename("target").to_frame()
    df["past_5_days"] = np.sqrt(sq_returns.rolling(5).mean())
    df["past_22_days"] = np.sqrt(sq_returns.rolling(22).mean())
    df["today_vol"] = np.abs(log_returns)   
    
    return df.dropna()

