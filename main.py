import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import baselines,metrics


data = yf.download("^GSPC", start="1990-01-01", auto_adjust=True)


log_returns = np.log(data["Close"]["^GSPC"]).diff().iloc[1:]


sq_returns = np.square(log_returns)
vol_of_5days = np.sqrt(sq_returns.rolling(5).mean()).shift(-5).iloc[:log_returns.shape[0] - 5]


train = vol_of_5days.loc["1990" : "2018"]
val = vol_of_5days.loc["2019" : "2022"]
test = vol_of_5days.loc["2023" : ]


