import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import baselines,metrics,models,features


data = yf.download("^GSPC", start="1990-01-01", auto_adjust=True)


df = features.creater_features(data)


train = df.loc["1990" : "2018"]
val = df.loc["2019" : "2022"]
test = df.loc["2023" : ]





