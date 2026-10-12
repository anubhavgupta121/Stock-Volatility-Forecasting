import yfinance as yf
import matplotlib.pyplot as plt



data = yf.download("SPY", start="2007-01-01", auto_adjust=True)
import numpy as np

print(data.shape)