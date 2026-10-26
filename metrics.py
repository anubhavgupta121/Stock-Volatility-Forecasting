import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def RMSE(pred: pd.Series, actual: pd.Series) -> float:

    both = pd.concat([pred, actual], axis=1, join="inner").dropna() #drop any non-matching dates
    errors = both.iloc[:, 0] - both.iloc[:, 1]
    return float(np.sqrt((errors ** 2).mean()))

def QLIKE(pred: pd.Series, actual: pd.Series) -> float:
    both = pd.concat([pred, actual], axis=1, join="inner").dropna()  # matching dates, no missing values
    assert (both.iloc[:, 0] > 0).all(), "predictions must be positive"
    ratio = (both.iloc[:, 1] / both.iloc[:, 0]) ** 2  # actual variance / predicted variance
    return float((ratio - np.log(ratio) - 1).mean())