import numpy as np
import pandas as pd
from arch import arch_model

def naive(returns: pd.Series, window: int = 5) -> pd.Series:
    """Naive forecast: next week's volatility = volatility of the past `window` days.

    The value on day t uses returns from days t-window+1 to t (today included),
    so it only uses information available after day t's close.
    """
    past_var = (returns ** 2).rolling(window).mean()  # average squared return over the past window
    return np.sqrt(past_var)
def rolling(returns: pd.Series, window: int = 22) -> pd.Series:
    '''naive with a larger window'''
    past_var = (returns ** 2).rolling(window).mean()  # average squared return over the past window
    return np.sqrt(past_var)

def ewm(returns: pd.Series) -> pd.Series:

    ewm_keep = (returns ** 2).ewm(alpha=0.06,adjust=False).mean()
    return np.sqrt(ewm_keep)

def garch(returns: pd.Series, cutoff: str) -> pd.Series:
    """GARCH(1,1) forecast of next-week volatility (typical daily move).

    Like EWMA, but reaction (alpha), memory (beta) and a long-run anchor (omega)
    are learned from data before `cutoff`, instead of being fixed.
    """
    r100 = returns * 100  # arch fits more reliably on values around 1
    model = arch_model(r100, mean="Zero", vol="GARCH", p=1, q=1)
    res = model.fit(last_obs=cutoff, disp="off")              # learn omega, alpha, beta before cutoff
    fc = res.forecast(horizon=5, start=cutoff, reindex=False)  # 5 daily variances per date
    daily_var = fc.variance / 100**2                          # undo the scaling
    return np.sqrt(daily_var.mean(axis=1))  