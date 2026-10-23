import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def HAR(train : pd.DataFrame, cal : pd.DataFrame) -> pd.Series:

    X = train[["today_vol", "past_5_days", "past_22_days"]] #HAR is defined only for these columns
    y = train["target"]
    X_c = sm.add_constant(X)

    X_pred = cal[["today_vol", "past_5_days", "past_22_days"]]
    X_pred = sm.add_constant(X_pred)

    model = sm.OLS(y,X_c)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    y_pred = res.predict(X_pred)


    return y_pred

def HAR_LOG(train : pd.DataFrame, cal : pd.DataFrame) -> pd.Series:

    X = np.log(train[["today_vol", "past_5_days", "past_22_days"]])
    y = np.log(train["target"])
    X_c = sm.add_constant(X)
    
    X_pred = np.log(cal[["today_vol", "past_5_days", "past_22_days"]])
    X_pred = sm.add_constant(X_pred)

    model = sm.OLS(y,X_c)
    res = model.fit()
    y_pred = res.predict(X_pred) 

    return np.exp(y_pred) *  np.exp(res.mse_resid / 2)


def random_forest(train: pd.DataFrame, val: pd.DataFrame, cols: list, leaf: int = 50) -> pd.Series:
    X_train, y_train = train[cols], train["target"]
    X_val = val[cols]

    model = RandomForestRegressor(
        n_estimators=500,       
        min_samples_leaf=leaf,  
        random_state=42,        
        n_jobs=-1,             
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)                 
    return pd.Series(preds, index=X_val.index)  
    
    

    



