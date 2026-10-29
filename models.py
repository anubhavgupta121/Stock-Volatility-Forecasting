import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import metrics

def HAR(train : pd.DataFrame, cal : pd.DataFrame, cols : list) -> pd.Series:

    X = train[cols] #HAR is defined only for these columns
    y = train["target"]
    X_c = sm.add_constant(X)

    X_pred = cal[cols]
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
    
def XGBOOST(train: pd.DataFrame, val: pd.DataFrame, cols: list,
            depth: int, lr: float, n: int = 2000, es_frac: float = 0.15) -> pd.Series:
    # carve the LAST part of training (by date) as the early-stopping set
    cut = int(len(train) * (1 - es_frac))
    fit_part, es_part = train.iloc[:cut], train.iloc[cut:]

    # QLIKE as the stopping metric (XGBoost passes plain numpy arrays)
    def qlike_metric(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-8, None)   # early trees can dip to <= 0
        return metrics.QLIKE(pd.Series(y_pred), pd.Series(y_true))

    model = XGBRegressor(
        n_estimators=n,              # just a ceiling now
        max_depth=depth,
        learning_rate=lr,
        early_stopping_rounds=50,    # stop after 50 trees with no improvement in teh carved-out validation set from the otignal training set.
        eval_metric=qlike_metric,
        random_state=42,
    )
    model.fit(
        fit_part[cols], fit_part["target"],
        eval_set=[(es_part[cols], es_part["target"])],
        verbose=False,
    )
    print(f"stopped at tree {model.best_iteration}")

    preds = model.predict(val[cols])   # uses the best iteration automatically
    return pd.Series(preds, index=val.index)

def XG_obj(y_act : pd.Series,  z : pd.Series) :
    lo = np.exp(-z)
    grad = 1 - y_act*(lo)
    hess = y_act*lo

    return grad,hess


def XGBOOST_QLIKE(train: pd.DataFrame, val: pd.DataFrame, cols: list,
            depth: int, lr: float, n: int = 2000, es_frac: float = 0.15) -> pd.Series:
    # carve the LAST part of training (by date) as the early-stopping set
    cut = int(len(train) * (1 - es_frac))
    fit_part, es_part = train.iloc[:cut], train.iloc[cut:]

    # QLIKE as the stopping metric (XGBoost passes plain numpy arrays)
    def qlike_metric(y_true, z):
        y_pred_vol = np.sqrt(np.exp(z))
        y_true_vol = np.sqrt(y_true)  #converting back to volatilities
        return metrics.QLIKE(pd.Series(y_pred_vol), pd.Series(y_true_vol))

    model = XGBRegressor(
        n_estimators=n,              # just a ceiling now
        max_depth=depth,
        learning_rate=lr,
        objective=XG_obj,
        base_score = np.log((fit_part["target"]**2).mean()),
        early_stopping_rounds=50,    # stop after 50 trees with no improvement in teh carved-out validation set from the otignal training set.
        eval_metric=qlike_metric,
        random_state=42,
    )
    model.fit(
        fit_part[cols], fit_part["target"]**2,
        eval_set=[(es_part[cols], es_part["target"]**2)],  ##here now we are preidcting varience, not volatility so that we can feed it into QLIKE
        verbose=False,
        
    )
    print(f"stopped at tree {model.best_iteration}")

    preds = np.sqrt(np.exp(model.predict(val[cols])))   # uses the best iteration automatically
    return pd.Series(preds, index=val.index)

def XGBOOST_HYBRID(train, val, cols, depth, lr, garch_var_train, garch_var_val,
                   n=2000, es_frac=0.15):
    cut = int(len(train) * (1 - es_frac))
    fit_part, es_part = train.iloc[:cut], train.iloc[cut:]

    # each day's starting point = log of GARCH's variance forecast
    m_fit = np.log(garch_var_train.loc[fit_part.index])
    m_es  = np.log(garch_var_train.loc[es_part.index])
    m_val = np.log(garch_var_val.loc[val.index])

    def qlike_metric(y_true, z):
        return metrics.QLIKE(pd.Series(np.sqrt(np.exp(z))), pd.Series(np.sqrt(y_true)))

    model = XGBRegressor(
        n_estimators=n, max_depth=depth, learning_rate=lr,
        objective=XG_obj,                 # unchanged
        early_stopping_rounds=50, eval_metric=qlike_metric, random_state=42,
    )                                     # no base_score: base_margin replaces it
    model.fit(
        fit_part[cols], fit_part["target"] ** 2, base_margin=m_fit,
        eval_set=[(es_part[cols], es_part["target"] ** 2)], base_margin_eval_set=[m_es],
        verbose=False,
    )
    print(f"stopped at tree {model.best_iteration}")
    z = model.predict(val[cols], base_margin=m_val)   # includes the GARCH start
    return pd.Series(np.sqrt(np.exp(z)), index=val.index)
    

    



