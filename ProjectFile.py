import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.stats.weightstats as wt

import inspect


def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

# Raw Data Input - Stocks
def getStockData(dir, file_name):
    path = dir
    stock = pd.DataFrame()
    for name in file_name:
        stock[name[:-10]] = pd.read_excel(os.path.join(path, name), header=6, index_col=0).iloc[:, 0]
    stock = stock.sort_index(ascending=True).dropna()
    # print(len(stock))
    return stock

# Raw Data Input - Options
def getOptionData(dir, file_name):
    path = dir
    option_dir = file_name[0]
    IV = pd.read_excel(os.path.join(path, option_dir), index_col=0)
    IV.sort_index().dropna()
    return IV

# Portfolio Construction
def get_Portfolio(pr1, pr2, sh1, sh2):
    port = sh1 * pr1 + sh2 * pr2
    return port

# derive log-return of portfolio
def log_return(port):
    r = np.log(port) - np.log(port.shift(1))
    r = r.dropna().sort_index()
    return r

# HVaR - relative method & long
def get_relative_HisVaR(price, t_days, t_rolling, p, position):
    rtn = np.log(price) - np.log(price.shift(t_days))
    HisVaR_rtn = rtn.rolling(t_rolling).quantile(p)
    HisVaR = position - position * np.exp(HisVaR_rtn)
    return HisVaR

# HVaR - relative method & short
def get_relative_HisVaR_Short(price, t_days, t_rolling, p, position):
    rtn = np.log(price) - np.log(price.shift(t_days))
    HisVaR_rtn = rtn.rolling(t_rolling).quantile(1 - p)
    HisVaR = -position + position * np.exp(HisVaR_rtn)
    return HisVaR

# HES - relative method & long
def get_relative_HisES(price, t_days, t_rolling, p, position):
    rtn = np.log(price) - np.log(price.shift(t_days))
    ESN = rtn.rolling(t_rolling).apply(lambda x: np.mean(x[x <= np.percentile(x, p * 100)]))
    ES = position - position * np.exp(ESN)
    return ES

# HES - relative method & short
def get_relative_HisES_Short(price, t_days, t_rolling, p, position):
    rtn = np.log(price) - np.log(price.shift(t_days))
    ESN = rtn.rolling(t_rolling).apply(lambda x: np.mean(x[x >= np.percentile(x, (1-p) * 100)]))
    ES = -position + position * np.exp(ESN)
    return ES

# HVaR - absolute method & long
def get_absolute_HisVaR(price, t_days, t_rolling, p, position):
    rtn = price - price.shift(t_days)
    HisVaR_rtn = rtn.rolling(t_rolling).quantile(p)[:]
    HisVaR = - position / price * HisVaR_rtn
    return HisVaR

# HVaR - absolute method & short
# YES WE DONT MODEL IT #

# HES - absolute method & long
def get_absolute_HisES(price, t_days, t_rolling, p, position):
    rtn = price - price.shift(t_days)
    ESN = rtn.rolling(t_rolling).apply(lambda x: np.mean(x[x <= np.percentile(x, p * 100)]))
    ES = - position / price * ESN
    return ES

# HES - absolute method & short
def get_absolute_HisES_Short(price, t_days, t_rolling, p, position):
    rtn = price - price.shift(t_days)
    ESN = rtn.rolling(t_rolling).apply(lambda x: np.mean(x[x >= np.percentile(x, (1-p) * 100)]))
    ES = position / price * ESN
    return ES

def get_absolute_HisES_Short(price, t_days, t_rolling, p, position):
    rtn = price - price.shift(t_days)
    ESN = rtn.rolling(t_rolling).apply(lambda x: np.mean(x[x >= np.percentile(x, (1-p) * 100)]))
    ES = -position + position * np.exp(ESN)
    return ES

# 
def get_equivalent_lambda(t, x=0.2):
    l = np.exp(np.log(x) / (t * 252))
    return l

# 
def get_Sigma(r, t):
    time = t * 252
    r_sq = np.square(r)
    mean = r.rolling(time).mean()
    mean_sq = r_sq.rolling(time).mean()
    var = mean_sq - np.square(mean)
    sigma = np.sqrt(var * 252)
    return sigma

# 
def get_Miu(r, t):
    time = t * 252
    sigma = get_Sigma(r, t)
    mean = r.rolling(time).mean()
    miu = 252 * mean + 0.5 * np.square(sigma)
    return miu

# GBM VaR - long
def para_GBMVaR(initial, miu, sigma, t, p):
    GBMVaR = initial - initial * np.exp(sigma * np.sqrt(t / 252) * norm.ppf(p) +
                                        (miu - 0.5 * np.square(sigma)) * (t / 252))
    return GBMVaR

# GBM VaR - short
def para_GBMVaR_Short(initial, miu, sigma, t, p):
    GBMVaR = -initial + initial * np.exp(sigma * np.sqrt(t / 252) * norm.ppf(1 - p) +
                                        (miu - 0.5 * np.square(sigma)) * (t / 252))
    return GBMVaR

# GBM ES - long
def para_GBMES(initial, miu, sigma, t, p):
    ESVaR = initial - initial * np.exp(miu * t / 252) * norm.cdf(norm.ppf(p) - sigma * np.sqrt(t / 252)) / p
    return ESVaR

# GBM ES - short
def para_GBMES_Short(initial, miu, sigma, t, p):
    ESVaR = initial * np.exp(miu * t / 252) * norm.cdf(-norm.ppf(1-p) + sigma * np.sqrt(t / 252)) / p - initial
    return ESVaR


def get_eq_Sigma(r, t):
    lam = get_equivalent_lambda(t)
    lam_series = np.logspace(0, 3000, 3000, base=lam)[::-1]
    mean = (r.rolling(3000).apply(lambda x: sum(x * lam_series))) * (1 - lam)
    r_sq = np.square(r)
    mean_sq = (r_sq.rolling(3000).apply(lambda x: sum(x * lam_series))) * (1 - lam)
    var = mean_sq - np.square(mean)
    sigma = np.sqrt(var * 252)
    return sigma


def get_eq_Miu(r, t):
    lam = get_equivalent_lambda(t)
    lam_series = np.logspace(0, 3000, 3000, base=lam)
    mean = (r.rolling(3000).apply(lambda x: sum(x * lam_series))) * (1 - lam)
    sigma = get_eq_Sigma(r, t)
    miu = mean * 252 + 0.5 * np.square(sigma)
    return miu


def norm_win_GBMVaR(initial, sh1, sh2, pr1, pr2, t_days, t_window, p):
    r1 = log_return(pr1)
    r2 = log_return(pr2)
    u1 = get_Miu(r1, t_window)
    u2 = get_Miu(r2, t_window)
    s1 = get_Sigma(r1, t_window)
    s2 = get_Sigma(r2, t_window)
    V0 = initial
    scale = initial / (sh1 * pr1 + sh2 * pr2)
    mean = (sh1 * pr1 * np.exp(u1 * t_days / 252) + sh2 * pr2 * np.exp(u2 * t_days / 252)) * scale
    correlation = r1.rolling(t_window * 252).corr(r2).dropna()
    mean_sq = (np.square(sh1 * pr1) * np.exp((2 * u1 + np.square(s1)) * t_days / 252) + np.square(sh2 * pr2) * np.exp((2 * u2 + np.square(s2)) * t_days / 252) + 2 * sh1 * sh2 * pr1 * pr2 * np.exp((u1 + u2 + correlation * s1 * s2) * t_days / 252)) * np.square(scale)
    var = -np.square(mean) + mean_sq
    sd = np.sqrt(var)
    VaR = V0 - (mean + norm.ppf(p) * sd)
    return VaR


def norm_win_GBMES(initial, sh1, sh2, pr1, pr2, t_days, t_window, p):
    r1 = log_return(pr1)
    r2 = log_return(pr2)
    u1 = get_Miu(r1, t_window)
    u2 = get_Miu(r2, t_window)
    s1 = get_Sigma(r1, t_window)
    s2 = get_Sigma(r2, t_window)
    V0 = initial
    scale = initial / (sh1 * pr1 + sh2 * pr2)
    mean = (sh1 * pr1 * np.exp(u1 * t_days / 252) + sh2 * pr2 * np.exp(u2 * t_days / 252)) * scale
    correlation = r1.rolling(t_window * 252).corr(r2).dropna()
    mean_sq = (np.square(sh1 * pr1) * np.exp((2 * u1 + np.square(s1)) * t_days / 252) + np.square(sh2 * pr2) * np.exp(
        (2 * u2 + np.square(s2)) * t_days / 252) + 2 * sh1 * sh2 * pr1 * pr2 * np.exp(
        (u1 + u2 + correlation * s1 * s2) * t_days / 252)) * np.square(scale)
    var = -np.square(mean) + mean_sq
    sd = np.sqrt(var)
    ES = mean + sd * norm.pdf(norm.ppf(1-p)) / p - V0
    return ES


def norm_eq_GBMVaR(initial, sh1, sh2, pr1, pr2, t_days, t_window, p):
    r1 = log_return(pr1)
    r2 = log_return(pr2)
    u1 = get_eq_Miu(r1, t_window)
    u2 = get_eq_Miu(r2, t_window)
    s1 = get_eq_Sigma(r1, t_window)
    s2 = get_eq_Sigma(r2, t_window)
    V0 = initial
    scale = initial / (sh1 * pr1 + sh2 * pr2)
    mean = (sh1 * pr1 * np.exp(u1 * t_days / 252) + sh2 * pr2 * np.exp(u2 * t_days / 252)) * scale
    correlation = r1.rolling(t_window * 252).corr(r2).dropna()
    mean_sq = (np.square(sh1 * pr1) * np.exp((2 * u1 + np.square(s1)) * t_days / 252) + np.square(sh2 * pr2) * np.exp((2 * u2 + np.square(s2)) * t_days / 252) + 2 * sh1 * sh2 * pr1 * pr2 * np.exp((u1 + u2 + correlation * s1 * s2) * t_days / 252)) * np.square(scale)
    var = -np.square(mean) + mean_sq
    sd = np.sqrt(var)
    VaR = V0 - (mean + norm.ppf(p) * sd)
    return VaR


def norm_eq_GBMES(initial, sh1, sh2, pr1, pr2, t_days, t_window, p):
    r1 = log_return(pr1)
    r2 = log_return(pr2)
    u1 = get_eq_Miu(r1, t_window)
    u2 = get_eq_Miu(r2, t_window)
    s1 = get_eq_Sigma(r1, t_window)
    s2 = get_eq_Sigma(r2, t_window)
    V0 = initial
    scale = initial / (sh1 * pr1 + sh2 * pr2)
    mean = (sh1 * pr1 * np.exp(u1 * t_days / 252) + sh2 * pr2 * np.exp(u2 * t_days / 252)) * scale
    correlation = r1.rolling(t_window * 252).corr(r2).dropna()
    mean_sq = (np.square(sh1 * pr1) * np.exp((2 * u1 + np.square(s1)) * t_days / 252) + np.square(sh2 * pr2) * np.exp(
        (2 * u2 + np.square(s2)) * t_days / 252) + 2 * sh1 * sh2 * pr1 * pr2 * np.exp(
        (u1 + u2 + correlation * s1 * s2) * t_days / 252)) * np.square(scale)
    var = -np.square(mean) + mean_sq
    sd = np.sqrt(var)
    ES = mean + sd * norm.pdf(norm.ppf(1-p)) / p - V0
    return ES


def MC_Port_VaR(initial, miu, sigma, t_days, sample, p):
    w = np.random.standard_normal(sample)
    VaRlst = []
    for i in range(len(miu)):
        VaR = np.percentile(initial - initial * np.exp((miu[i] - np.square(sigma[i]) / 2) * (t_days / 252) +
                                                       sigma[i] * w * np.sqrt(t_days / 252)), (1 - p) * 100)
        VaRlst.append(VaR)
    return VaRlst


def MC_Port_VaR_Short(initial, miu, sigma, t_days, sample, p):
    w = np.random.standard_normal(sample)
    VaRlst = []
    for i in range(len(miu)):
        VaR = np.percentile(-initial + initial * np.exp((miu[i] - np.square(sigma[i]) / 2) * (t_days / 252) +
                                                       sigma[i] * w * np.sqrt(t_days / 252)), (1 - p) * 100)
        VaRlst.append(VaR)
    return VaRlst


def MC_Port_ES(initial, miu, sigma, t_days, sample, p):
    w = np.random.standard_normal(sample)
    ESlst = []
    for i in range(len(miu)):
        Vt = initial - initial * np.exp((miu[i] - np.square(sigma[i]) / 2) * (t_days / 252) + sigma[i] * w * np.sqrt(t_days / 252))
        ES = np.mean(Vt[Vt >= np.percentile(Vt, (1-p) * 100)])
        ESlst.append(ES)
    return ESlst


def MC_Port_ES_Short(initial, miu, sigma, t_days, sample, p):
    w = np.random.standard_normal(sample)
    ESlst = []
    for i in range(len(miu)):
        Vt = -initial + initial * np.exp((miu[i] - np.square(sigma[i]) / 2) * (t_days / 252) + sigma[i] * w * np.sqrt(t_days / 252))
        ES = np.mean(Vt[Vt >= np.percentile(Vt, (1-p) * 100)])
        ESlst.append(ES)
    return ESlst


def MC_Norm_VaR(initial, price1, price2, t, t_all, t_days, sh1, sh2, sample, p):
    VaR = []
    r1 = log_return(price1)
    r2 = log_return(price2)
    miu1 = get_Miu(r1, t)
    miu2 = get_Miu(r2, t)
    sigma1 = get_Sigma(r1, t)
    sigma2 = get_Sigma(r2, t)
    mean = [0, 0]
    correlation = r1.rolling(t * 252).corr(r2).dropna()
    for i in range(len(correlation)):
        corr = [[1, correlation[i]], [correlation[i], 1]]
        w = np.random.multivariate_normal(mean, corr, sample)
        w1 = np.array([x[0] for x in w])
        w2 = np.array([x[1] for x in w])
        m1 = miu1[i]
        m2 = miu2[i]
        s1 = sigma1[i]
        s2 = sigma2[i]
        S1 = price1[i]
        S2 = price2[i]
        scale = initial / (sh1 * S1 + sh2 * S2)
        LossN = np.percentile(
            10000 - scale * (sh1 * S1 * np.exp((m1 - s1 ** 2 / 2) * (t_days / 252) + s1 * w1 * np.sqrt(t_days / 252)) +
                             sh2 * S2 * np.exp((m2 - s2 ** 2 / 2) * (t_days / 252) + s2 * w2 * np.sqrt(t_days / 252))),
            (1 - p) * 100)
        VaR.append(LossN)
    return VaR


def MC_eq_Norm_VaR(initial, price1, price2, t, t_all, t_days, sh1, sh2, sample, p):
    VaR = []
    r1 = log_return(price1)
    r2 = log_return(price2)
    miu1 = get_eq_Miu(r1, t)
    miu2 = get_eq_Miu(r2, t)
    sigma1 = get_eq_Sigma(r1, t)
    sigma2 = get_eq_Sigma(r2, t)
    mean = [0, 0]
    correlation = r1.rolling(t * 252).corr(r2).dropna()
    for i in range(len(correlation)):
        corr = [[1, correlation[i]], [correlation[i], 1]]
        w = np.random.multivariate_normal(mean, corr, sample)
        w1 = np.array([x[0] for x in w])
        w2 = np.array([x[1] for x in w])
        m1 = miu1[i]
        m2 = miu2[i]
        s1 = sigma1[i]
        s2 = sigma2[i]
        S1 = price1[i]
        S2 = price2[i]
        scale = initial / (sh1 * S1 + sh2 * S2)
        LossN = np.percentile(
            10000 - scale * (sh1 * S1 * np.exp((m1 - s1 ** 2 / 2) * (t_days / 252) + s1 * w1 * np.sqrt(t_days / 252)) +
                             sh2 * S2 * np.exp((m2 - s2 ** 2 / 2) * (t_days / 252) + s2 * w2 * np.sqrt(t_days / 252))),
            (1 - p) * 100)
        VaR.append(LossN)
    return VaR


def MC_Norm_ES(initial, price1, price2, t, t_all, t_days, sh1, sh2, sample, p):
    ES = []
    r1 = log_return(price1)
    r2 = log_return(price2)
    miu1 = get_Miu(r1, t)
    miu2 = get_Miu(r2, t)
    sigma1 = get_Sigma(r1, t)
    sigma2 = get_Sigma(r2, t)
    mean = [0, 0]
    correlation = r1.rolling(t * 252).corr(r2).dropna()
    for i in range(len(correlation)):
        corr = [[1, correlation[i]], [correlation[i], 1]]
        w = np.random.multivariate_normal(mean, corr, sample)
        w1 = np.array([x[0] for x in w])
        w2 = np.array([x[1] for x in w])
        m1 = miu1[i]
        m2 = miu2[i]
        s1 = sigma1[i]
        s2 = sigma2[i]
        S1 = price1[i]
        S2 = price2[i]
        scale = initial / (sh1 * S1 + sh2 * S2)
        Vt = 10000 - scale * (sh1 * S1 * np.exp((m1 - s1 ** 2 / 2) * (t_days / 252) + s1 * w1 * np.sqrt(t_days / 252)) +
                             sh2 * S2 * np.exp((m2 - s2 ** 2 / 2) * (t_days / 252) + s2 * w2 * np.sqrt(t_days / 252)))
        LossN = np.mean(Vt[Vt >= np.percentile(Vt, (1 - p) * 100)])
        ES.append(LossN)
    return ES


def MC_eq_Norm_ES(initial, price1, price2, t, t_all, t_days, sh1, sh2, sample, p):
    ES = []
    r1 = log_return(price1)
    r2 = log_return(price2)
    miu1 = get_eq_Miu(r1, t)
    miu2 = get_eq_Miu(r2, t)
    sigma1 = get_eq_Sigma(r1, t)
    sigma2 = get_eq_Sigma(r2, t)
    mean = [0, 0]
    correlation = r1.rolling(t * 252).corr(r2).dropna()
    for i in range(len(correlation)):
        corr = [[1, correlation[i]], [correlation[i], 1]]
        w = np.random.multivariate_normal(mean, corr, sample)
        w1 = np.array([x[0] for x in w])
        w2 = np.array([x[1] for x in w])
        m1 = miu1[i]
        m2 = miu2[i]
        s1 = sigma1[i]
        s2 = sigma2[i]
        S1 = price1[i]
        S2 = price2[i]
        scale = initial / (sh1 * S1 + sh2 * S2)
        Vt = 10000 - scale * (sh1 * S1 * np.exp((m1 - s1 ** 2 / 2) * (t_days / 252) + s1 * w1 * np.sqrt(t_days / 252)) +
                              sh2 * S2 * np.exp((m2 - s2 ** 2 / 2) * (t_days / 252) + s2 * w2 * np.sqrt(t_days / 252)))
        LossN = np.mean(Vt[Vt >= np.percentile(Vt, (1 - p) * 100)])
        ES.append(LossN)
    return ES


def get_ATM_d1(rf, iv, maturity):
    d1_value = (rf + np.square(iv) / 2) * maturity / (iv * np.sqrt(maturity))
    return d1_value


def optionHedge(initial, rf, liq, port, IV, maturity, miu, sigma, VaR, t_days, sample, p):
    WN = np.random.standard_normal(sample)
    S0 = port[-1]
    u = miu[-1]
    s = sigma[-1]
    iv = IV.iloc[-1, -1] / 100.

    time_expiration = maturity - t_days / 252
    d1 = get_ATM_d1(rf, iv, maturity)
    d1prime = get_ATM_d1(rf, iv, time_expiration)
    d2 = d1 - iv * np.sqrt(maturity)
    d2prime = d1prime - iv * np.sqrt(time_expiration)
    Option0 = np.exp(-rf * maturity) * S0 * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    flag = VaR[-1] * (1 - liq)
    x = 1

    while True:

        Stock_Shares = initial * x / S0
        Option_Shares = initial * (1 - x) / Option0

        Optiont = np.exp(-rf * (252 - t_days) / 252) * S0 * norm.cdf(-d2prime) - S0 * np.exp((u - np.square(s) / 2) *
                                                                                             (
                                                                                                     t_days / 252) + s * WN * np.sqrt(
            t_days / 252)) * norm.cdf(-d1prime)
        MCVaR = np.percentile(Stock_Shares * (S0 - (S0 * np.exp((u - np.square(s) / 2) * (t_days / 252) +
                                                                s * WN * np.sqrt(t_days / 252)))) + Option_Shares * (
                                      Option0 - Optiont), (1 - p) * 100)
        # print(MCVaR)

        if MCVaR <= flag:
            # print(x)
            break
        x -= 0.0001
    return x


def CallOptionHedge(initial, rf, liq, port, IV, maturity, miu, sigma, VaR, t_days, sample, p):
    WN = np.random.standard_normal(sample)
    S0 = port[-1]
    u = miu[-1]
    s = sigma[-1]
    iv = IV.iloc[-1, -1] / 100.

    time_expiration = maturity - t_days / 252
    d1 = get_ATM_d1(rf, iv, maturity)
    d1prime = get_ATM_d1(rf, iv, time_expiration)
    d2 = d1 - iv * np.sqrt(maturity)
    d2prime = d1prime - iv * np.sqrt(time_expiration)
    Option0 = - np.exp(-rf * maturity) * S0 * norm.cdf(d2) + S0 * norm.cdf(d1)

    flag = VaR[-1] * (1 - liq)
    x = 1

    while True:

        Stock_Shares = initial * x / S0
        Option_Shares = initial * (1 - x) / Option0

        Optiont = -np.exp(-rf * (252 - t_days) / 252) * S0 * norm.cdf(d2prime) + S0 * np.exp((u - np.square(s) / 2) *
                                                                                             (
                                                                                                     t_days / 252) + s * WN * np.sqrt(
            t_days / 252)) * norm.cdf(d1prime)
        MCVaR = np.percentile(Stock_Shares * (-S0 + (S0 * np.exp((u - np.square(s) / 2) * (t_days / 252) +
                                                                s * WN * np.sqrt(t_days / 252)))) + Option_Shares * (
                                      Option0 - Optiont), (1 - p) * 100)
        # print(MCVaR)

        if MCVaR <= flag:
            # print(x)
            break
        x -= 0.0001
    return x


def backtesting(initial, port, VaRL, VaRS, t_days, t_window):

    exceptions_Long_P = []
    exceptions_Short_P = []

    RL_Long_P = []
    RL_Short_P = []

    T = t_window - t_days

    '''Long&Short'''
    for i in range(len(port) - t_window):
        num_Long_P = 0
        num_Short_P = 0

        for j in range(T):
            # Calculate
            P_Last_P = port[i + j]
            P_Start_P = port[i + j + t_days]
            RealLoss_Long_P = initial - initial / P_Start_P * P_Last_P
            RealLoss_Short_P = -RealLoss_Long_P

            # Calculate the exceptions
            if RealLoss_Long_P > VaRL[i + j + t_days]:
                num_Long_P += 1
            if RealLoss_Short_P > VaRS[i + j + t_days]:
                num_Short_P += 1
            if j == 0:
                RL_Long_P.append(RealLoss_Long_P)
                RL_Short_P.append(RealLoss_Short_P)

        exceptions_Long_P.append(num_Long_P)
        exceptions_Short_P.append(num_Short_P)

    return RL_Long_P, RL_Short_P, exceptions_Long_P, exceptions_Short_P


def hypo_test(a_list):
    flag = 'Matches'
    # test_mean = np.mean(a_list)
    z, pval = wt.ztest(a_list, value=2.52)
    # print(z, pval)
    if pval < 0.05:
        flag = "Doesn't match!"
    return flag


def plotFunction(lstOfLists):  
    for lst in lstOfLists:
        plt.plot(lst, label=retrieve_name(lst)) #flip the timeline
        plt.legend()
    plt.ylabel('Dollar')
    plt.xlabel('Date')
    #plt.title(title)
    plt.grid()
    plt.show()
    return 0

if __name__ == "__main__":
    # Stock data
    path = '/Users/mac/Downloads/DataFile'
    file_name = ['p_data.xlsx', 'd_data.xlsx']
    stock = getStockData(path, file_name)

    # Option data
    iv = getOptionData(path, ['SPX IV.xlsx'])

    # t-days VaR
    t = 5

    # T Time window (years)
    T = 5

    # The shares of stock A and B
    # Initial Capital
    # (1-p1)% VaR
    # (1-p2)% ES
    # Sample path number
    A_hand = 167 # DIS
    B_hand = 47 #PG
    Initial = 10000
    p1 = 0.01
    p2 = 0.025
    sample = 10000 

    # Get price of stock A and B and portfolio
    price1 = stock.iloc[:, 0]
    price2 = stock.iloc[:, 1]
    port = get_Portfolio(price1, price2, A_hand, B_hand)

    # Get log return
    log_rtn_P = log_return(port)
    log_rtn_A = log_return(price1)
    log_rtn_B = log_return(price2)

    '''Long Part'''
    # Get Historical VaR and ES
    # Re_HisVaR = get_relative_HisVaR(port, t, 252 * T, p1, Initial)
    # Re_HisES = get_relative_HisES(port, t, 252*T, p2, Initial)
    # Ab_HisVaR = get_absolute_HisVaR(port, t, 252 * T, p1, Initial)
    # Ab_HisES = get_absolute_HisES(port, t, 252 * T, p2, Initial)

    # Get Parametric VaR and ES
    # GBM for Portfolio(window)
    miuP = get_Miu(log_rtn_P, T)
    sigmaP = get_Sigma(log_rtn_P, T)
    #GBM_window_VaR = para_GBMVaR(Initial, miuP, sigmaP, t, p1)
    GBM_window_ES = para_GBMES(Initial, miuP, sigmaP, t, p2)

    # GBM for Portfolio(equivalent)
    eq_sigmaP = get_eq_Sigma(log_rtn_P, T)
    eq_miuP = get_eq_Miu(log_rtn_P, T)
    # GBM_eq_VaR = para_GBMVaR(Initial, eq_miuP, eq_sigmaP, t, p1)
    GBM_eq_ES = para_GBMES(Initial, eq_miuP, eq_sigmaP, t, p2)

    # GBM for underlying stocks(window)
    # miuA = get_Miu(log_rtn_A, T)
    # sigmaA = get_Sigma(log_rtn_A, T)
    # miuB = get_Miu(log_rtn_B, T)
    # sigmaB = get_Sigma(log_rtn_B, T)
    # GBM_Normal_window_Var = norm_win=_GBMVaR(Initial, A_hand, B_hand, price1, price2, t, T, p1)
    # GBM_Normal_window_ES = norm_win_GBMVaR(Initial, A_hand, B_hand, price1, price2, t, T, p2)

    # GBM for underlying stocks(equivalent)
    # eq_miuA = get_eq_Miu(log_rtn_A, T)
    # eq_sigmaA = get_eq_Sigma(log_rtn_A, T)
    # eq_miuB = get_eq_Miu(log_rtn_B, T)
    # eq_sigmaB = get_eq_Sigma(log_rtn_B, T)
    # GBM_Normal_eq_Var = norm_eq_GBMVaR(Initial, A_hand, B_hand, price1, price2, t, T, p1)
    # GBM_Normal_eq_ES = norm_eq_GBMVaR(Initial, A_hand, B_hand, price1, price2, t, T, p2)

    ''' plotFunction([GBM_Normal_window_Var, GBM_Normal_window_ES, GBM_Normal_eq_Var, GBM_Normal_eq_ES]) '''

    # Monte Carlo
    # MCVaR_window_P = MC_Port_VaR(Initial, miuP, sigmaP, t, sample, p1)
    # MCES_window_P = MC_Port_ES(Initial, miuP, sigmaP, t, sample, p2)
    # MCVaR_eq_P = MC_Port_VaR(Initial, eq_miuP, eq_sigmaP, t, sample, p1)
    # MCES_eq_P = MC_Port_ES(Initial, eq_miuP, eq_sigmaP, t, sample, p2)4

    '''plotFunction([MCVaR_window_P, MCES_window_P, MCVaR_eq_P, MCES_eq_P])'''

    # MCVaR_window_norm = MC_Norm_VaR(Initial, price1, price2, T, 20, t, A_hand, B_hand, sample, p1)
    # MCES_window_norm = MC_Norm_ES(Initial, price1, price2, T, 20, t, A_hand, B_hand, sample, p2)
    # MCVaR_eq_norm = MC_eq_Norm_VaR(Initial, price1, price2, T, 20, t, A_hand, B_hand, sample, p1)
    # MCES_eq_norm = MC_eq_Norm_ES(Initial, price1, price2, T, 20, t, A_hand, B_hand, sample, p2)

    '''plotFunction([MCVaR_window_norm, MCES_window_norm, MCVaR_eq_norm, MCES_eq_norm])'''

    '''plotFunction([GBM_window_VaR, GBM_Normal_window_Var])'''

    '''plotFunction([MCVaR_window_norm, MCVaR_window_P])'''

    '''Short Part'''
    # Get Historical VaR and ES
    # Re_HisVaR_Short = get_relative_HisVaR_Short(port, t, 252 * T, p1, Initial)
    # Re_HisES_Short = get_relative_HisES_Short(port, t, 252*T, p2, Initial)
    # Ab_HisVaR_Short = get_absolute_HisVaR_Short(port, t, 252 * T, p1, Initial)
    # Ab_HisES_Short = get_absolute_HisES_Short(port, t, 252 * T, p2, Initial)

    # Get Parametric VaR and ES
    # GBM for Portfolio(window)
    # GBM_window_VaR_Short = para_GBMVaR_Short(Initial, miuP, sigmaP, t, p1)
    GBM_window_ES_Short = para_GBMES_Short(Initial, miuP, sigmaP, t, p2)

    # GBM for Portfolio(equivalent)
    # GBM_eq_VaR_Short = para_GBMVaR_Short(Initial, eq_miuP, eq_sigmaP, t, p1)
    GBM_eq_ES_Short = para_GBMES_Short(Initial, eq_miuP, eq_sigmaP, t, p2)

    '''plotFunction([GBM_window_VaR_Short,GBM_eq_VaR_Short])'''
    plotFunction([GBM_window_ES_Short,GBM_eq_ES_Short])

    # Monte Carlo
    # MCVaR_window_P_Short = MC_Port_VaR_Short(Initial, miuP, sigmaP, t, sample, p1)
    # MCES_window_P_Short = MC_Port_ES_Short(Initial, miuP, sigmaP, t, sample, p2)
    # MCVaR_eq_P_Short = MC_Port_VaR_Short(Initial, eq_miuP, eq_sigmaP, t, sample, p1)
    # MCES_eq_P_Short = MC_Port_ES_Short(Initial, eq_miuP, eq_sigmaP, t, sample, p2)

    # plotFunction([MCVaR_window_P_Short,MCES_window_P_Short,MCVaR_eq_P_Short,MCES_eq_P_Short])

    # Option
    # Initial parameters
    # Risk-free rate
    # Option Maturity
    # The percentage of VaR need to decrease
    rf = 0.005
    Maturity = 1

    # Long Portfolio Long Put
    # seq = np.arange(0, 0.4, 0.01)
    # Long_Portfolio_Long_Put = []
    # for i in seq:
    #     x = optionHedge(Initial, rf, i, port, iv, Maturity, miuP, sigmaP, GBM_window_VaR, t, sample, p1)
    #     k = np.around((1 - x) * 100, 2)
    #     Long_Portfolio_Long_Put.append(k)

    # plotFunction([Long_Portfolio_Long_Put])

    # Short Portfolio Long Call
    # Liquidate percentage (lst here is a list contains different liquidate ratio)
    # seq = np.arange(0, 0.4, 0.01)
    # Short_Portfolio_Long_Call = []
    # for i in seq:
    #     x = CallOptionHedge(Initial, rf, i, port, iv, Maturity, miuP, sigmaP, GBM_window_VaR, t, sample, p1)
    #     k = np.around((1 - x) * 100, 2)
    #     Short_Portfolio_Long_Call.append(k)
    #     # print('The percentage needed to be liquidated : ' + str(k) + '%')
    # plotFunction([Short_Portfolio_Long_Call])

    # Backtesing time horizon
    # Backtesting
    t_bt = 252
    # RL_Long_P, RL_Short_P, exceptions_Long_P, exceptions_Short_P = backtesting(Initial, port, GBM_window_VaR ,GBM_window_VaR_Short, t, t_bt)
    # RL_eq_Long_P, RL_eq_Short_P, exceptions_eq_Long_P, exceptions_eq_Short_P = backtesting(Initial, port, GBM_eq_VaR, GBM_eq_VaR_Short ,t, t_bt)
    ''' plotFunction([RL_eq_Long_P, RL_eq_Short_P])
    plotFunction([exceptions_eq_Long_P, exceptions_eq_Short_P])'''



