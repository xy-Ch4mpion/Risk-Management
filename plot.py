import inspect
import scipy as scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import scoreatpercentile
from numpy.random import normal 
from numpy.random import multivariate_normal

def import_data( file_source ):
    raw_data_frame = pd.read_csv(file_source)['0']
    lst = list(raw_data_frame)
    return lst

def plotFunction(lstOfLists):  
    for lst in lstOfLists:
        plt.plot(lst[::-1], label=retrieve_name(lst)) #flip the timeline
        plt.legend()
    plt.show()
    return 0

def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

GBM_VaR_long = import_data('/Users/mac/Downloads/datafolder/GBM_VaR_5_year_GBM_LONG.csv')

GBM_VaR_short = import_data('/Users/mac/Downloads/datafolder/GBM_VaR_5_year_short_postition.csv')

GBM_VaR_F_long = import_data('/Users/mac/Downloads/datafolder/GBM_VaR_F_long1.csv')

GBM_VaR_XRX_long = import_data('/Users/mac/Downloads/datafolder/GBM_VaR_XRX_long1.csv')

GBM_VaR_F_short = import_data('/Users/mac/Downloads/datafolder/GBM_VaR_F_short1.csv')

GBM_VaR_XRX_short = import_data('/Users/mac/Downloads/datafolder/GBM_VaR_XRX_short1.csv')

GBM_ES_F_short = import_data('/Users/mac/Downloads/datafolder/GBM_ES_F_short1.csv')

GBM_ES_XRX_short = import_data('/Users/mac/Downloads/datafolder/GBM_ES_XRX_short1.csv')

exp_VaR_5_year = import_data('/Users/mac/Downloads/datafolder/exp_VaR_5_yearlong.csv')

exp_df = pd.read_excel('/Users/mac/Downloads/datafolder/expo.xlsx')

print(exp_df)

ex

# plotFunction([ exp_VaR_5_year, GBM_VaR_long ])