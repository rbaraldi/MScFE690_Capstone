import os
import datetime
import numpy as np
import pandas as pd
import pyfolio as pf
import matplotlib.pyplot as plt
from mlfinlab.util import get_daily_vol
from mlfinlab.filters import cusum_filter
from mlfinlab.labeling import add_vertical_barrier, get_events, get_bins
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import classification_report, r2_score
import graphviz
from datetime import datetime
import talib as ta

import quandl

quandl.ApiConfig.api_key = "svsiy-zxmV-Stnbzu6zp"
glb_file_list = []


# Load market data using Quandl API and store it into Excel files
# Filtering only the desirable dates
def load_market_data_from_api(dt_start = '01/01/2014', dt_end='06/30/2020'):
    df_spx = quandl.get("CHRIS/CME_ES2")
    df_fste = quandl.get("CHRIS/LIFFE_Z2")
    df_dax = quandl.get("CHRIS/EUREX_FDAX2")

    df_fste.to_excel(r'.\DB\db_ftse.xlsx')
    glb_file_list.append(r'.\DB\db_ftse.xlsx')
    df_spx.to_excel(r'.\DB\db_spx.xlsx')
    glb_file_list.append(r'.\DB\db_spx.xlsx')
    df_dax.to_excel(r'.\DB\db_dax.xlsx')
    glb_file_list.append(r'.\DB\db_dax.xlsx')

    glb_file_list.append()


def load_market_data(dt_start = '01/01/2014', dt_end='06/30/2020'):
    df = DataFrame()
    for _file in glb_file_list:
        df_temp = pd.read_excel(_file, index_col=None, na_values=['NULL'], parse_dates=True)
        df = df.append(df_temp)
    return df

