import os
import sys
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
from os import system
from IPython.display import display

import quandl

quandl.ApiConfig.api_key = "svsiy-zxmV-Stnbzu6zp"
glb_file_list = []


# Load market data using Quandl API and store it into Excel files
# Filtering only the desirable dates
def load_market_data_from_api(dt_start = '01/01/2014', dt_end='06/30/2020'):
    df_spx = quandl.get("CHRIS/CME_ES2")
    df_fste = quandl.get("CHRIS/LIFFE_Z2")
    df_dax = quandl.get("CHRIS/EUREX_FDAX2")

    df_fste.to_excel(r'.\MScFE690_Capstone\DB\db_ftse.xlsx')
    glb_file_list.append(r'.\MScFE690_Capstone\DB\db_ftse.xlsx')
    df_spx.to_excel(r'.\MScFE690_Capstone\DB\db_spx.xlsx')
    glb_file_list.append(r'.\MScFE690_Capstone\DB\db_spx.xlsx')
    df_dax.to_excel(r'.\MScFE690_Capstone\DB\db_dax.xlsx')
    glb_file_list.append(r'.\MScFE690_Capstone\DB\db_dax.xlsx')

    glb_file_list.append()


def load_market_data(dt_start = '01/01/2014', dt_end='06/30/2020'):
    df = DataFrame()
    for _file in glb_file_list:
        df_temp = pd.read_excel(_file, index_col=None, na_values=['NULL'], parse_dates=True)
        df = df.append(df_temp)
    return df

#print(sys.argv[0])
df = pd.read_excel(r'.\MScFE690_Capstone\DB\db_ftse.xlsx', index_col=None, na_values=['NULL'], parse_dates=True)
df.rename(columns={'Settle':'Close'}, inplace=True)
df.dropna(inplace=True)

##########################################################
# Calculating the classical indicators for trend following
##########################################################
df['MA07'] = ta.SMA(df['Close'].values, timeperiod=7)
df['EMA07'] = ta.EMA(df['Close'].values, timeperiod=7)
df['MA21'] = ta.SMA(df['Close'].values, timeperiod=21)
df['EMA21'] = ta.EMA(df['Close'].values, timeperiod=21)
df['RSI'] = ta.RSI(df['Close'].values, timeperiod=14)
df['ATR'] = ta.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['ADX'] = ta.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
df['MOM'] = ta.MOM(df['Close'].values, timeperiod=7)

macd, macdsignal, macdhist = ta.MACD(df['Close'].values, fastperiod=7, slowperiod=21, signalperiod=5)
df['MACD'] = macd
df['MACDsignal'] = macdsignal

df.fillna(0, inplace=True)

# Log transformation
df['target'] = (df['Close']-df['Close'].min()+1).transform(np.log)
#  normalization
df['normalized'] = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())

# Calculating the columns to be used as predictors for the averages and the MACD
df['ClgtMA07'] = np.where(df['Close'] > df['MA07'], 1, -1)
df['ClgtEMA07'] = np.where(df['Close'] > df['EMA07'], 1, -1)
df['ClgtMA21'] = np.where(df['Close'] > df['MA21'], 1, -1)
df['ClgtEMA21'] = np.where(df['Close'] > df['EMA21'], 1, -1)
df['MACDSIGgtMACD'] = np.where(df['MACDsignal'] > df['MACD'], 1, -1)

#df.to_excel(r'.\MScFE690_Capstone\DB\db_ftse_indicators.xlsx')

df['Return'] = df['Close'].pct_change(1).shift(-1)
df['target_class'] = np.where(df.Return > 0, 1, 0)
classification_target = df['target_class']
df['target_regression'] = df['Return']
regression_target = df['target_regression']

# Setting up the predictors dataframe
indicators = ['MOM','ClgtEMA07','ClgtEMA21','MACDSIGgtMACD','ATR','ADX','RSI','ClgtMA07','ClgtMA21']
#indicators = ['MACDSIGgtMACD','RSI','ClgtEMA07','ClgtEMA21','ClgtMA07','ClgtMA21','ADX']
#indicators = ['ClgtMA21','ClgtEMA21','MACDSIGgtMACD']
df_predictors = df[indicators]

# Defining the training ans testing datasets to be used in the Classification Tree
df_pred_class_train, df_pred_class_test, class_target_train, class_target_test = train_test_split(df_predictors, classification_target, test_size=0.3, random_state=123, stratify=classification_target, shuffle=True)

# Creating the Classification Tree
DT_classifier = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=3)

# Training and fitting the model with our dataset
DT_classifier = DT_classifier.fit(df_pred_class_train, class_target_train)

# Forecasting using the test dataset
class_target_pred = DT_classifier.predict(df_pred_class_test)
# df['class_prediction'] = class_target_pred

# df.to_excel(r'.\MScFE690_Capstone\DB\db_ftse_pred.xlsx')

# Getting performance data
DT_report = classification_report(class_target_test, class_target_pred)

diagram = tree.export_graphviz(DT_classifier, out_file=None,filled=True,feature_names=indicators)
graphviz.Source(diagram).view()

print(DT_report)

# Calculating Metrics for the Classification algorithm
acc_score = metrics.accuracy_score(class_target_test, class_target_pred)

auc = metrics.auc(np.sort(class_target_test), class_target_pred)
c_matrix = metrics.confusion_matrix(class_target_test, class_target_pred)
f1_score = metrics.f1_score(class_target_test, class_target_pred, average='macro')
precision = metrics.precision_score(class_target_test, class_target_pred, average='macro')
recall = metrics.recall_score(class_target_test, class_target_pred, average='macro')

fpr, tpr, threshold = metrics.roc_curve(class_target_test, class_target_pred, pos_label=2)

print(f"Accuracy Score: {acc_score}", f"AUC Score:{auc}", f"Confusion Matrix: {c_matrix}", f"f1 Score: {f1_score}",f"Precision score: {precision}", f"Recall Score: {recall}")

plt.plot(fpr, threshold, label='ROC curve (area = %0.3f)' % auc)

# #
# #  Factsheet - Classification Decision Tree
# #

# #loading dataset
# df.index = pd.to_datetime(df['Date'])
# df = df.drop('Date', axis=1)

# vol = get_daily_vol(close=df['Close'], lookback=5)
# cusum_events = cusum_filter(df['Close'], threshold=vol['2014-01-24':'2020-06-24'].mean()*0.5)
# vertical_barriers = add_vertical_barrier(t_events=cusum_events, close=df['Close'], num_days=1)

# pt_sl = [1, 2]
# min_ret = 0.0001
# triple_barrier_events = get_events(close=df['Close'],
#                                                t_events=cusum_events,
#                                                pt_sl=pt_sl,
#                                                target=vol,
#                                                min_ret=min_ret,
#                                                num_threads=3,
#                                                vertical_barrier_times=vertical_barriers)
# labels = get_bins(triple_barrier_events, df['Close'])

# # # Returns
# pred_dates = df_pred_class_test.Date

# # #pf.plot_rolling_returns(labels['ret'])
# pf.create_returns_tear_sheet(labels['ret'])

# # # Predicted returns
# predicted = labels.loc[pred_dates, 'ret'] * class_target_pred

# # #pf.plot_rolling_returns(predicted)
# pf.create_returns_tear_sheet(predicted)
