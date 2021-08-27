# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 11:58:30 2021

@author: PriyankRao
Regression
"""
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime  as dt
import matplotlib.pyplot as plt

# =============================================================================
# setting the working directory
# =============================================================================

os.chdir(r"C:\Users\PriyankRao\OneDrive - Sempra Energy\Data\Bechmarking_1\Main Files\Benchmarking Database\temp file")
os.getcwd()
# =============================================================================
# getting the actuals data
# =============================================================================

actuals = pd.read_excel(r'Actuals_.xlsx', sheet_name = 'Sheet1', converters = {'TM1 ID': str})
actuals.dtypes
total_actuals = actuals[actuals['Cost Function']=='All Functions wo GMA']
total_actuals = total_actuals[['TM1 ID', 'Actuals']]
# =============================================================================
# getting the p6 informations
# =============================================================================
p6 = pd.read_excel(r"p6.xlsx", converters = {'TM1 ID': str, 'WOA': str})


# =============================================================================
# getting the data together from p6 and actuals files
# =============================================================================

p6.columns
p6['TM1 ID'].value_counts()

Actuals_AF = pd.merge(total_actuals, p6, on = 'TM1 ID', how = 'left') 
# df = Actuals_AF[Actuals_AF['Type of Work'].isnull()]
Actuals_AF.columns


Actuals_AF['Type of Work'].value_counts()

Actuals_AF[Actuals_AF['Type of Work'].isnull()]

typ = ['Replace', 'Test']

Actuals_AF = Actuals_AF[Actuals_AF['Type of Work'].isin(typ)].reset_index(drop = True)

Actuals_AF = Actuals_AF[[ 'SCG-District','Type of Work', 
                         'Total Mileage/Section  ', 'Test', 'Replace', 'Abandon','Actuals' ]]

Actuals_AF.columns
sns_plot = sns.pairplot(Actuals_AF, x_vars = ['Total Mileage/Section  ','Test', 'Replace', 'Abandon'], y_vars ='Actuals', kind = 'scatter', size = 7)
fig = sns_plot.savefig("Scatter.png")

plt.figure(figsize = (16,6))
heatmap = sns.heatmap(variables.corr(), vmin =-1, vmax = 1, annot =True)
heatmap.set_title('correlation Heatmap', fontdict={'fontsize':12}, pad = 12)
plt.savefig('heatmap.png', dpi =300, bbox_inches = 'tight')


enc = ohe(handle_unknown ='ignore')

Actuals_AF = Actuals_AF.fillna("N/A")

# , 'PSEP Portfolio', 'Department'
enc_df = pd.DataFrame (enc.fit_transform(Actuals_AF[['Type of Work', 'SCG-District']]).toarray())
# enc_df.to_excel(r"df.xlsx", index = False)

Actuals_AF.columns
Actuals_AF = Actuals_AF.join(enc_df)
typwrk = Actuals_AF[[
 'Type of Work',
 'Total Mileage/Section  ',
 'Test',
 'Replace',
 'Abandon',
 'Actuals',
 0,
 1,
 2,
 3,
 4,
 5,
 6,
 7,
 8,
 9,
 10,
 11,
 12,
 13,
 14,
 15,
 16,
 17,
 18,
 19,
 20,
 21,
 22,
 23,
 24,
 25,
 26,
 27,
 28,
 29]]

typwrk['Type of Work']
typwrk = typwrk.drop_duplicates(subset = ['Type of Work']) 
typwrk.to_excel(r"typwork.xlsx", index = False )

dist = Actuals_AF[['SCG-District',
 
 0,
 1,
 2,
 3,
 4,
 5,
 6,
 7,
 8,
 9,
 10,
 11,
 12,
 13,
 14,
 15,
 16,
 17,
 18,
 19,
 20,
 21,
 22,
 23,
 24,
 25,
 26,
 27,
 28,
 29]]

dist = dist.drop_duplicates(subset = ['SCG-District']) 
dist.to_excel(r"district.xlsx", index = False )



list(Actuals_AF.columns)
projActuals = Actuals_AF.copy()
projActuals.columns

# projActuals = projActuals.drop(['SCG-District', 'Type of Work', 'Abandon', 1,2,4,5,6,7,8,9,10,15,16,17,18,20,21,22,23,25,26,28], axis=1)
projActuals = projActuals.drop(['SCG-District', 'Type of Work', 'Abandon'], axis=1)

Actuals_AF.to_excel(r"C:\Users\PriyankRao\OneDrive - Sempra Energy\Data\Bechmarking_1\Main Files\Benchmarking Database\temp file\actuals.xlsx", index = False)
projActuals.to_excel("est.xlsx", index = False)

projActuals.columns
X = projActuals.drop(['Actuals'], axis=1)

Y =projActuals['Actuals']

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size =0.7, test_size = 0.3, random_state = 100)

X_train
y_train

X_train.dtypes
y_train

X_train_sm = sm.add_constant(X_train)

lr = sm.OLS(y_train, X_train_sm).fit()

lr.params
lr.summary()
list(lr.params.index)
lr.params.values



X_test_sm = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test_sm)
y_test
os.getcwd()    
y_test_pred.to_excel(r"Prediction_1.xlsx")
y_test.to_excel("y_1.xlsx")
pre = pd.DataFrame(y_test_pred, columns={'Prediction'})

dataframe =  pd.merge(Actuals_AF, X_test_sm, left_index = True, right_index = True, how = 'right')
dataframe = pd.merge(dataframe, y_test, left_index = True, right_index = True, how = 'right')
dataframe = pd.merge(dataframe, pre, left_index = True, right_index = True, how = 'right')
dataframe.to_excel(r"PredictionResultDF.xlsx")

y_pre = y_test_pred.copy()
y_pre = y_pre.drop(index = [65])
y = y_test.copy()
y = y.drop(index = [65])
model_eval = pd.DataFrame(index = ['MLR_TR'], columns = ['RMSE'] )
model_eval.loc['MLR_TR', 'RMSE'] = np.sqrt(mean_squared_error(y, y_pre))

df_para = pd.DataFrame(data = lr.params.values, index = lr.params.index)
df_para.to_excel(r"Parameter.xlsx")
