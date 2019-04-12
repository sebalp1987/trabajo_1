import pandas as pd
import STRING
from linearmodels import RandomEffects
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

df = pd.read_csv(STRING.file_panel_data, sep=';')
df = df[df['DATE'] < '2014-02-01']
df = df.set_index(['DATE', 'MARKET'])
print(df.columns)

# Hausman Test to decided FE or RE

# Fixed Effects

# Random Effects
exog_vars = ['PPORTUGAL', 'Q', 'QPOR', 'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM', 'TME', 'TMAX', 'TMIN', 'PP',
             'WORKDAY', 'SUMMER', 'WINTER', 'NULL_PRICE', 'LITINIT']
exog = sm.add_constant(df[exog_vars])
mod = RandomEffects(df['P'], exog)
re_res = mod.fit()
print(re_res)

# Fixed Effect
exog_vars = ['PPORTUGAL', 'Q', 'QPOR', 'PRICE_OIL', 'PRICE_GAS', 'RISK_PREMIUM', 'TME', 'TMAX', 'TMIN', 'PP',
             'WORKDAY', 'SUMMER', 'WINTER', 'NULL_PRICE', 'LITINIT',  'TIMEEF_2013', 'FE_SPAIN']
exog = sm.add_constant(df[exog_vars])
mod = sm.OLS(endog=df['P'], exog=df[exog_vars])
re_res = mod.fit()
print(re_res.summary())

# DIF IN DIF
# y = beta_0 + beta_1*T + beta_2*S + beta_3*T*S + error
# T is = 1 when treatment period
# S is = 1 when trated group
# beta_3 is the Dif in Dif effect

df['T'] = df['DUMMY_FORW_10_DAY']
df['S'] = df['FE_SPAIN']
df['T*S'] = df['T']*df['S']

exog_vars = [ 'Q', 'TME', 'PP',
             'WORKDAY', 'WINTER', 'NULL_PRICE', 'LITINIT', 'T', 'S', 'T*S', 'PRICE_GAS', 'TREND', 'AR1', 'AR2',
              'AR3', 'AR4', 'AR5', 'AR6', 'AR7']
for col in exog_vars:
    if col not in ['WORKDAY', 'SUMMER', 'WINTER', 'NULL_PRICE', 'LITINIT', 'T', 'S', 'T*S', 'TREND']:
        df[col] = np.log(df[col] + 1)
df['P'] = np.log(df['P'] + 1)
print(df.shape)
df = df[exog_vars + ['P']].dropna(axis=0)
print(df.shape)

# QSPAIN predict
predictq = sm.OLS(endog=df['Q'], exog=df[['WORKDAY', 'TME', 'PP', 'WINTER']]).fit()
predictq = predictq.predict(df[['WORKDAY', 'TME', 'PP', 'WINTER']])
predictq = pd.DataFrame(predictq, columns=['Q_IV'])
df = pd.concat([df.drop(['Q', 'WORKDAY', 'TME', 'PP', 'WINTER'], axis=1), predictq], axis=1)
exog = sm.add_constant(df)
mod = RandomEffects(df['P'], df.drop('P', axis=1))
re_res = mod.fit()
print(re_res)

vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(df.drop('P', axis=1).values, i) for i in
              range(df.drop('P', axis=1).shape[1])]
vif['features'] = df.drop('P', axis=1).columns
print(vif)