import pandas as pd
import STRING
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plot
from scipy.stats import normaltest, shapiro
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import linear_model
import resource.temporal_statistics as sts
from preprocessing import stepwise_reg
from config import config
from statsmodels.sandbox.regression.gmm import IV2SLS
from sklearn.feature_selection import f_regression
from linearmodels.iv import IV2SLS as iv2reg
sns.set()

df = pd.read_csv(STRING.file_detrend, sep=';')

df = df.set_index('DATE')

df['C'] = pd.Series(1, index=df.index)

df['LQ1'] = df['QDIF'].shift(1)
df['LQ2'] = df['QDIF'].shift(2)


# AR Component
df['ar.L1.PSPAIN'] = df['PSPAIN'].shift(1)
df['ar.L2.PSPAIN'] = df['PSPAIN'].shift(2)
df['ar.L4.PSPAIN'] = df['PSPAIN'].shift(4)
df['ar.L6.PSPAIN'] = df['PSPAIN'].shift(6)

# MA component
width = 1
lag = df['PSPAIN'].shift(width)
window = lag.rolling(window=width + 2)
means = pd.DataFrame(window.mean().values, columns=['ma.L1.PSPAIN'], index=df.index)
df = pd.concat([df, means], axis=1)


# X, Y
y = df[['PSPAIN']]
df['WORKDAY'] = df['WORKDAY'] + 1
df['WORKDAY'] = np.log(df['WORKDAY'])-np.log((df['INDEX'] + 5))
df['TME'] = df['TME_MADRID']**2 / (df['TAVG']**2 + 0.001)


instruments = ['WORKDAY', 'TME', 'WINTER']
inst_square = []
for inst in instruments:
    df[inst] = (df[inst] - df[inst].shift(1)) - (df[inst].shift(1) - df[inst].shift(2))
instruments += []

df = df.dropna(axis=0)
print(df.shape)
variable_used = ['D1_sum(CICLO_COMBINADO)', 'LQ1', 'LQ2', 'D1_sum(FUEL_PRIMA)',
                 'D1_sum(HIDRAULICA_CONVENC)', 'DUMMY_FORW_45_DAY', 'ma.L1.PSPAIN', 'ar.L1.PSPAIN', 'ar.L2.PSPAIN',
                 'ar.L4.PSPAIN', 'ar.L6.PSPAIN']

variable_instrumented = ['QDIF']

"""
To conduct IV estimations, we need to have instrumental
variables (or instruments in short) that are (R1) uncorrelated with u but (R2) partially
and sufficiently strongly correlated with y2 once the other independent variables are
controlled for.
In practice, we can test the second requirement (b), but we can not test the first
requirement (a) because u is unobservable.

To test the second requirement (b), we need
to express a reduced form equation of y2 with all of exogenous variables. Exogenous
variables include all of independent variables that are not correlated with the error term
and the instrumental variable, z.
For the instrumental variable to satisfy the second requirement (R2), the estimated
coefficient of z must be significant. 
"""

# MODEL
# First Stage: Reduced Form
# QDIF
first_reg1 = sm.OLS(endog=df[variable_instrumented], exog=df[variable_used + instruments])
results = first_reg1.fit()
print(results.summary())


"""
In the first stage
regression, we should conduct a F-test on all instruments to see if instruments are jointly
significant in the endogenous variable, y2. 
"""
ftest = sm.OLS(endog=df[variable_instrumented], exog=df[instruments])
results = ftest.fit()

print('F-Statistic', results.fvalue)
print('pvalue', results.f_pvalue)
print(results.summary())

# Second Stage
y = df[['PSPAIN']]
reg = IV2SLS(endog=y, exog=df[variable_used + variable_instrumented], instrument=df[variable_used + instruments])
results = reg.fit()
print(results.summary())

# Second stage
mod = iv2reg(y, df[variable_used], df[variable_instrumented], df[instruments])
res = mod.fit(cov_type='unadjusted')
print(res.durbin())
print(res.wu_hausman())
print(res.wooldridge_regression)
print(res.sargan)
print(res.anderson_rubin)
print(res.basmann_f)