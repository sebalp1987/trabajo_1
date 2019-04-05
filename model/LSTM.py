import pandas as pd
import STRING
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error, recall_score, precision_score, fbeta_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import numpy as np
import os
from scipy.stats import norm
from scipy.stats import multivariate_normal
import seaborn as sns
from keras import losses
import math
sns.set()

np.random.seed(42)

"""
LSTM es capaz de recordar a lo largo de secuencias largas. Normalmente, el estado de la red se resetea despues
de cada batch termina de entrenar (epochs). Podemos ganar mayor control determinando cuándo la red se resetea haciendo
que el layer del LSTM sea "stateful". Esto implica que podemos construir estados sobre todo el training e incluso
mantenter este estado si es necesario.
Requiere que los datos no esten shuffle. Esto es porque queremos que aprenda de la secuencia de las observaciones y no 
que haga un shuffle dentro del epoch.
Requiere explícitamente resetear la red despues de cada epoch (models.reset_states()). Esto implica que tenemos
que crear nuestro propio loop fuera de los epochs y dentro de los epochs.
"""


train_normal = pd.read_csv(STRING.train, sep=';', parse_dates=['DATE'])
valid_normal = pd.read_csv(STRING.valid_normal, sep=';', parse_dates=['DATE'])
valid_mixed = pd.read_csv(STRING.valid_mixed, sep=';', parse_dates=['DATE'])
test = pd.read_csv(STRING.test, sep=';', parse_dates=['DATE'])

train_normal['CONTROL'] = pd.Series(0, index=train_normal.index)
valid_normal['CONTROL'] = pd.Series(1, index=valid_normal.index)
valid_mixed['CONTROL'] = pd.Series(2, index=valid_mixed.index)
test['CONTROL'] = pd.Series(3, index=test.index)

df = pd.concat([train_normal, valid_normal, valid_mixed, test], axis=0)
df = df.sort_values(by=['DATE'], ascending=[True]).reset_index(drop=True)

cols = df.columns.values.tolist()
print(cols)
df = df.set_index('DATE')

# WE CREATE A LAG
look_back = 7
for col in df.columns.values.tolist():
    if col != 'CONTROL':
        for i in range(1, look_back + 1, 1):
            df['L' + str(i) + '_' + col] = df[col].shift(i)

# LO PRIMERO ES ORDENAR PARA QUE QUEDEN LOS LAGS DE UN LADO Y LA VARIABLE Y DEL OTRO
cols = df.columns.values.tolist()
df = df[
    ['PSPAIN'] + [x for x in cols if x.startswith('L') and 'PSPAIN' in x] + [x for x in cols if x.startswith('L') and
                                                                             not 'PSPAIN' in x] +
    ['CONTROL', 'TARGET']]

df = df.dropna()

# WE REINDEX (because of NAN)
df = df.reset_index(drop=False)
train_normal = df[df['CONTROL'] == 0]
valid_normal['CONTROL'] = df[df['CONTROL'] == 1]
valid_mixed['CONTROL'] = df[df['CONTROL'] == 2]
test['CONTROL'] = df[df['CONTROL'] == 3]
df = df.set_index('DATE')

# We normalize
df_c = df[['CONTROL', 'TARGET']]
df_c = df_c.reset_index(drop=False)
df = df.reset_index(drop=True)
df = df.drop(['CONTROL', 'TARGET'], axis=1)
cols = df.columns.values.tolist()
scaler = MinMaxScaler(feature_range=(-1, 1))
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=cols)
features_name = [x for x in cols if x.startswith('L')]
features = int(len(features_name) / look_back)

df = pd.concat([df, df_c], axis=1)
train_normal = df[df['CONTROL'] == 0]
valid_normal = df[df['CONTROL'] == 1]
valid_mixed = df[df['CONTROL'] == 2]
test = df[df['CONTROL'] == 3]
train_normal = train_normal.drop(['CONTROL', 'DATE', 'TARGET'], axis=1)
valid_normal = valid_normal.drop(['CONTROL', 'DATE', 'TARGET'], axis=1)
valid_mixed = valid_mixed.drop(['CONTROL', 'DATE', 'TARGET'], axis=1)
test = test.drop(['CONTROL', 'DATE', 'TARGET'], axis=1)

# LO SEGUNDO ES DEFINIR COMO "X" TODOS LOS LAGS y COMO "Y" AL PESPANIA
print(train_normal.columns.values.tolist())
batch_size = 10
train_batch = int(math.floor(train_normal.shape[0] / batch_size) * batch_size)
valid_normal_batch = int(math.floor(valid_normal.shape[0] / batch_size) * batch_size)
valid_mixed_batch = int(math.floor(valid_mixed.shape[0] / batch_size) * batch_size)
test_batch = int(math.floor(test.shape[0] / batch_size) * batch_size)
print(train_batch)
train_normal_x, train_normal_y = train_normal.values[:train_batch, 1:], train_normal.values[:train_batch, 0]
valid_normal_x, valid_normal_y = valid_normal.values[:valid_normal_batch, 1:], valid_normal.values[:valid_normal_batch,
                                                                               0]
valid_mixed_x, valid_mixed_y = valid_mixed.values[:valid_mixed_batch, 1:], valid_mixed.values[:valid_mixed_batch, 0]
test_x, test_y = test.values[:test_batch, 1:], test.values[: test_batch, 0]

# We need the data as [samples, time_steps, features] - Now is [samples, features]
train_normal_x = np.reshape(train_normal_x, (train_normal_x.shape[0], look_back,
                                             features))  # TERCERO: TENEMOS QUE DEFINIR SAMPLES, TIME_STEPS=LAGS, FEATURES=FEATURES
valid_normal_x = np.reshape(valid_normal_x, (valid_normal_x.shape[0], look_back, features))
valid_mixed_x = np.reshape(valid_mixed_x, (valid_mixed_x.shape[0], look_back, features))
test_x = np.reshape(test_x, (test_x.shape[0], look_back, features))

print(train_normal_x.shape, train_normal_y.shape, valid_normal_x.shape, valid_normal_y.shape)

# LSTM
model = Sequential()
model.add(LSTM(100, batch_input_shape=(batch_size, train_normal_x.shape[1], train_normal_x.shape[2]), stateful=True,
               return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.compile(loss=losses.mean_squared_error, optimizer='adagrad')
print("Inputs: " + str(model.input_shape))
print("Outputs: " + str(model.output_shape))
print("Actual input: " + str(train_normal_x.shape))
print("Actual output:" + str(train_normal_y.shape))
early_stopping = EarlyStopping(patience=2)

for i in range(50):
    model.fit(train_normal_x, train_normal_y, epochs=1, batch_size=batch_size, validation_data=(valid_normal_x,
                                                                                                valid_normal_y),
              verbose=2, shuffle=False, callbacks=[early_stopping])
    model.reset_states()
# PLOT
'''
plot.plot(history.history['loss'], label='train')
plot.plot(history.history['val_loss'], label='test')
plot.legend()
plot.show()
'''

# PREDICTIONS
cols.remove('PSPAIN')
# Train
yhat = model.predict(train_normal_x, batch_size=batch_size)
print(yhat)
model.reset_states()
train_normal_x = train_normal_x.reshape(train_normal_x.shape[0], look_back * features)
print(yhat.shape)
print(train_normal_x.shape)
inv_yhat = np.concatenate((yhat, train_normal_x), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
train_prediction = inv_yhat[:, 0]

train_normal_y = train_normal_y.reshape(len(train_normal_y), 1)
inv_y = np.concatenate((train_normal_y, train_normal_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
train_true = inv_y[:, 0]

# Valid Mixed
yhat = model.predict(valid_mixed_x, batch_size=batch_size)
valid_mixed_x = valid_mixed_x.reshape(valid_mixed_x.shape[0], look_back * features)
inv_yhat = np.concatenate((yhat, valid_mixed_x), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
valid2_prediction = inv_yhat[:, 0]

valid_mixed_y = valid_mixed_y.reshape(len(valid_mixed_y), 1)
inv_y = np.concatenate((valid_mixed_y, valid_mixed_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
valid2_true = inv_y[:, 0]

# Test
yhat = model.predict(test_x, batch_size=batch_size)
test_x = test_x.reshape(test_x.shape[0], look_back * features)
inv_yhat = np.concatenate((yhat, test_x), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
test_prediction = inv_yhat[:, 0]

test_y = test_y.reshape(len(test_y), 1)
inv_y = np.concatenate((test_y, test_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
test_true = inv_y[:, 0]

# PLOT
rsme = np.sqrt(mean_squared_error(train_true, train_prediction))
print('RMSE TRAIN %.2f' % rsme)
mae = np.abs(train_true - train_prediction)
plot.figure(figsize=(15, 5))
plot.plot(train_true, label='true')
plot.plot(train_prediction, label='predicted')
plot.plot(mae, label='rmse')
plot.legend()
plot.title('Training Using 7 Timesteps')
plot.show()

# ERROR VECTOR TRAIN
train_error_vector = train_true - train_prediction
mean = np.mean(train_error_vector, axis=0)
cov = np.cov(train_error_vector, rowvar=False)
p_values = multivariate_normal.logpdf(train_error_vector, mean, cov)
plot.figure(figsize=(15, 5))
plot.plot(p_values)
plot.title('p-values Train')
plot.show()

# ERROR VECTOR OPTIMIZING THRESHOLD (VALID_2)
v2_error_vector = valid2_true - valid2_prediction
# v2_error_vector[v2_error_vector < 0] = 0

v2_p_values = multivariate_normal.logpdf(v2_error_vector, mean, cov)

predicted_valid = pd.DataFrame(valid2_prediction, columns=['predicted'])
v2_pvalues_df = pd.DataFrame(v2_p_values, columns=['LogPD'])
valid_mixed_c = df_c[df_c['CONTROL'] == 2].reset_index(drop=True)
test_c = df_c[df_c['CONTROL'] == 3].reset_index(drop=True)
valid_mixed_df = pd.concat([valid_mixed_c[['CONTROL', 'TARGET']], predicted_valid, v2_pvalues_df], axis=1)
valid_mixed_df['TARGET'] = valid_mixed_df['TARGET'].map(int)

thresholds = np.linspace(-50.0, 0.0, 1000)

scores = []

for threshold in thresholds:
    y_hat = [1 if e < threshold else 0 for e in valid_mixed_df.LogPD.values]
    y_hat = list(map(int, y_hat))

    scores.append([
        recall_score(y_pred=y_hat, y_true=valid_mixed_df.TARGET.values),
        precision_score(y_pred=y_hat, y_true=valid_mixed_df.TARGET.values),
        fbeta_score(y_pred=y_hat, y_true=valid_mixed_df.TARGET.values,
                    beta=1)])

scores = np.array(scores)
threshold = thresholds[scores[:, 2].argmax()]
threshold = -5.0
print('final Threshold SET by Valid Mixed', threshold)
predicted = [1 if e < threshold else 0 for e in valid_mixed_df.LogPD.values]
predicted = list(map(int, predicted))

print('PRECISION VALID MIXED', precision_score(valid_mixed_df.TARGET.values, predicted))
print('RECALL VALID MIXED', recall_score(valid_mixed_df.TARGET.values, predicted))
print('FBSCORE VALID MIXED', fbeta_score(valid_mixed_df.TARGET.values, predicted, beta=1))

# ERROR VECTOR TEST
rsme = np.sqrt(mean_squared_error(test_true, test_prediction))
print('RMSE TEST %.2f' % rsme)
test_error_vector = test_true - test_prediction
test_error_vector[test_error_vector < 0] = 0
test_p_values = multivariate_normal.logpdf(test_error_vector, mean, cov)

# PLOTS
# Plot v2 Pvalues
plot.figure()
plot.plot(v2_p_values, label='log PD', color=sns.xkcd_rgb["dark teal"])
plot.axhline(y=threshold, ls='dashed', label='Threshold', color=sns.xkcd_rgb["dark teal"])
plot.legend(bbox_to_anchor=(1, .45), borderaxespad=0.)
plot.xlabel("Time step")
plot.ylabel("Log PD")
plot.title("Validation Mixed p-values")
# plot.show()
plot.close()

# FINAL PLOT VALIDATION 2
f = plot.figure(figsize=(20, 10))
plot.subplots_adjust(hspace=0.1)
v2_below_threshold = np.where(v2_p_values <= threshold)
print(v2_below_threshold)
ax1 = plot.subplot(211)
ax1.plot(valid2_true, label='True Value', color=sns.xkcd_rgb["denim blue"])
ax1.plot(valid2_prediction, ls='dashed', label='Predicted Value', color=sns.xkcd_rgb["medium green"])
ax1.plot(abs(valid2_true - valid2_prediction), label='Error',
         color=sns.xkcd_rgb["pale red"])  # CAMBIAR ESTOS A LA FUNCION DE PERDIDA DE LSTM
for column in v2_below_threshold[0]:
    ax1.axvline(x=column, color=sns.xkcd_rgb["dark peach"], alpha=.5)

ax1.axvline(x=v2_below_threshold[0][-1], color=sns.xkcd_rgb["dark peach"], alpha=.5)
# for row in v2_true_anomalies:
#    plot.plot(row, validation2_true[row], 'r.', markersize=20.0)
ax1.legend(bbox_to_anchor=(1.02, .3), borderaxespad=0., frameon=True)
ax1.set_xticklabels([])
plot.ylabel("Power Price")
plot.title("Validation Mixed. Using 7 timestep")

# plot v2 log PD
ax2 = plot.subplot(212)
values_date = np.arange(0, len(valid_mixed_c.index), 50)
valid_mixed_c['DATE'] = valid_mixed_c['DATE'].astype(str)
df_dates_xticks = valid_mixed_c[valid_mixed_c.index.isin(values_date)]
ax2.plot(v2_p_values, label='Log PD', color=sns.xkcd_rgb["dark teal"])
anomalies = valid_mixed_c[valid_mixed_c['TARGET'] == 1]
anomalies = anomalies.index.values.tolist()
print(anomalies)
for i in anomalies:
    ax2.axvline(x=i, color=sns.xkcd_rgb["dark peach"], alpha=.5)
ax2.axhline(y=threshold, ls='dashed', label='Threshold', color=sns.xkcd_rgb["dark teal"])
ax2.legend(bbox_to_anchor=(1, .3), borderaxespad=0., frameon=True)
plot.ylabel("Log PD")
ax2.set_xticklabels(df_dates_xticks['DATE'], rotation=30, fontsize=8)
plot.xticks(values_date)
plot.title("Validation Mixed p-values")

# Set up the xlabel and xtick
# xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
plot.xlabel("Time step")
plot.show()

print(v2_below_threshold[0])
anomaly_dates_valid = valid_mixed_c[valid_mixed_c.index.isin(v2_below_threshold[0])]
print(anomaly_dates_valid)

# TEST EVALUATION
test_below_threshold = np.where(test_p_values <= threshold)
f = plot.figure(figsize=(20, 10))
plot.subplots_adjust(hspace=0.01)

ax1 = plot.subplot(211)
ax1.plot(test_true, label='True Value', color=sns.xkcd_rgb["denim blue"])
ax1.plot(test_prediction, ls='dashed', label='Predicted Value', color=sns.xkcd_rgb["medium green"])
ax1.plot(abs(test_true - test_prediction), label='Error', color=sns.xkcd_rgb["pale red"])
for column in test_below_threshold[0]:
    ax1.axvline(x=column, color=sns.xkcd_rgb["dark peach"], alpha=0.5)
# for row in test_true_anomalies:
#    plot.plot(row, test_true[row], 'r.', markersize=20.0)
ax1.legend(bbox_to_anchor=(1, 1), borderaxespad=0., frameon=True)
plot.ylabel("Power Price")
# plot.title("Test. Using 1 timestep")

ax2 = plot.subplot(212)
values_date = np.arange(0, len(test_c.index), 50)
test_c['DATE'] = test_c['DATE'].astype(str)
df_dates_xticks = test_c[test_c.index.isin(values_date)]
ax2.plot(test_p_values, label='Log PD', color=sns.xkcd_rgb["dark teal"])
anomalies = test_c[test_c['TARGET'] == 1]
anomalies = anomalies.index.values.tolist()
print(anomalies)
for i in anomalies:
    ax2.axvline(x=i, color=sns.xkcd_rgb["dark peach"], alpha=.5)
ax2.axhline(y=threshold, ls='dashed', label='Threshold', color=sns.xkcd_rgb["dark teal"])
ax2.legend(bbox_to_anchor=(1, .3), borderaxespad=0., frameon=True)
plot.ylabel("Log PD")
ax2.set_xticklabels(df_dates_xticks['DATE'], rotation=30, fontsize=8)
plot.xticks(values_date)
plot.title("test p-values")

# Set up the xlabel and xtick
# xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
plot.xlabel("Time step")
plot.show()

print(test_below_threshold[0])
anomaly_dates_test = test_c[test_c.index.isin(test_below_threshold[0])]
print(anomaly_dates_test)

predicted_test = pd.DataFrame(test_prediction, columns=['predicted'])
test_pvalues_df = pd.DataFrame(test_p_values, columns=['LogPD'])
test_df = pd.concat([test_c[['CONTROL', 'TARGET']], predicted_test, test_pvalues_df], axis=1)
test_df['TARGET'] = test_df['TARGET'].map(int)
predicted = [1 if e < threshold else 0 for e in test_df.LogPD.values]
predicted = list(map(int, predicted))
print('PRECISION TEST', precision_score(test_df.TARGET.values, predicted))
print('RECALL TEST', recall_score(test_df.TARGET.values, predicted))
print('FBSCORE TEST', fbeta_score(test_df.TARGET.values, predicted, beta=1))
