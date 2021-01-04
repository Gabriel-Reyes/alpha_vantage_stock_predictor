from setup import av_key

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from alpha_vantage.timeseries import TimeSeries


# current evaluation is for Chevron

ts = TimeSeries(key=av_key, output_format='pandas')

cvx, info_cvx = ts.get_daily_adjusted('CVX', outputsize='full')
cvx = cvx.reset_index()


# adding on basic historical trends, previous work has shown that this is a good start to a stock estimator

def historical_trends(df):
    target = '5. adjusted close'

    df = df.sort_values('date')

    df['mean_5'] = df[target].rolling(window=5).mean().shift()
    df['mean_30'] = df[target].rolling(window=30).mean().shift()
    df['mean_365'] = df[target].rolling(window=365).mean().shift()
    df['std_5'] = df[target].rolling(window=5).std().shift()
    df['std_30'] = df[target].rolling(window=30).std().shift()
    df['std_365'] = df[target].rolling(window=365).std().shift()

    return df.dropna()

cvx = historical_trends(cvx)


# splitting for train/test in sklearn train_test_split style, sliding window for quarter

def date_split(df, year, quarter):

    features = ['mean_5', 'mean_30', 'mean_365', 'std_5', 'std_30', 'std_365']
    label = '5. adjusted close'

    if quarter == 1:
        month = 1
    if quarter == 2:
        month = 4
    if quarter == 3:
        month = 7
    if quarter == 4:
        month = 10

    train = df[df['date'] < datetime(year=year, month=month, day=1)]
    test = df[(df['date'].dt.year == year) & (df['date'].dt.quarter == quarter)]

    X_train = train[features]
    X_test = test[features]
    y_train = train[label]
    y_test = test[label]

    return X_train, X_test, y_train, y_test


# list of models to test, linear regression and mlp regressor are the most effective models

lr = LinearRegression()
nn = MLPRegressor()

models = [['lr', lr], ['nn', nn]]
quarters = [1, 2, 3, 4]

prediction_list = []

for model in models:

    for quarter in quarters:
        X_train, X_test, y_train, y_test = date_split(cvx, 2020, quarter)

        results = pd.DataFrame()
        results['actual'] = y_test
        results['date'] = cvx['date']
        results['model'] = model[0]

        model[1].fit(X_train, y_train)
        prediction = model[1].predict(X_test)
        results['predictions'] = prediction

        prediction_list.append(results)

        print(model[0])
        print('-'*25)
        print('quarter:', quarter)
        print('-'*15)
        print('mae:', mean_absolute_error(y_test, prediction))
        print('mse:', mean_squared_error(y_test, prediction))
        print('r2:', r2_score(y_test, prediction))
        print('\n')

predictions = pd.concat(prediction_list)


# combining & visualizing quarterly predictions for each model

for model in models:
    df = predictions[predictions['model'] == model[0]]
    sns.lineplot(df['date'], df['actual'])
    sns.lineplot(df['date'], df['predictions'])
    plt.legend(['actual', model[0]])
    plt.show()