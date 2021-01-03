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


# splitting for train/test in sklearn train_test_split style, testing 2020 year

def date_split(df):

    features = ['mean_5', 'mean_30', 'mean_365', 'std_5', 'std_30', 'std_365']
    label = '5. adjusted close'

    train = df[df['date'] < datetime(year=2020, month=1, day=1)]
    test = df[df['date'] >= datetime(year=2020, month=1, day=1)]

    X_train = train[features]
    X_test = test[features]
    y_train = train[label]
    y_test = test[label]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = date_split(cvx)


# list of models to test, linear regression and mlp regressor are the most effective models

rfr = RandomForestRegressor()
knn = KNeighborsRegressor()
lr = LinearRegression()
nn = MLPRegressor()

models = [knn, lr, rfr, nn]

results = pd.DataFrame()
results['actual'] = y_test
results['date'] = cvx['date']

for model in enumerate(models):
    model[1].fit(X_train, y_train)
    predictions = model[1].predict(X_test)
    results[model[0]] = predictions
    print(model[1])
    print('-'*25)
    print('mae:', mean_absolute_error(y_test, predictions))
    print('mse:', mean_squared_error(y_test, predictions))
    print('r2:', r2_score(y_test, predictions))
    print('\n')


# visualizing predictions across prices and time

sns.pairplot(results, y_vars='actual')
plt.show()

sns.lineplot(x=results['date'], y=results['actual'])
sns.lineplot(x=results['date'], y=results[0])
sns.lineplot(x=results['date'], y=results[1])
sns.lineplot(x=results['date'], y=results[2])
sns.lineplot(x=results['date'], y=results[3])
plt.xticks(rotation=30)
plt.legend(['actual', 'knn', 'lr', 'rfr', 'nn'])
plt.show()