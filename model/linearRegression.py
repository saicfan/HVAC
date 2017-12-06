import numpy as np
import pandas as pd
# import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.lines as mlines
from preProcessing import freq_table
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


temp_columns = ['season', 'Weekday', 'Hour of Day', 'Trip Elapsed Minutes', 'Outside Temp', 'Inside Temp',
                'Left Temp Change', 'Wind Level Change', 'Power Mode', 'Heading', 'Fuel', 'Avg. Speed',
                'Avg. RPM', 'Wind Level']

wind_columns = ['season', 'Weekday', 'Hour of Day', 'Trip Elapsed Minutes', 'Outside Temp', 'Inside Temp',
                'Left Temp Change', 'Wind Level Change', 'Power Mode', 'Heading', 'Fuel', 'Avg. Speed',
                'Avg. RPM', 'Left Temp']


def temp_generate_train(train_df):
    train_df['season'] = train_df['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
    x_train = train_df[temp_columns]
    y_train = train_df['Left Temp']
    return x_train, y_train


def wind_generate_train(train_df):
    train_df['season'] = train_df['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
    x_train = train_df[wind_columns]
    y_train = train_df['Wind Level']
    return x_train, y_train


def temp_linear_regression(x_train, y_train):
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    predictions = lm.predict(temp_x_test)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_line(mlines.Line2D([15, 26], [15, 26], color='red'))
    ax.scatter(predictions, temp_y_test)
    major_ticks = np.arange(15, 27, 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    plt.ylabel('True Values')
    plt.xlabel('Predictions')
    fig.suptitle('Vehicle %s Left Temp - Accuracy Score: %9.8f' % (vehicle, model.score(temp_x_test, temp_y_test)))
    plt.title('Mean Squared Error: %9.8f' % mean_squared_error(temp_y_test, predictions))


def wind_linear_regression(x_train, y_train):
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    predictions = lm.predict(wind_x_test)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_line(mlines.Line2D([0, 9], [0, 9], color='red'))
    ax.scatter(predictions, wind_y_test)
    major_ticks = np.arange(0, 9, 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    plt.ylabel('True Values')
    plt.xlabel('Predictions')
    fig.suptitle('Vehicle %s Wind Level - Accuracy Score: %9.8f' % (vehicle, model.score(wind_x_test, wind_y_test)))
    plt.title('Mean Squared Error: %9.8f' % mean_squared_error(wind_y_test, predictions))


for j in range(len(freq_table)):
    vehicle = freq_table.index[j]
    record = freq_table.values[j]

    temp_train = pd.read_csv('../data/v%s_%d/left_temp_train.csv' % (vehicle, record))
    temp_train = temp_train.dropna(axis=0)

    if len(temp_train) >= 800:
        temp_test = pd.read_csv('../data/v%s_%d/left_temp_test.csv' % (vehicle, record))
        temp_test = temp_test.dropna(axis=0)
        temp_test['season'] = temp_test['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
        temp_x_test = temp_test[temp_columns]
        temp_y_test = temp_test['Left Temp']

        for i in [200, 400, 600, 800]:
            t = temp_train[0: i]
            x, y = temp_generate_train(t)
            temp_linear_regression(x, y)
            plt.savefig('../output/v%s_%d/LeftTemp_%d.jpg' % (vehicle, record, i))

    wind_train = pd.read_csv('../data/v%s_%d/wind_level_train.csv' % (vehicle, record))
    wind_train = temp_train.dropna(axis=0)

    if len(wind_train) >= 800:
        wind_test = pd.read_csv('../data/v%s_%d/wind_level_test.csv' % (vehicle, record))
        wind_test = wind_test.dropna(axis=0)
        wind_test['season'] = wind_test['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
        wind_x_test = wind_test[wind_columns]
        wind_y_test = wind_test['Wind Level']

        for k in [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]:
            w = wind_train[0: k]
            xx, yy = wind_generate_train(w)
            wind_linear_regression(xx, yy)
            plt.savefig('../output/v%s_%d/WindLevel_%d.jpg' % (vehicle, record, k))

#x = sm.add_constant(x)

#model = sm.OLS(y, x).fit()
#predictions = model.predict(x) # make the predictions by the model

#print(model.summary())
#print(vehicle, record)
