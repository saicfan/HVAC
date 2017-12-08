import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.lines as mlines
from preProcessing import freq_table
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

mse = pd.DataFrame(index=[200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800],
                   columns=['Left Temp', 'Wind Level'])

temp_columns = ['season', 'Weekday', 'Hour of Day', 'Trip Elapsed Minutes', 'Outside Temp', 'Inside Temp',
                'Left Temp Change', 'Wind Level Change', 'Power Mode', 'Heading', 'Fuel', 'Avg. Speed', 'Avg. RPM']

wind_columns_train = ['season', 'Weekday', 'Hour of Day', 'Trip Elapsed Minutes', 'Outside Temp', 'Inside Temp',
                      'Left Temp Change', 'Wind Level Change', 'Power Mode', 'Heading', 'Fuel', 'Avg. Speed',
                      'Avg. RPM', 'Left Temp']

wind_columns_test = ['season', 'Weekday', 'Hour of Day', 'Trip Elapsed Minutes', 'Outside Temp', 'Inside Temp',
                     'Left Temp Change', 'Wind Level Change', 'Power Mode', 'Heading', 'Fuel', 'Avg. Speed',
                     'Avg. RPM', 'left temp']


def temp_generate_train(train_df):
    train_df['season'] = train_df['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
    x_train = train_df[temp_columns]
    y_train = train_df['Left Temp']
    return x_train, y_train


def wind_generate_train(train_df):
    train_df['season'] = train_df['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
    x_train = train_df[wind_columns_train]
    y_train = train_df['Wind Level']
    return x_train, y_train


def temp_qq(ax, y, y_predictions):
    ax.add_line(mlines.Line2D([15, 26], [15, 26], color='red'))
    ax.scatter(y_predictions, y)
    major_ticks = np.arange(15, 27, 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    plt.ylabel('True Values')
    plt.xlabel('Predictions')
    plt.title('Vehicle %s Left Temp - Mean Squared Error: %9.8f' % (vehicle, mean_squared_error(y, y_predictions)))


def wind_qq(ax, y, y_predictions):
    ax.add_line(mlines.Line2D([0, 9], [0, 9], color='red'))
    ax.scatter(y_predictions, y)
    major_ticks = np.arange(0, 9, 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    plt.ylabel('True Values')
    plt.xlabel('Predictions')
    plt.title('Vehicle %s Wind Level - Mean Squared Error: %9.8f' % (vehicle, mean_squared_error(y, y_predictions)))


for j in range(len(freq_table)):
    vehicle = freq_table.index[j]
    record = freq_table.values[j]

    train = pd.read_csv('../data/v%s_%d/wind_level_train.csv' % (vehicle, record))
    train = train.dropna(axis=0)

    if len(train) >= 800:
        test = pd.read_csv('../data/v%s_%d/wind_level_test.csv' % (vehicle, record))
        test = test.dropna(axis=0)
        test['season'] = test['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
        x1_test = test[temp_columns]
        y1_test = test['Left Temp']

        for k in [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]:
            w = train[0: k]

            xx, yy = temp_generate_train(w)
            lm_1 = linear_model.LinearRegression()
            model_1 = lm_1.fit(xx, yy)
            temp_predictions = lm_1.predict(x1_test)

            fig = plt.figure()
            ax_1 = fig.add_subplot(211)
            temp_qq(ax_1, y1_test, temp_predictions)
            mse['Left Temp'].loc[k] = mean_squared_error(y1_test, temp_predictions)

            test['left temp'] = temp_predictions
            x2_test = test[wind_columns_test]
            y2_test = test['Wind Level']

            xxx, yyy = wind_generate_train(w)
            lm_2 = linear_model.LinearRegression()
            model_2 = lm_2.fit(xxx, yyy)
            wind_predictions = lm_2.predict(x2_test)

            ax_2 = fig.add_subplot(212)
            wind_qq(ax_2, y2_test, wind_predictions)
            mse['Wind Level'].loc[k] = mean_squared_error(y2_test, wind_predictions)

            fig.tight_layout()
            plt.savefig('../output/v%s_%d/Cascading_%d.png' % (vehicle, record, k))

        ax = mse.plot(kind='bar', color=['r', 'b'], title='Vehicle %s Cascading Prediction Set' % vehicle)
        ax.set_xlabel('Event History Trained')
        ax.set_ylabel('Mean Squared Error')
        plt.savefig('../output/v%s_%d/cascading.jpg' % (vehicle, record))


