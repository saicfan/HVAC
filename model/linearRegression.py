#import numpy as np
import pandas as pd
#import statsmodels.api as sm
from sklearn import linear_model
from preProcessing import freq_table
from matplotlib import pyplot as plt


columns = ['season', 'Weekday', 'Hour of Day', 'Trip Elapsed Minutes', 'Outside Temp', 'Inside Temp',
           'Left Temp Change', 'Wind Level Change', 'Power Mode', 'Heading', 'Fuel', 'Avg. Speed',
           'Avg. RPM', 'Wind Level']


def generate_train(train_df):
    train_df['season'] = train_df['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
    x_train = train_df[columns]
    y_train = train_df['Left Temp']
    return x_train, y_train


def linear_regression(x_train, y_train):
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    predictions = lm.predict(x_test)
    plt.scatter(predictions, y_test)
    plt.ylabel('True Values')
    plt.xlabel('Predictions')
    plt.title('Vehicle %s Model - Accuracy Score: %9.8f' % (vehicle, model.score(x_test, y_test)))


for j in range(len(freq_table)):
    vehicle = freq_table.index[j]
    record = freq_table.values[j]

    train = pd.read_csv('../data/v%s_%d/left_temp_train.csv' % (vehicle, record))
    train = train.dropna(axis=0)

    if len(train) >= 600:
        test = pd.read_csv('../data/v%s_%d/left_temp_test.csv' % (vehicle, record))
        test = test.dropna(axis=0)
        test['season'] = test['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
        x_test = test[columns]
        y_test = test['Left Temp']

        for i in [200, 400, len(train)]:
            t = train[0: i]
            x, y = generate_train(t)
            linear_regression(x, y)
            plt.savefig('../output/v%s_%d/model_%d.jpg' % (vehicle, record, i))

#x = sm.add_constant(x)

#model = sm.OLS(y, x).fit()
#predictions = model.predict(x) # make the predictions by the model

#print(model.summary())
#print(vehicle, record)
