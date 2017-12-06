import os
import errno
import random
import pandas as pd

df = pd.read_csv('../data/bigml_5a2051a4460180175c007584.csv')
freq_table = df.Vehicle.value_counts()
freq_table.to_csv('../data/vehicle_list.csv')

for i in range(len(freq_table)):
    vehicle = freq_table.index[i]
    record = freq_table.values[i]

    data_directory = '../data/v%s_%d' % (vehicle, record)
    out_directory = '../output/v%s_%d' % (vehicle, record)
    try:
        os.makedirs(data_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs(out_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    vdf = df[df.Vehicle == freq_table.index[i]]
    vdf.index = range(0, len(vdf))
    vdf.to_csv('../data/v%s_%d/logs.csv' % (vehicle, record), index=False)
    s = vdf['Left Temp Class'].map({'Aggressive Cooling': 16.0, 'Aggressive Heating': 25.0})
    vdf['Left Temp'] = vdf['Left Temp'].fillna(s)

    wvdf = vdf.drop(columns=['Weather - Type', 'Weather - Temp', 'Weather - Wind Speed', 'Weather - Humidity',
                             'Left Temp Class', 'Wind Pattern'])
    wvdf = wvdf[wvdf['Wind Level'].isnull() == False]
    wvdf.index = range(0, len(wvdf))

    test_idx = random.sample(range(len(wvdf)), int(0.2 * len(wvdf)))
    train_idx = [i for i in range(len(wvdf)) if i not in test_idx]

    wind_train = wvdf[wvdf.index.isin(train_idx)]
    wind_train.to_csv('../data/v%s_%d/wind_level_train.csv' % (vehicle, record), index=False)

    wind_test = wvdf[wvdf.index.isin(test_idx)]
    wind_test.to_csv('../data/v%s_%d/wind_level_test.csv' % (vehicle, record), index=False)

    tvdf = vdf.drop(columns=['Weather - Type', 'Weather - Temp', 'Weather - Wind Speed', 'Weather - Humidity',
                             'Left Temp Class', 'Wind Pattern'])
    tvdf = tvdf[tvdf['Left Temp'].isnull() == False]
    tvdf.index = range(0, len(tvdf))

    test_index = random.sample(range(len(tvdf)), int(0.2 * len(tvdf)))
    train_index = [i for i in range(len(tvdf)) if i not in test_index]

    temp_train = tvdf[tvdf.index.isin(train_index)]
    temp_train.to_csv('../data/v%s_%d/left_temp_train.csv' % (vehicle, record), index=False)

    temp_test = tvdf[tvdf.index.isin(test_index)]
    temp_test.to_csv('../data/v%s_%d/left_temp_test.csv' % (vehicle, record), index=False)

