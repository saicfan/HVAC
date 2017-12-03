import glob
import numpy as np
import pandas as pd
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from preProcessing import freq_table


label = ['Season', 'Weekday', 'Hour of Day', 'Trip Elapsed Minutes', 'Outside Temp', 'Inside Temp',
         'Left Temp Change', 'Wind Level Change', 'Weather - Type', 'Weather - Temp', 'Weather - Wind Speed',
         'Weather - Humidity', 'Power Mode', 'Heading', 'Fuel', 'Avg. Speed', 'Avg. RPM',
         'Left Temp', 'Wind Level']


def correlation_matrix(v, r):
    f = pd.read_csv('../data/v%s_%d/logs.csv' % (v, r))
    df = f[label]
    df['Season'] = df['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    major_ticks = np.arange(0, len(label), 1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    cmap = cm.get_cmap('jet')
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True, which='major', linestyle='--', color='grey')
    plt.title('Vehicle %s Feature Correlation' % vehicle)
    labels = label
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    plt.xticks(rotation=45)
    #plt.show(block=True)
    plt.savefig('../output/v%s_%d/correlation.jpg' % (v, r))


for i in range(len(freq_table)):
    vehicle = freq_table.index[i]
    record = freq_table.values[i]
    correlation_matrix(vehicle, record)
    #print(plt.isinteractive())