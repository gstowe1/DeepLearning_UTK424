import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('combined.csv')

class windowGenerator():
    '''
    This class is used to create a window of data for the model to train on.
    It will return an array X and Y where X is the input data and Y is the label data.
    X will be of shape (Total # of windows, input_width)
    Y will be of shape (Total # of windows)
    '''
    def __init__(self, data, input_width=12, label_width=1, shift=1):
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.data = data

        self.X = []
        self.Y = []

        # Go through each column (which is a generator)
        for i, col in enumerate(data.columns):
            if i == 0:
                continue

            # Find the first index where the value is not NaN
            first_index = data[col].first_valid_index()
            # Find the last index where the value is not NaN
            last_index = data[col].last_valid_index()

            # Create a series of the data from the first index to the last index
            series = data[col][first_index:last_index+1]
            # Interpolate the data
            series = series.interpolate(method='linear', limit_direction='both')

            # Create the windows
            for j in range(len(series) - self.total_window_size):
                self.X.append(series[j:j+self.input_width].values)
                self.Y.append(series[j+self.input_width:j+self.total_window_size].values)

window_obj = windowGenerator(df)

# Save X and Y to pickle files
with open('X.pkl', 'wb') as f:
    pickle.dump(window_obj.X, f)
with open('Y.pkl', 'wb') as f:
    pickle.dump(window_obj.Y, f)