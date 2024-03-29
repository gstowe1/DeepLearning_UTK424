import tensorflow as tf
import numpy as np

class _MODEL:
    def __init__(self):
        pass

    def input(self, name=None):
        input = tf.keras.layers.Input(shape=(12,1), name=name)
        return input

    def conv(self, name=None):
        conv = tf.keras.layers.Conv1D(filters=128, kernel_size=1, name=name, activation="relu")
        return conv

    def max(self, name=None):
        return tf.keras.layers.MaxPool1D(name=name)

    def lstm(self, name=None):
        lstm = tf.keras.layers.LSTM(128, name=name, return_sequences=False, activation='relu')
        return lstm

    def dropout(self, name=None, amount=0.1):
        dropout = tf.keras.layers.Dropout(amount)
        return dropout

    def fc(self, name=None):
        return tf.keras.layers.Dense(1,activation='linear' ,name=name)

    def create_model(self):
        input = self.input("Input Layer")
        conv1 = self.conv("Conv1")(input)
        conv2 = self.conv("Conv2")(conv1)
        maxPool = self.max("MaxPool")(conv2)
        lstm1 = self.lstm("LSTM1")(maxPool)
        dropout = self.dropout("DROPOUT1")(lstm1)
 

        fc = self.fc("FC")(dropout)

        model = tf.keras.models.Model(inputs=input, outputs=fc)

        return model
