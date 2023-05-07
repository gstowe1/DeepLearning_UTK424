import tensorflow as tf

class CNN_LSTM:
    def __init__(self):
        pass

    def input(self, name=None):
        input = tf.keras.layers.Input(shape=(12,1), name=name)
        # reshaped_input = tf.reshape(input, [-1, 12, 1])  # reshape to 3D array
        return input

    def conv(self, name=None):
        conv = tf.keras.layers.Conv1D(filters=128, kernel_size=1, name=name, activation="relu")
        return conv

    def max(self, name=None):
        return tf.keras.layers.MaxPool1D(name=name)

    def lstm(self, name=None):
        lstm = tf.keras.layers.LSTM(128, name=name, return_sequences=True, activation='relu')
        return lstm

    def fc(self, name=None):
        return tf.keras.layers.Dense(1,activation='sigmoid' ,name=name)

    def create_model(self):
        input = self.input("Input Layer")
        # conv1 = self.conv("Conv1")(input)
        conv2 = self.conv("Conv2")(input)
        # maxPool = self.max("MaxPool")(conv2)

        lstm1 = self.lstm("LSTM1")(conv2)
        # lstm1 = self.lstm("LSTM1")(maxPool)
        lstm2 = self.lstm("LSTM2")(lstm1)
        lstm3 = self.lstm("LSTM3")(lstm2)

        flat = tf.keras.layers.Flatten()(lstm3)
        fc = self.fc("FC")(flat)
        # fc = self.fc("FC")(lstm3)
        model = tf.keras.models.Model(inputs=input, outputs=fc)

        return model
