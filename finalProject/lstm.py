import tensorflow as tf

class _MODEL:
    def __init__(self):
        pass

    def input(self, name=None):
        input = tf.keras.layers.Input(shape=(12,1), name=name)
        return input

    def conv(self, name=None):
        conv = tf.keras.layers.Conv1D(filters=5, kernel_size=1, name=name, activation="relu")
        return conv

    def max(self, name=None):
        return tf.keras.layers.MaxPool1D(name=name)

    def lstm(self, name=None, return_seq=False):
        lstm = tf.keras.layers.LSTM(5, name=name, return_sequences=return_seq, activation='relu')
        return lstm

    def fc(self, name=None):
        return tf.keras.layers.Dense(1,activation='linear' ,name=name)
    
    def dropout(self, name=None, amount=0.025):
        dropout = tf.keras.layers.Dropout(amount)
        return dropout

    def create_model(self):
        input = self.input("Input Layer")
      
        lstm1 = self.lstm("LSTM1")(input)
        fc = self.fc("FC")(lstm1)
        model = tf.keras.models.Model(inputs=input, outputs=fc)

        return model
