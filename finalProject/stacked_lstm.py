import tensorflow as tf

class _MODEL:
    def __init__(self):
        pass

    def input(self, name=None):
        input = tf.keras.layers.Input(shape=(12,1), name=name)
        return input

    def lstm(self,unit=20, name=None, return_seq=False, go_backwards=False):
        lstm = tf.keras.layers.LSTM(unit, name=name, return_sequences=return_seq, go_backwards=go_backwards, activation='relu')
        return lstm

    def fc(self, name=None):
        return tf.keras.layers.Dense(1,activation='linear' ,name=name)
    
 

    def create_model(self):
        input = self.input("Input Layer")
        lstm1 = self.lstm(12, "LSTM1", return_seq=True)(input)
        lstm2 = self.lstm(24, "LSTM2", return_seq=True)(lstm1)
        lstm3 = self.lstm(28, "LSTM3", return_seq=False)(lstm2)
      
        fc = self.fc("FC")(lstm3)
        model = tf.keras.models.Model(inputs=input, outputs=fc)

        return model
