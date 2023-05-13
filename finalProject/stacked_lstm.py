import tensorflow as tf

class _MODEL:
    def __init__(self):
        pass

    def input(self, name=None):
        input = tf.keras.layers.Input(shape=(12,1), name=name)
        return input

    def conv(self, name=None, input_shape=(12,)):
        conv = tf.keras.layers.Conv1D(filters=200,input_shape=input_shape ,kernel_size=3, name=name, activation="relu")
        return conv

    def max(self, name=None):
        return tf.keras.layers.MaxPool1D(name=name)
    
    def adverage(self,name=None):
        return tf.keras.layers.AvgPool1D(name=name)
    

    def lstm(self,unit=20, name=None, return_seq=False, go_backwards=False):
        lstm = tf.keras.layers.LSTM(unit, name=name, return_sequences=return_seq, go_backwards=go_backwards, activation='relu')
        return lstm

    def fc(self, name=None):
        return tf.keras.layers.Dense(1,activation='linear' ,name=name)
    
    def dropout(self, name=None, amount=0.025):
        dropout = tf.keras.layers.Dropout(amount, name=name)
        return dropout

    def create_model(self):
        input = self.input("Input Layer")
        lstm1 = self.lstm(12, "LSTM1", return_seq=True)(input)
        lstm2 = self.lstm(24, "LSTM2", return_seq=True)(lstm1)
        lstm3 = self.lstm(28, "LSTM3", return_seq=False)(lstm2)
      
        # conv1 = self.conv("CONV")(lstm3)


        fc = self.fc("FC")(lstm3)
        model = tf.keras.models.Model(inputs=input, outputs=fc)

        return model
