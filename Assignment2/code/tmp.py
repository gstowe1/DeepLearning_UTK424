import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import generateExample2

#Create a feed forward network
model=Sequential()

# Add convolutional layers, flatten, and fully connected layer
model.add(layers.Conv2D(2,3,input_shape=(7,7,1),activation='sigmoid')) 

# Call weight/data generating function
l1k1,l1k2,l1b1,l1b2,l2k1,l2b,l3,l3b,input, output = generateExample2()

#Set weights to desired values 

#setting weights and bias of first layer.
l1k1=l1k1.reshape(3,3,1,1)
l1k2=l1k2.reshape(3,3,1,1)

w1=np.concatenate((l1k1,l1k2),axis=3)
model.layers[0].set_weights([w1,np.array([l1b1[0],l1b2[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)

#Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
img=np.expand_dims(input,axis=(0,3))


#print needed values.
np.set_printoptions(precision=5)
print('model output before:')
print(model.predict(img))


