"""
Devs: Gabriel Stowe and Eli Carter
Date: 2/10/23
"""

import numpy as np
import sys
from parameters import generateExample2

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):

        self.activation = activation
        self.input_num = input_num + 1
        self.lr = lr

#If weights are None, generate random values. Weights include the bias, therefore the size of the weights is input_num + 1. 
        if weights is None:
            self.weights = np.random.rand(self.input_num)
        else:
            if type(weights) != np.ndarray:
                weights = np.array(weights)
            
            self.weights = weights


        self.inputs = np.array(np.zeros(self.input_num))
        #Set last element to equal 1. This is for the bias.
        self.inputs[-1] = 1
        self.output = 0
        
     
    #This method returns the activation of the net
    def activate(self,net):
        #if activation function is logistic
        if self.activation == 1:
            return 1 / (1 + np.exp(-net))
        
        return net

    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        #If the input is not inserted as a np.array, convert python list to  np.array
        if type(input) != np.ndarray:
            input = np.array(input)

        #set the self.inputs var to equal the input. Last element of inputs has to remain 1 for the bias. 
        self.inputs[:-1] = input

        #Calculate the net. 
        net = np.dot(self.inputs,self.weights.T)

        self.output = self.activate(net)
        return self.output


    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        #derivative of sigmoid function is f(x)*(1-f(x))
        if self.activation == 1:
            return self.output * (1 - self.output)
        #derivative of linear function is 1
        else:
            return 1
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        #Calculate the partial derivative to update weights later
        self.pd = wtimesdelta * self.activationderivative() * self.inputs

        #Return delta * w
        return wtimesdelta * self.activationderivative() * self.weights
        
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        print("\t\tweights before update: ", self.weights)
        print("\t\tpartial derivative: ", self.pd)
        self.weights = self.weights - self.pd * self.lr
        print("\t\tweights after update: ", self.weights)
        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):

        self.numOfNeurons = numOfNeurons
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self.weights = weights

        self.neurons = []

        for i in range(self.numOfNeurons):
            if self.weights is None:
                self.neurons.append(Neuron(activation=self.activation,input_num=self.input_num,lr=self.lr,weights=None))
            else:
                self.neurons.append(Neuron(activation=self.activation,input_num=self.input_num,lr=self.lr,weights=weights[i]))

    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        #Stores the output
        output = np.array(np.zeros(self.numOfNeurons))

        #Check if input is a numpy array. If not, make it a numpy array. 
        if type(input) != np.ndarray:
            input = np.array(input)

        #Loop through the neurons and calculate its output. 
        for i,neuron in enumerate(self.neurons):
            output[i] = neuron.calculate(input)

        return output                    
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        # deltas = np.array(np.zeros(len(self.neurons)))
        deltas = []
        for i, neuron in enumerate(self.neurons):
            deltas.append(neuron.calcpartialderivative(wtimesdelta[i]))
            neuron.updateweight()
        
        return np.sum(deltas,axis=0)

# A Convolutional Layer Class
class Convolutional:
        #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self, num_kernel=1, kernel_size=3, activation=0, input_nums=[3,5,5], lr=0.01, weights=None):
        self.num_kernel = num_kernel
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.activation = activation
        self.lr = lr
        self.weights = weights
        self.neurons = []
        self.input_nums = input_nums

        n_array_w = input_nums[1] - kernel_size + 1
        n_array_h = input_nums[2] - kernel_size + 1
        n_array_d = num_kernel

        if self.weights is None:
            print("Initializing weights randomly...")
            n_weights = np.random.rand(num_kernel, (self.kernel_size * self.kernel_size)*input_nums[0] + 1)
        else:
            n_weights = self.weights


        for kernel in range(num_kernel):
            hs = []
            for i in range(n_array_w):
                ds = []
                for j in range(n_array_h):
                    print(f'\t\tCreating Neuron with weights: {n_weights[kernel]}')
                    ds.append(Neuron(activation=self.activation, input_num=self.kernel_size**2 * input_nums[0], lr=self.lr, weights=n_weights[kernel]))
                hs.append(ds)
            self.neurons.append(hs)


    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        #Stores the output
        output = np.zeros(np.shape(self.neurons))

        for kernel in range(len(self.neurons)):
            for h in range(len(self.neurons[kernel])):
                for w in range(len(self.neurons[kernel][h])):
                    # if the input has multiple channels loop through them and add their outputs
                    if len(np.shape(input)) == 3:
                        n_input = []
                        # concatenate multiple channels to the input
                        for channel in range(0, len(input)):
                            n_input = np.concatenate((n_input, input[channel][h:h+self.kernel_size,w:w+self.kernel_size].flatten()))
                        output[kernel][h][w] = self.neurons[kernel][h][w].calculate(n_input)
                    else:
                        output[kernel][h][w] = self.neurons[kernel][h][w].calculate(input[h:h+self.kernel_size,w:w+self.kernel_size].flatten())

        return output                 
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        # print("\twtimesdelta input shape: ", np.shape(wtimesdelta))
        wtd_matrix = np.zeros(self.input_nums)
        # print('\t\twtd_matrix output shape: ', np.shape(wtd_matrix))
        total_pd = np.zeros((self.kernel_size * self.kernel_size)*self.input_nums[0] + 1)

        for kernel in range(len(self.neurons)):
            for h in range(len(self.neurons[kernel])):
                for w in range(len(self.neurons[kernel][h])):

                    # Calculate the partial derivative
                    partial_deriv = self.neurons[kernel][h][w].calcpartialderivative(wtimesdelta[kernel][h][w])[:-1].reshape(self.input_nums[0], self.kernel_size,self.kernel_size)
                    
                    # Place the partial derivative in the correct spot in the matrix
                    for channel in range(self.input_nums[0]):
                        wtd_matrix[channel][h:h+self.kernel_size,w:w+self.kernel_size] += partial_deriv[channel]

                    # Add the partial derivative to the total
                    total_pd += self.neurons[kernel][h][w].pd

        for kernel in range(len(self.neurons)):
            for h in range(len(self.neurons[kernel])):
                for w in range(len(self.neurons[kernel][h])):
                    self.neurons[kernel][h][w].pd = total_pd[kernel]
                    print(f'At Neuron {h} {w} in kernel {kernel}')
                    self.neurons[kernel][h][w].updateweight()

        return wtd_matrix

# A Flatten Layer Class
class Flatten:
    def __init__(self, input_nums=[3,5,5]):
        self.input_nums = input_nums
        self.numOfNeurons = input_nums[0] * input_nums[1] * input_nums[2]

    def calculate(self, input):
        return input.flatten()
                   
    def calcwdeltas(self, wtimesdelta):
        return wtimesdelta[:-1].reshape(self.input_nums)

#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, input_num, loss, lr):
        # input size, loss function and learning rate are only base args, the rest are done through addLayer()
        self.input_num = input_num
        self.loss = loss
        self.lr = lr
        self.layers = []

    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        print("Calculating")
        for i, layer in enumerate(self.layers):
            output = layer.calculate(input)
            input = output
        return output
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        if type(yp)  != np.ndarray:
            yp = np.array(yp)
        if type(y) != np.ndarray:
            y = np.array(y)
        #Square Loss
        if self.loss == 0:
            return (y - yp)**2
        # Cross Entropy
        else:
            return -y*np.log(yp) - (1-y)*np.log(1-yp)
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        #Square Loss Derivative
        if self.loss == 0:
            return -2*(y- yp)
        # Cross Entropy Derivative
        else:
            return -y/yp + (1-y)/(1-yp)
        
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        #Check if x and y are numpy arrays. If not, convert them.  
        if type(x) != np.ndarray:
            x = np.array(x)
        if type(y) != np.ndarray:
            y = np.array(y)

        #Calculate Prdicted Ouput
        yp = self.calculate(x)

        delta = self.lossderiv(yp,y)
        i = len(self.layers)
        print("Backpropagating")
        for r_layer in reversed(self.layers):
            print("Layer: ", i, " Type: ", r_layer.__class__.__name__)
            i -= 1
            delta = r_layer.calcwdeltas(delta)

        return self.calculateloss(yp,y)

    

    def addLayer(self, layer_type, numOfNeurons=0, activation=0, num_kernel=1, kernel_size=3, weights=None):
        if layer_type == 'conv':
            print('Adding Convolutional Layer')
            #Convolutional layer args: (self, num_kernel=1, kernel_size=3, activation=0, input_nums=[5,5,3], lr=0.01, weights=None)
            # If we are in a convolutional layer, then there can not be a fully connected layer before it
            if len(self.layers) == 0:
                input_num_for_layer = self.input_num
            else:
                #Figure out input num from current last layer
                w = len(self.layers[-1].neurons[0])
                h = len(self.layers[-1].neurons[0][0])
                d = self.layers[-1].num_kernel
                input_num_for_layer = [d,w,h]
            print("\tWith input num: ", input_num_for_layer)
            print("\tWith kernel size: ", kernel_size)
            print("\tWith num kernel: ", num_kernel)
            print("\tWith weights: ", weights)

            layer = Convolutional(num_kernel, kernel_size, activation, input_num_for_layer, self.lr, weights)

        elif layer_type == 'flatten':
            # Figure out input num from current last layer
            if len(self.layers) == 0:
                input_num_for_layer = self.input_num
            else:
                w = len(self.layers[-1].neurons[0])
                h = len(self.layers[-1].neurons[0][0])
                d = self.layers[-1].num_kernel
                input_num_for_layer = [d,h,w]
            layer = Flatten(input_num_for_layer)

        else:
            #FullyConnected layer args: (self,numOfNeurons, activation, input_num, lr, weights=None)
            #Figure out input num from current last layer
            if len(self.layers) == 0:
                input_num_for_layer = self.input_num
            else:
                input_num_for_layer = self.layers[-1].numOfNeurons

            layer = FullyConnected(numOfNeurons, activation, input_num_for_layer, self.lr, weights)

        self.layers.append(layer)

if __name__=="__main__":
    if (len(sys.argv)<2):
        w=[[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]]
        
        # Old network contrsuction N = NeuralNetwork( numOfLayers=2 , numOfNeurons=[2,2], input_num=[2,2], activation=[1,1],loss=0, lr=0.5,weights=w)
        N = NeuralNetwork(input_num=2, loss=0, lr=0.5)
        N.addLayer('fc', numOfNeurons=2, activation=1, weights=w[0])
        N.addLayer('fc', numOfNeurons=2, activation=1, weights=w[1])

        print("Inital Output: ",N.calculate([0.05,0.1]))
        print("Expected Output: [0.01,0.99]")
        
        for i in range(1000):
            N.train([0.05,0.1],[0.01,0.99])
        print("After 1000 Epochs Output: ",N.calculate([0.05,0.1]))

    elif (sys.argv[2]=='example'):

        w=[[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]]
        x = np.array([0.05,0.1])
        y = np.array([0.01,0.99])
       
        N = NeuralNetwork( numOfLayers=2 , numOfNeurons=[2,2], input_num=[2,2], activation=[1,1],loss=0, lr=float(sys.argv[1]),weights=w)
        N.train(x,y)

        print("After training Output: ",np.around(N.calculate(x),3))
        for i in range(N.numOfLayers):
            for j in range(N.layers[i].numOfNeurons):
                print(f"    Layer {i+1} Neuron {j+1} weights: {[round(w,2) for w in N.layers[i].neurons[j].weights]}")

    elif(sys.argv[2]=='and'):
        
        input_nums = [2]
        num_neurons = [1]
        num_layers = 1
        activations = [1]

        N = NeuralNetwork(input_num=2, loss=0, lr=float(sys.argv[1]))

        for i in range(1100):
            N.train([0,0],np.array([0]))
            N.train([1,0],np.array([0]))
            N.train([1,1],np.array([1]))
            N.train([0,1],np.array([0]))
            if i%100 == 0:
                print(f"Epoch {i}:")
                print(f"    0 and 0: {round(N.calculate([0,0])[0], 3)}")
                print(f"    0 and 1: {round(N.calculate([0,1])[0], 3)}")
                print(f"    1 and 0: {round(N.calculate([1,0])[0], 3)}")
                print(f"    1 and 1: {round(N.calculate([1,1])[0], 3)}")

    elif(sys.argv[2]=='xor'):
        
        input_nums = [2,2,3]
        num_neurons = [2,3,1]
        num_layers = 3
        activations = [1,1,1,1]

        N = NeuralNetwork(input_num=2, loss=0, lr=float(sys.argv[1]))
        N.addLayer('fc', numOfNeurons=2, activation=1)
        N.addLayer('fc', numOfNeurons=3, activation=1)
        N.addLayer('fc', numOfNeurons=1, activation=1)

        for i in range(5100):
            N.train(np.array([0,0]),np.array([0]))
            N.train(np.array([0,1]),np.array([1]))
            N.train(np.array([1,0]),np.array([1]))
            N.train(np.array([1,1]),np.array([0]))
            if i%100 == 0:
                print(f"Epoch {i}:")
                print(f"    0 and 0: {np.around(N.calculate(np.array([0,0])),3)[0]}")
                print(f"    0 and 1: {np.around(N.calculate(np.array([0,1])),3)[0]}")
                print(f"    1 and 0: {np.around(N.calculate(np.array([1,0])),3)[0]}")
                print(f"    1 and 1: {np.around(N.calculate(np.array([1,1])),3)[0]}")

    elif(sys.argv[2]=='example1'):
        # Keras model
        pass

    elif(sys.argv[2]=='example2'):
        # Keras model
        # model.add(layers.Conv2D(2,3,input_shape=(7,7,1),activation='sigmoid')) 
        # model.add(layers.Conv2D(1,3,activation='sigmoid'))
        # model.add(layers.Flatten())
        # model.add(layers.Dense(1,activation='sigmoid'))

        # Load example weights
        l1k1,l1k2,l1b1,l1b2,l2k1,l2b,l3,l3b,input, output = generateExample2()

        # Set weights and biases for layer 1
        # Flatten for neuron
        l1k1=l1k1.reshape(3,3,1,1)
        l1k2=l1k2.reshape(3,3,1,1)

        # Add bias
        l1k1 = np.append(l1k1,l1b1)
        l1k2 = np.append(l1k2,l1b2)

        # make final array of weights
        w1 = np.array([l1k1,l1k2])
        print(f'w1: {w1}')

        # Set weights and biases for layer 2
        # Flatten for neuron
        l2k1 = l2k1.reshape(3,3,2,1)

        # Add bias
        l2k1 = np.append(l2k1,l2b)

        # make final array of weights
        w2 = np.array([l2k1])

        # Set weights and biases for layer 3

        l3k1 = np.append(l3,l3b)
        w3 = np.array([l3k1])

        print(f'w1: {w1}')
        print(f'w2: {w2}')
        print(f'w3: {w3}')
        print(f'input: {input}')
        print(f'output: {output}')


        N = NeuralNetwork(input_num=[1,7,7], loss=0, lr=float(sys.argv[1]))
        N.addLayer('conv', num_kernel=2, kernel_size=3, activation=1, weights = w1)
        N.addLayer('conv', num_kernel=1, kernel_size=3, activation=1, weights = w2)
        N.addLayer('flatten')
        N.addLayer('fc', numOfNeurons=1, activation=1, weights = w3)

        print(f"Before training Output: {N.calculate(input)}")
        for i in range(1000):
            print(N.train(input,output))
        print(f"After training Output: {N.calculate(input)}")