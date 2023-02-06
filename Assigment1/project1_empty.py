import numpy as np
import sys
import matplotlib.pyplot as plt # Only imported for plotting

"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        if weights is None:
            self.weights = np.random.rand(self.input_num+1)
        else:
            self.weights = weights

        # Storing info for backpropagation
        self.input = np.array(np.zeros(self.input_num+1))
        self.output = 0
        self.pd = np.array(np.zeros(self.input_num+1))
        
    #This method returns the activation of the net
    def activate(self,net):
        if self.activation == 1:
            return 1 / (1 + np.exp(-net))
        else:
            return net

    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        net = 0
        self.input = input
        for i in range(len(self.weights)-1):
            net += self.input[i] * self.weights[i] 
        net += self.weights[-1]

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
        # Note: at this point wtimesdelta is a single number

        #calculate the partial derivative for each weight
        for i in range(len(self.weights)-1):
            self.pd[i] = wtimesdelta * self.activationderivative() * self.input[i]
        #calculate the partial derivative for the bias
        self.pd[-1] = wtimesdelta * self.activationderivative() * 1

        #return the array of delta*w to be used in the previous layer
        return wtimesdelta * self.activationderivative() * self.weights[:-1]

    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        self.weights = self.weights - self.lr * self.pd

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.input_num = input_num
        self.layer = []
        self.list_of_deltaw = []
        
        if weights is not None:
            for i in range(numOfNeurons):
                self.layer.append(Neuron(activation,input_num,lr,weights[i]))
        else:
            for i in range(numOfNeurons):
                self.layer.append(Neuron(activation,input_num,lr,None))

        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        outputs = []
        for i in range(len(self.layer)):
            outputs.append(self.layer[i].calculate(input))
        return outputs        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        for i in range(len(self.layer)):
            self.list_of_deltaw.append(self.layer[i].calcpartialderivative(wtimesdelta[i]))
            self.layer[i].updateweight()
        return np.sum(self.list_of_deltaw, axis=0)

        

           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, input_num, activation, loss, lr, weights=None):
        
        #Create a list of layers. 
        self.layers = []
        self.loss = loss
        self.numOfNeurons = numOfNeurons
        self.numOfLayers = numOfLayers
        self.weights = weights
        #if weights is non  e, create a 3d array of random weights. 
        for i in range(0,numOfLayers):
            if weights is not None:
                self.layers.append(FullyConnected(numOfNeurons[i],activation,input_num[i],lr,weights[i]))
            else:
                self.layers.append(FullyConnected(numOfNeurons[i],activation,input_num[i],lr,None))
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):       
        outputs = []
        for i in range(len(self.layers)):
            outputs = self.layers[i].calculate(input)
            input = outputs
        return outputs        
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y): #expectes np arrays
        # Sum of Square Error
        if self.loss == 0:
            return 0.5 * (y - yp)**2
        # Cross Entropy
        else:
            return -y*np.log(yp) - (1-y)*np.log(1-yp)
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0:
            deriv = -(y- yp)
            return deriv
        else:
            deriv = -y/yp + (1-y)/(1-yp)
            return deriv
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y): #expects np arrays
        #forward propogation
        yp = np.array(self.calculate(x))
        #calculate the loss
        loss = self.calculateloss(yp,y)

        #back propogation
        deltas = self.lossderiv(yp,y)
        for i in range(len(self.layers)-1,-1,-1):
            deltas = self.layers[i].calcwdeltas(deltas)

        return loss


if __name__=="__main__":
    if (len(sys.argv)<2):
        input_nums = [2,2]
        num_neurons = [2,2]
        num_layers = 2

        w=[np.array([[.15,.2,.35],[.25,.3,.35]]),np.array([[.4,.45,.6],[.5,.55,.6]])]
        class_inputs = np.array([.05,.10])
        desire = np.array([0.01,0.99])

        # print('a good place to test different parts of your code')

        # print("Testing Neuron Class")
        # h1 = Neuron(1,2,0.5,w[0][0]).calculate(class_inputs)
        # h2 = Neuron(1,2,0.5,w[0][1]).calculate(class_inputs)
        # print(f"h1: {h1}")
        # print(f"h2: {h2}")
        # print(f"o1: {Neuron(1,2,0.5,w[1][0]).calculate([h1,h2])}")
        # print(f"o2: {Neuron(1,2,0.5,w[1][1]).calculate([h1,h2])}")

        # #Testing FullyConnected Class
        # print('\nTesting Layer Class')
      
        # inputs = class_inputs
        # for i in range(len(w)):
        #     firstLayer = FullyConnected(2,1,2,0,w[i]).calculate(inputs)
        #     print(f'Layer {i+1}: {firstLayer}')
        #     inputs = firstLayer

        #Testing NeuralNetwork Class
        print('\nTesting NeuralNetwork Class')

        losses = []
        for lr in np.linspace(0.1, 0.9, 9):
            N = NeuralNetwork(num_layers,num_neurons,input_nums,1,0,lr,w)
            losses_for_lr = []
            for i in range(0,1000,1):
                N.train(class_inputs,desire)  
                losses_for_lr.append(sum(N.calculateloss(N.calculate(class_inputs),desire)))
                print(f"Step: {i} yp: {N.calculate(class_inputs)} Total loss: {sum(N.calculateloss(N.calculate(class_inputs),desire))}")
            losses.append(losses_for_lr)

        # Plot the losses for each learning rate as seperate lines on the same plot
        for i in range(len(losses)):
            plt.plot(losses[i], label=f'Learning Rate: {np.linspace(0.1, 0.9, 9)[i]:.1f}')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.show()


    elif (sys.argv[2]=='example'):
        print('run example from class')

        input_nums = [2,2]
        num_neurons = [2,2]
        num_layers = 2

        w=[np.array([[.15,.2,.35],[.25,.3,.35]]),np.array([[.4,.45,.6],[.5,.55,.6]])]
        class_inputs = [.05,.10]
        desire = [0.01,0.99]

        N = NeuralNetwork(num_layers,num_neurons,input_nums,1,0,float(sys.argv[1]),w)
        N.train(class_inputs,desire)

        # Print out according to rubric
        print(f"Single step")
        print(f"    Output: {N.calculate(class_inputs)}")
        for i in range(len(N.layers)):
            for j in range(len(N.layers[i].layer)):
                print(f"    Layer {i+1} Neuron {j+1} weights: {N.layers[i].layer[j].weights}")

        for i in range(1,1001,1):
            N.train(class_inputs,desire)    
            print(f"{i} steps {N.calculate(class_inputs)}")

        
    elif(sys.argv[2]=='and'):
        print('learn and')
        input_nums = [2]
        num_neurons = [1]
        num_layers = 1

        N = NeuralNetwork(num_layers,num_neurons,input_nums,1,0,float(sys.argv[1]), None)

        for i in range(1000):
            N.train([0,0],[0])
            N.train([1,0],[0])
            N.train([1,1],[1])
            N.train([0,1],[0])
            if i%100 == 0:
                print(f"Epoch {i}:")
                print(f"    0 and 0: {N.calculate([0,0])}")
                print(f"    0 and 1: {N.calculate([0,1])}")
                print(f"    1 and 0: {N.calculate([1,0])}")
                print(f"    1 and 1: {N.calculate([1,1])}")


        print('After training:')
        print(f"    0 and 0: {round(N.calculate([0,0])[0])}")
        print(f"    0 and 1: {round(N.calculate([0,1])[0])}")
        print(f"    1 and 0: {round(N.calculate([1,0])[0])}")
        print(f"    1 and 1: {round(N.calculate([1,1])[0])}")

        for i in range(len(N.layers)):
            for j in range(len(N.layers[i].layer)):
                print(f"    Layer {i+1} Neuron {j+1} weights: {N.layers[i].layer[j].weights}")

        
    elif(sys.argv[2]=='xor'):
        print('learn xor')
        input_nums = [2,2,3,3]
        num_neurons = [2,3,3,1]
        num_layers = 4

        print("SINGLE PRECEPTRON")
        N = NeuralNetwork(1,[1],2,1,0,float(sys.argv[1]), None)

        for i in range(1000):
            N.train([0,0],[0])
            N.train([0,1],[1])
            N.train([1,0],[1])
            N.train([1,1],[0])
            if i%100 == 0:
                print(f"Epoch {i}:")
                print(f"    0 and 0: {N.calculate([0,0])}")
                print(f"    0 and 1: {N.calculate([0,1])}")
                print(f"    1 and 0: {N.calculate([1,0])}")
                print(f"    1 and 1: {N.calculate([1,1])}")

        print("WITH HIDDEN LAYER")
        N = NeuralNetwork(num_layers, num_neurons,input_nums,1,0,float(sys.argv[1]), None)

        for i in range(2000):
            N.train(np.array([0,0]),np.array([0]))
            N.train(np.array([0,1]),np.array([1]))
            N.train(np.array([1,0]),np.array([1]))
            N.train(np.array([1,1]),np.array([0]))
            if i%100 == 0:
                print(f"Epoch {i}:")
                print(f"    0 and 0: {N.calculate([0,0])}")
                print(f"    0 and 1: {N.calculate([0,1])}")
                print(f"    1 and 0: {N.calculate([1,0])}")
                print(f"    1 and 1: {N.calculate([1,1])}")
        