import numpy as np
import csv
import random
import copy
import tensorflow as tf
from tensorflow import keras

def sigmoid(x): # every output is between 0 and 1
    return 1/(1+np.exp(-x))

def derivativeSigmoid(y): # for help w/gradient descent
    return y*(1-y)

def accuracy(nn, testData, testTargets):
    totalCounter = 0
    trueCounter = 0

    for n, i in enumerate(testData):
        guess = nn.feedforward(i)
        totalCounter+=1
        if testTargets[n] == 1 and guess >=0.5:
            trueCounter+=1
        elif testTargets[n] == 0 and guess<0.5:
            trueCounter+=1

    return str(trueCounter/totalCounter)

class NeuralNetwork:

    # constructor for a 2-Layer Neural Network
    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.weights_IH = np.random.rand(self.hiddenNodes, self.inputNodes)*2-1
        self.weights_HO = np.random.rand(self.outputNodes, self.hiddenNodes)*2-1
        self.bias_H = np.random.rand(self.hiddenNodes, 1)*2-1
        self.bias_O = np.random.rand(self.outputNodes, 1)*2-1
        self.learning_rate = 0.1

    # with the given weights + bias, continue to "feed-forward" the values and calculate the output
    # this is the neural network's prediction

    def feedforward(self, inputList):
        inputs = np.array(inputList).reshape(len(inputList), 1)
        hidden = np.matmul(self.weights_IH, inputs)
        hidden = hidden + self.bias_H
        hidden = sigmoid(hidden)
        output = np.matmul(self.weights_HO, hidden)
        output = output + self.bias_O
        output = sigmoid(output)

        return output

    def train(self, inputList, targetList):

        inputs = np.array(inputList)

        if (isinstance(inputList, list)):
            inputs = inputs.reshape(len(inputList), 1)
        hidden = np.matmul(self.weights_IH, inputs)
        hidden = hidden + self.bias_H
        hidden = sigmoid(hidden)
        outputs = np.matmul(self.weights_HO, hidden)
        outputs = outputs + self.bias_O
        outputs = sigmoid(outputs)

        # note that we could just change above code to feedforward, but we must add reshape

        targets = np.array(targetList)
        if (isinstance(targetList, list)):
            targets = targets.reshape(len(targetList), 1)

        output_errors = targets - outputs
        gradients = derivativeSigmoid(outputs)
        gradients = np.multiply(gradients, output_errors)
        gradients = gradients * self.learning_rate

        # using the method of gradient descent, finding which weights we need to increase/decrease to better our prediction

        hidden_T = np.transpose(hidden)
        weight_HO_deltas = np.matmul(gradients, hidden_T)

        self.weights_HO = self.weights_HO + weight_HO_deltas
        self.bias_O = self.bias_O+gradients

        # travels back through each layer

        who_t = np.transpose(self.weights_HO)
        hidden_errors = np.matmul(who_t, output_errors)

        hidden_gradient = derivativeSigmoid(hidden)
        hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
        hidden_gradient = hidden_gradient*self.learning_rate

        inputs_T = np.transpose(inputs)
        weights_IH_deltas = np.matmul(hidden_gradient, inputs_T)

        self.weights_IH = self.weights_IH + weights_IH_deltas #adjust weights and bias w/ calculated deltas
        self.bias_H = self.bias_H+hidden_gradient


#for items in data:
    #for n, i in enumerate(items):

#output = nn.feedforward(input)
#print(output)
