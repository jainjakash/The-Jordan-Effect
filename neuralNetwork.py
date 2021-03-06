import numpy as np
import csv
import random
import copy
import tensorflow as tf
from tensorflow import keras


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def sigmoid(x):
    return 1/(1+np.exp(-x))


def derivativeSigmoid(y):
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

    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.weights_IH = np.random.rand(self.hiddenNodes, self.inputNodes)*2-1
        self.weights_HO = np.random.rand(self.outputNodes, self.hiddenNodes)*2-1
        self.bias_H = np.random.rand(self.hiddenNodes, 1)*2-1
        self.bias_O = np.random.rand(self.outputNodes, 1)*2-1
        self.learning_rate = 0.1


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

        # could just change this to feedforward, but must add reshape

        targets = np.array(targetList)
        if (isinstance(targetList, list)):
            targets = targets.reshape(len(targetList), 1)

        output_errors = targets - outputs
        gradients = derivativeSigmoid(outputs)
        gradients = np.multiply(gradients, output_errors)
        gradients = gradients * self.learning_rate

        hidden_T = np.transpose(hidden)
        weight_HO_deltas = np.matmul(gradients, hidden_T)

        self.weights_HO = self.weights_HO + weight_HO_deltas
        self.bias_O = self.bias_O+gradients

        who_t = np.transpose(self.weights_HO)
        hidden_errors = np.matmul(who_t, output_errors)

        hidden_gradient = derivativeSigmoid(hidden)
        hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
        hidden_gradient = hidden_gradient*self.learning_rate

        inputs_T = np.transpose(inputs)
        weights_IH_deltas = np.matmul(hidden_gradient, inputs_T)

        self.weights_IH = self.weights_IH + weights_IH_deltas
        self.bias_H = self.bias_H+hidden_gradient


#Simple Data Set, with 10000 epochs and shuffle

'''nn = NeuralNetwork(2, 2, 1)

inputs = [[1, 0], [0, 1], [1, 1], [0, 0]]
targets = [1, 1, 0, 0]

for x in range(0, 10000):
    temp = list(zip(inputs, targets))
    random.shuffle(temp)
    inputs, targets = zip(*temp)
    for i in range(len(inputs)):
        nn.train(inputs[i], targets[i])

guess1 = nn.feedforward([1, 0])
guess2 = nn.feedforward([0, 1])
guess3 = nn.feedforward([1, 1])
guess4 = nn.feedforward([0, 0])
print(guess1)
print(guess2)
print(guess3)
print(guess4)'''

normalizedOneData = []
trainData = []
testData = []
normalizedOneTargets = []
trainTargets = []
testTargets = []
NN1 = NeuralNetwork(7, 5, 1)
NN2 = NeuralNetwork(7, 5, 1)

with open('JordanDataNormalizedOne.csv', newline='') as f:
    reader = csv.reader(f)
    normalizedOneData = list(reader)

normalizedOneData.pop(0)

for items in normalizedOneData:
    for n, i in enumerate(items):
        if i =='CHI':
            items[n] = 1
        if i == 'WAS':
            items[n] = 0
for items in normalizedOneData:
    for n, i in enumerate(items):
        if is_integer(i):
            items[n] = int(i)
        elif not is_integer(i):
            #print(items[n])
            items[n] = float(i)
for items in normalizedOneData:
    for n, i in enumerate(items):
        if n==0 or n==5 or n==6 or n==7:
            items[n] = items[n]/100;
trainData = copy.deepcopy(random.sample(normalizedOneData, len(normalizedOneData))) # so it doesn't point to the same reference
for items in normalizedOneData:
    normalizedOneTargets.append(items.pop(3))

print("Normalized One Data:")
print(normalizedOneData)
print("Normalized One Targets: ")
print(normalizedOneTargets)

for x in range(0, 10):
    for i in range(len(normalizedOneData)):
        NN1.train(normalizedOneData[i], normalizedOneTargets[i])

'''
# Age, Team, Home, Win, Game Started, RB(Off+Def), Assists, Points
sampleDataPoint11 = [0.216899384, 1, 1, 1, 0.10, 0.10, 0.30]# won the game
sampleDataPoint12 = [0.2484599589, 1, 0, 1, 0.05, 0.02, 0.16]# lost the game
sampleDataPoint13 = [0.2100, 1, 0, 1, 0.02, 0.02, 0.10]
guess11 = NN1.feedforward(sampleDataPoint11)
guess12 = NN1.feedforward(sampleDataPoint12)
guess13 = NN1.feedforward(sampleDataPoint13)
print(guess11)
print(guess12)
print(guess13)'''



testData = trainData[859:len(trainData)]
trainData = trainData[:858]

for items in trainData:
    trainTargets.append(items.pop(3))
for items in testData:
    testTargets.append(items.pop(3))



for x in range(0, 10):
    for i in range(len(trainData)):
        NN2.train(trainData[i], trainTargets[i])

print("Accuracy: "+accuracy(NN2, testData, testTargets))


'''model = keras.Sequential([
    keras.layers.
])'''


'''
with open('JordanDataEdited.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

data.pop(0)

for items in data:
    for n, i in enumerate(items):
        if i =='CHI':
            items[n] = 1
        if i == 'WAS':
            items[n] = 0
for items in data:
    for n, i in enumerate(items):
        if is_integer(i) == True:
            items[n] = int(i)
        elif is_integer(i) == False:
            #print(items[n])
            items[n] = float(i)

winLoss = []

for items in data:
    winLoss.append(items.pop(6))

print(data)
print(winLoss)

#t = [1,2,3,4,5,6,7,8,9]
#print (t[:6] + t[7:len(t)])
#print (t[6])

MJ_Neural_Network = NeuralNetwork(25, 17, 1)

statsOne = [1, 21, 252, 21.6899384, 1, 1, 16, 1, 40, 5, 16, 0, 0, 6, 7, 1, 5, 6, 7, 2, 4, 5, 2, 16, 12.5]
statsTwo = [2, 21, 253, 21.69267625, 1, 0, -2, 1, 34, 8, 13, 0, 0, 5, 5, 3, 2, 5, 5, 2, 1, 3, 4, 21, 19.4]
statsThree = [0, 0, 0, 0, 1, 0, 1, 1, 4, 1, 2, 4, 5, 6, 6, 0, 7, 7, 4, 2, 0, 2, 1, 3, 2]

print(MJ_Neural_Network.feedforward(statsOne))
print(MJ_Neural_Network.feedforward(statsTwo))
print(MJ_Neural_Network.feedforward(statsThree))

for x in range(0, 10):
    for i in range(len(data)):
        MJ_Neural_Network.train(data[i], winLoss[i])


print(len(data))
print(MJ_Neural_Network.feedforward(statsOne))
print(MJ_Neural_Network.feedforward(statsTwo))
print(MJ_Neural_Network.feedforward(statsThree))'''


#for items in data:
    #for n, i in enumerate(items):

#output = nn.feedforward(input)
#print(output)
