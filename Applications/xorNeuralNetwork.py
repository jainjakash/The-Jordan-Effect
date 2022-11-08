from neuralNetwork import NeuralNetwork
import numpy as np
import csv
import random
import copy
import tensorflow as tf
from tensorflow import keras

# Simple Data Set, with 10000 epochs and shuffle
# This implements the exclusive OR operation, AKA XOR

nn = NeuralNetwork(2, 2, 1)

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
print("Neural Network's Prediction for [1,0]: " + str(guess1[0]))
print("Neural Network's Prediction for [0,1]: " + str(guess2[0]))
print("Neural Network's Prediction for [1,1]: " + str(guess3[0]))
print("Neural Network's Prediction for [0,0]: " + str(guess4[0]))