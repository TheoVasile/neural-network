import random, math

def sigmoid(z):
    """
    Returns the input number z after performing a sigmoid operation on it
    >>> sigmoid(0)
    0.707
    >>> sigmoid(2)
    0.938
    """
    return math.sqrt(1 / (1 + math.e ** (-z)))

class Node:
    """
    A single node within the network
    """
    def __init__(self, bias = random.random() * 2 - 1):
        self.value = 0
        self.bias = bias

class Layer:
    """
    A single layer within the network that contains its own nodes
    """
    def __init__(self, nodesLength : int):
        self.nodesList = [Node() for node in range(0, nodesLength)]
        self.length = nodesLength
    def nodeAt(self, j):
        return self.nodesList[j]

class Network:
    """
    The neural network itself, containing individual layers that contain individual nodes
    """
    def __init__(self, layersList : list):
        self.layersList = layersList
        self.weights = {}
        for l in range(1, len(layersList)):
            currentLayer = self.layersList[l]
            previousLayer = self.layersList[l-1]
            self.weights[l] = {}
            for j in range(0, currentLayer.length):
                for k in range(0, previousLayer.length):
                    self.weights[l][(k, j)] = random.random() * 2 - 1
        print(self.weights)

Network([Layer(3), Layer(2), Layer(3)])