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
    def __init__(self, nodesList):
        self.nodesList = nodesList

class Network:
    """
    The neural network itself, containing individual layers that contain individual nodes
    """
    def __init__(self, layersList):
        self.layersList = layersList