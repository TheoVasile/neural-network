import random, math

class Node:
    def __init__(self, bias = random.random() * 2 - 1):
        self.value = 0
        self.bias = bias

class Layer:
    def __init__(self, nodesList):
        self.nodesList = nodesList