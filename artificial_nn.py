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
    def values(self):
        """
        return all the node values in the list
        """
        return [node.value for node in self.nodesList]

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
            for j in range(currentLayer.length):
                for k in range(0, previousLayer.length):
                    self.weights[l][(k, j)] = random.random() * 2 - 1
        print(self.weights)
    def feedForward(self, inputValues):
        """
        return the output of the neural network based on a specified input
        """
        for i in range(self.layersList[0].length):
            self.layersList[0].nodeAt(i).value = inputValues[i]

        for l in range(1, len(self.layersList)):
            currentLayer = self.layersList[l]
            previousLayer = self.layersList[l - 1]
            for j in range(currentLayer.length):
                currentNode = currentLayer.nodeAt(j)
                for k in range(previousLayer.length):
                    previousNode = previousLayer.nodeAt(k)
                    currentNode.value += previousNode.value * self.weights[l][(k, j)]
                    previousNode.value = 0
                currentNode.value += currentNode.bias
                currentNode.value = sigmoid(currentNode.value)

        return self.layersList[-1]



neuralNet = Network([Layer(3), Layer(2), Layer(3)])
print(neuralNet.feedForward([3, 2, 4]).values())