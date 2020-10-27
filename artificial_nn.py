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

def test(iterations):
    """
    prints neural net output for a set amount of iterations. The neural net
    should take in 2 inputs (1 or 0), and is expected to return 1 if they are the same, or 0 if
    they are different. The neural network should learn this pattern
    """
    net = Network([Layer(2), Layer(3), Layer(1)])
    for x in range(iterations):
        input = [random.randint(0, 1), random.randint(0, 1)]
        output = [int(input[0] == input[1])]
        print(net.feedForward(input).values(), output)
        net.backPropogate(output)

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
        Return all the node values in the list
        """
        return [node.value for node in self.nodesList]
    def resetValues(self):
        """
        Set all the values in the layer to 0
        """
        for node in self.nodesList:
            node.value = 0

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
                currentNode.value += currentNode.bias
                currentNode.value = sigmoid(currentNode.value)

        return self.layersList[-1]
    def backPropogate(self, outputValues):
        derivatives = {}
        derivatives[len(self.layersList) - 1] = {}

        #iterate backwards through layers
        for l in range(len(self.layersList) - 1, 0, -1):
            currentLayer = self.layersList[l]
            nextLayer = self.layersList[l - 1]
            derivatives[l-1] = {}

            #iterate through the nodes in the layer behind the current layer
            for k in range(self.layersList[l - 1].length):
                nextNode = nextLayer.nodeAt(k)
                derivatives[l - 1][k] = 0

                #iterate through the nodes in the current layer
                for j in range(self.layersList[l].length):
                    currentNode = currentLayer.nodeAt(j)

                    if l == len(self.layersList) - 1:
                        derivatives[l][j] = -2 * (outputValues[j] - currentNode.value) * currentNode.value * (1 - currentNode.value)
                        currentNode.bias -= derivatives[l][j]
                    derivatives[l - 1][k] += derivatives[l][j] * self.weights[l][(k, j)] * nextNode.value * (1 - nextNode.value)

                    self.weights[l][(k, j)] -= derivatives[l][j] * nextNode.value

                nextNode.bias -= derivatives[l - 1][k]
        print(derivatives)

test(100)