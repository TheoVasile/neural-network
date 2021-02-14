from typing import List, Dict, Tuple
import random
import math
import doctest
import sympy

VARIABILITY = 2

def sigmoid(z):
    """
    Returns the input number z after performing a sigmoid operation on it
    >>> sigmoid(0)
    0.7071067811865476
    >>> sigmoid(2)
    0.9385078997951388
    """
    return math.sqrt(1 / (1 + math.e ** (-z)))


class Node:
    """
    A single node within the network

    === Attributes ==
    value: the value a node holds at any given moment
    bias: the bias the node adds to its value
    """
    # attribute types
    value: float
    bias: float

    def __init__(self):
        """
        Initialize node
        """
        self.value = 0
        self.bias = random.random() * VARIABILITY - VARIABILITY / 2


class Layer:
    """
    A single layer within the network that contains its own nodes

    === Attributes ===
    nodesList: a list containing all the nodes in the layer
    length: the amount of nodes in the layer
    """
    # attribute types
    nodes_list: List[Node]
    length: int

    def __init__(self, length: int):
        """
        Initialize layer

        precondition: the length of <nodes_list> must be greater than 0
        """
        self.nodes_list = [Node() for node in range(0, length)]
        self.length = length

    def node_at(self, j: int) -> Node:
        """
        Return the node at position {j} in the layer
        """
        return self.nodes_list[j]

    def values(self) -> List[float]:
        """
        Return all the node values in the list
        """
        return [node.value for node in self.nodes_list]


class Network:
    """
    The neural network itself, containing individual layers that contain
    individual nodes

    === Attributes ===
    learning_rate: how much an individual change will affect the network
    layers_list: a list containing all the layers in the network
    weights: all the weights between nodes in the network
    """
    # attribute types
    learning_rate: float
    layers_list: List[Layer]
    weights: Dict[int, Dict[Tuple[int, int], float]]

    def __init__(self, layers_list: list):
        """
        initialize the network

        precondition: the length of <layers_list> must be greater than
        or equal to 2
        """
        self.learning_rate = 0.002
        self.layers_list = layers_list
        self.weights = {}
        for layer in range(0, len(layers_list)):
            current_layer = self.layers_list[layer]
            previous_layer = self.layers_list[layer - 1]
            self.weights[layer] = {}
            for j in range(current_layer.length):
                for k in range(0, previous_layer.length):
                    self.weights[layer][(
                        k, j)] = random.random() * VARIABILITY - VARIABILITY / 2

    def feed_forward(self, input_values: List[float]) -> Layer:
        """
        return the output of the neural network based on a specified input

        precondition: the length of <input_values> must be greater than 0
        """
        for i in range(self.layers_list[0].length):
            node = self.layers_list[0].node_at(i)
            node.value = input_values[i]

        for layer in range(1, len(self.layers_list)):
            current_layer = self.layers_list[layer]
            previous_layer = self.layers_list[layer - 1]
            for j in range(current_layer.length):
                current_node = current_layer.node_at(j)
                current_node.value = 0
                for k in range(previous_layer.length):
                    previous_node = previous_layer.node_at(k)
                    current_node.value += previous_node.value * \
                                          self.weights[layer][(k, j)]
                current_node.value += current_node.bias
                current_node.value = sigmoid(current_node.value)

        return self.layers_list[-1]

    def back_propagate(self, output_values: List[float]) -> None:
        """
        Mutate the weights and biases of the network based on a given output

        Precondition: the length of <output_values> must be greater than 0
        """
        derivatives = {len(self.layers_list) - 1: {}}

        # iterate backwards through layers
        for layer in range(len(self.layers_list) - 1, 0, -1):
            current_layer = self.layers_list[layer]
            next_layer = self.layers_list[layer - 1]
            derivatives[layer - 1] = {}

            # iterate through the nodes in the layer behind the current layer
            for k in range(self.layers_list[layer - 1].length):
                next_node = next_layer.node_at(k)
                derivatives[layer - 1][k] = 0

                # iterate through the nodes in the current layer
                for j in range(self.layers_list[layer].length):
                    current_node = current_layer.node_at(j)

                    if layer == len(self.layers_list) - 1:
                        derivatives[layer][j] = -2 * (output_values[j] -
                                                      current_node.value) * \
                                                current_node.value * (
                                                        1 - current_node.value)
                        current_node.bias -= self.learning_rate * \
                                             derivatives[layer][
                                                 j]
                    derivatives[layer - 1][k] += derivatives[layer][j] * \
                                                 self.weights[layer][
                                                     (k,
                                                      j)] * next_node.value * (
                                                         1 - next_node.value)

                    self.weights[layer][(k, j)] -= self.learning_rate * \
                                                   derivatives[layer][
                                                       j] * next_node.value

                next_node.bias -= self.learning_rate * derivatives[layer - 1][k]


class Matrix:
    """
    An implementation of a matrix using the built in data types

    === Attributes ===
    rows: a list of rows in the matrix
    columns: a list of columns in the matrix
    """
    # attribute types
    rows: List[List[float]]
    columns: List[List[float]]

    def __init__(self, data: List[List[float]]):
        self.rows = data
        self.columns = [[row[column] for row in data] for column in range(len(data[0]))]

    def insert(self, row: List[float]) -> None:
        """
        Insert a new <row> into the matrix

        >>> matrix = Matrix([[1, 2], [5, 1]])
        >>> matrix.insert([3, 4])
        >>> matrix.columns
        [[1, 5, 3], [2, 1, 4]]
        >>> matrix.rows
        [[1, 2], [5, 1], [3, 4]]
        """
        self.rows.append(row)
        for column in range(len(row)):
            self.columns[column].append(row[column])

    def remove_row(self, row: int) -> None:
        """
        Remove a <row> from the matrix

        >>> matrix = Matrix([[1, 2], [5, 1], [3, 4]])
        >>> matrix.remove_row(1)
        >>> matrix.columns
        [[1, 3], [2, 4]]
        >>> matrix.rows
        [[1, 2], [3, 4]]
        """
        for column in range(len(self.columns)):
            self.columns[column].pop(row)
        self.rows.pop(row)
    def remove_column(self, column: int) -> None:
        """
        Remove a <column> from the matrix

        >>> matrix = Matrix([[1, 2], [5, 1], [3, 4]])
        >>> matrix.remove_column(1)
        >>> matrix.columns
        [[1, 5, 3]]
        >>> matrix.rows
        [[1], [5], [3]]
        """
        for row in range(len(self.rows)):
            self.rows[row].pop(column)
        self.columns.pop(column)

    def mean(self, column: int) -> float:
        """
        Return the mean of a <column> in the matrix

        >>> matrix = Matrix([[1, 2], [3, 4]])
        >>> matrix.mean(0)
        2.0
        """
        return sum(self.columns[column]) / len(self.columns[column])

    def get(self, column: int, row: int) -> float:
        """
        Return the value at a certain <row> and <column> in the matrix

        >>> matrix = Matrix([[1, 2], [3, 4]])
        >>> matrix.get(0, 1)
        3
        """
        return self.rows[row][column]

    def covariance(self):
        """
        Returns the covariance matrix of a data set

        >>> matrix = Matrix([[1, 2, 3], [2, 6, 1]])
        >>> matrix = matrix.covariance()
        >>> matrix.rows
        [[0.25, 1.0, -0.5], [1.0, 4.0, -2.0], [-0.5, -2.0, 1.0]]
        """
        covariance_matrix = Matrix([[0] * len(self.rows[0])])
        for y in range(len(self.rows[0])):
            row = []
            for x in range(len(self.rows[0])):
                covariance_ = 0
                for i in range(len(self.columns[0])):
                    covariance_ += (self.get(x, i) - self.mean(x)) * (self.get(y, i) - self.mean(y))
                covariance_ /= len(self.columns) - 1
                row.append(covariance_)
            covariance_matrix.insert(row)
        covariance_matrix.remove_row(0)

        return covariance_matrix


class PCA:
    """
    Principal Component Analysis reduces the dimensionality of large pieces of
    data.
    Most of the mathematics involve matrix calculations, but for the sake of
    better understanding the underlying maths, no built in matrixes will be used

    === Attributes ===
    data: the input data, taking the form of a series of lists, with each value
    being on its own axis
    principle_components: the amount of dimensions to reduce the input data to
    """
    # attribute types
    data: Matrix
    principle_components: int

    def __init__(self, data: List[List[float]], principle_components: int):
        """
        Initialize attributes
        """
        self.data = Matrix(data)
        self.principle_components = principle_components

if __name__ == "__main__":
    doctest.testmod()
