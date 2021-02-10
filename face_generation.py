import pygame as pg
from artificial_nn import Network, Layer
import random
from typing import List
from PIL import Image
import pickle

W = 200
H = 200
screen = pg.display.set_mode((W, H))
pg.display.set_caption("face generation")
clock = pg.time.Clock()
fps = 60


def display(pixels: List[float], side_length: int, rgb: bool = True) -> None:
    """
    Draws pixels to the screen based on a a list <pixels> which contains
    information about the color of a pixel on screen so that it creates
    a square image of a certain <side_length>, with or without <rgb> color.
    """
    n = int(min(W, H) / side_length)
    color_channels = (2 * rgb + 1)
    for i in range(0, len(pixels), color_channels):
        y = int(i / (color_channels * 32))
        x = (i - (color_channels * 32) * y) / color_channels
        if rgb:
            color = (int(pixels[i] * 255), int(pixels[i+1] * 255),
                     int(pixels[i+2] * 255))
        else:
            color = [int(pixels[i] * 255)] * 3
        pg.draw.rect(screen, color, (int(x * n), int(y * n), n, n), 0)


def learn() -> List[float]:
    """
    Pass in an image as an output, run it through the network and reduce the
    dimensions, before increasing it back to the original size. Backpropagate
    using the input image as reference to adjust the network weights and biases
    """
    choice = random.randint(1, 4601)
    img = Image.open(f"face_training_set/{choice}.png")
    img = list(img.getdata())
    input_ = []
    for i in img:
        if type(i) == int:
            i = [i, i, i]
        input_.extend([i[0]/255, i[1]/255, i[2]/255])

    screen.fill([0, 0, 0])
    output_ = network.feed_forward(input_).values()
    network.back_propagate(input_)

    pg.image.save(screen, "test.png")
    return output_


def save(file_name: str) -> None:
    """
    Save the neural network as a file, with all its weights and biases intact
    """
    with open(f"{file_name}.pkl", "wb") as file:
        pickle.dump(network, file, pickle.HIGHEST_PROTOCOL)


def load(file_name: str) -> Network:
    """
    Load the neural network from a file, with all its weights and biases intact
    """
    with open(f"{file_name}.pkl", "rb") as file:
        network_ = pickle.load(file)
        return network_


network = Network([Layer(3072), Layer(100), Layer(10), Layer(100), Layer(3072)])
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0, 0, 0))

    inputs = [random.random() for x in range(10)]
    output = learn()
    display(output, 32)

    pg.display.update()
    clock.tick(fps)
