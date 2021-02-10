import pygame as pg
from artificial_nn import Network, Layer
import random
from typing import List

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


network = Network([Layer(10), Layer(100), Layer(1024)])
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0, 0, 0))

    inputs = [random.random() for x in range(10)]
    output = network.feed_forward(inputs).values()
    display(output, 32)

    pg.display.update()
    clock.tick(fps)
