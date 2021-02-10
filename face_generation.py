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

def display(pixels: List[float]) -> None:
    n = int(min(W, H) / 32)
    for i in range(0, len(pixels), 3):
        y = int(i / (3 * 32))
        x = (i - (3 * 32) * y) / 3
        pg.draw.rect(screen, (int(pixels[i] * 255), int(pixels[i+1] * 255),
                              int(pixels[i+2] * 255)), (int(x * n), int(y * n),
                                                        n, n), 0)

network = Network([Layer(10), Layer(100), Layer(3072)])
running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0, 0, 0))

    inputs = [random.random() for x in range(10)]
    output = network.feed_forward(inputs).values()
    display(output)

    pg.display.update()
    clock.tick(fps)
