import pygame as pg

W = 200
H = 200
screen = pg.display.set_mode((W, H))
pg.display.set_caption("face generation")
clock = pg.time.Clock()
fps = 60

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    screen.fill((0, 0, 0))

    ...

    pg.display.update()
    clock.tick(fps)
