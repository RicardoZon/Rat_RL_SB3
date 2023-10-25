import pygame
from pygame.locals import *
import sys

pygame.init()
display = pygame.display.set_mode((300, 300))

while 1:
    for event in pygame.event.get():
        if event.type == QUIT:  # QUIT
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                print("a键按下")
            if event.key == pygame.K_s:
                print("s键按下")
            if event.key == pygame.K_d:
                print("d键按下")

        # if event.type == pygame.KEYUP:
        #     if event.key == pygame.K_a:
        #         print("a键释放")
