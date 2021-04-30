from config.baseline import config as base_config
from config.single_frame import config as sf_config
import torch

from models.world.world_model import World_Model
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent

import matplotlib.pyplot as plt
from gym import wrappers
import gym
import numpy as np

import pygame
from pygame.locals import *
from PIL import Image, ImageTk
import time

env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, base_config)

model.load("results/world_model_weights_7_100.pth")
#agent = HumanAgent()
reward = 0
state = model.reset()
env.close()
done = False

pygame.init()

FramePerSec = pygame.time.Clock()
 
displaysurface = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Simulated Environment")

crashed = False

def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

while not crashed:
    #print("RUNNING")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True

    action = agent.act(state, reward, done)

    reward = 0
    for _ in range(1):
        _, r, _ = model.step(action)
    reward+=r
    next_state = model.render()
    state = next_state

    displaysurface.fill((0,0,0))
    #img = Image.fromarray((model.render() * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST)
    img = pygame.image.load('results/plt1.png')
    #displaysurface.blit(pilImageToSurface(img),(0,0))
    displaysurface.blit(img,(0,0))
    #displaysurface.blit(,(0,0))

    pygame.display.update()
    FramePerSec.tick(60)


