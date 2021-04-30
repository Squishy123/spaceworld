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

import pyglet
from PIL import Image, ImageTk
import time

env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, base_config)

model.load("results/world_model_weights_10_100.pth")
reward = 0
state = model.reset()
env.close()
done = False

win = pyglet.window.Window(width=400, height=400)
keys = pyglet.window.key.KeyStateHandler()
win.push_handlers(keys)
current_frame = pyglet.image.ImageData(400, 400, 'rgb', Image.fromarray((model.render() * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST).tobytes())

def update(dt):
    action = 0
    if keys[pyglet.window.key.A]:
        print("A")
        action = 1
    elif keys[pyglet.window.key.W]:
        action = 2
    elif keys[pyglet.window.key.W]:
        action = 3

    reward = 0
    for _ in range(1):
        _, r, _ = model.step(action)
    reward+=r
    next_state = model.render()
    state = next_state

    current_frame = pyglet.image.ImageData(400, 400, 'rgb', Image.fromarray((model.render() * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST).tobytes())

@win.event
def on_draw():
    win.clear()
    current_frame.blit(0, 0)

pyglet.clock.schedule_interval(update, 0.1)
pyglet.app.run()


