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
import imageio


env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, base_config)
'''
model.load("results/world_model_weights_10_100.pth")
reward = 0
state = model.reset()
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
        print("W")
        action = 2
    elif keys[pyglet.window.key.D]:
        print("D")
        action = 3

    reward = 0
    for _ in range(1):
        _, r, _ = model.step(action)
    reward += r
    model.env.render()
    # print(reward)

    current_frame = pyglet.image.ImageData(400, 400, 'rgb', Image.fromarray((model.render() * 255).astype(np.uint8)).resize((400, 400), Image.NEAREST).tobytes())


@win.event
def on_draw():
    win.clear()
    current_frame.blit(0, 0)


pyglet.clock.schedule_interval(update, 0.1)
pyglet.app.run()
'''
model.load("results_baseline/world_model_weights_10_100.pth")
# agent = HumanAgent()

state = model.reset()
reward = 0
done = False
with imageio.get_writer("test_out.mp4", fps=2) as video:
    for i in range(100):
        # print(i)
        action = agent.act(state, reward, done)
        # print(action)
        model.env.step(action)
        env_state = model.env.render()
        reward = 0
        for _ in range(1):
            _, reward, _ = model.step(action)
        next_state = model.render()

        video.append_data(model.render())
        state = next_state

env.close()
