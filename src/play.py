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


env = gym.make('LunarLander-v2')
agent = RandomAgent(env.action_space)
model = World_Model(env, agent, base_config)

model.load("results/world_model_weights_1_60.pth")
agent = HumanAgent()
state = model.reset()
reward = 0
done = False

while True:
    action = agent.act(state, reward, done)
    state, reward, done, _ = model.step(action)

    plt.imshow(model.render())
    plt.draw()
    plt.pause(1e-5)

    if done:
        break
